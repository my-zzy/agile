"""Kinematic and command-frame checks without Isaac Sim."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from agile.controllers.wrist_base_planner import ArmKinematics, WristBasePlanner, wrap_angle


def serial_arm(q):
    axes = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    offsets = np.array(
        [[0, -0.15, 0.3], [0, -0.04, 0], [0, 0, -0.08], [0, 0, -0.15], [0.1, 0, -0.12], [0.04, 0, 0], [0.03, 0, 0]]
    )
    p, r = np.zeros(3), Rotation.identity()
    for angle, axis, offset in zip(q, axes, offsets, strict=True):
        p += r.apply(offset)
        r = r * Rotation.from_rotvec(angle * axis)
    return p, r


@pytest.fixture
def model():
    q = np.array([-0.6, -0.3, 0.1, 0.5, 0.2, 0.3, 0.1])
    p, r = serial_arm(q)
    jac = np.empty((6, 7))
    for j in range(7):
        delta = np.eye(7)[j] * 1e-6
        pp, rp = serial_arm(q + delta)
        pm, rm = serial_arm(q - delta)
        jac[:, j] = np.r_[(pp - pm) / 2e-6, (rp * rm.inv()).as_rotvec() / 2e-6]
    return ArmKinematics(q, np.r_[p, r.as_quat()], jac, np.tile([-2.5, 2.5], (7, 1)), q)


def test_calibrated_fk_matches_independent_chain_away_from_calibration(model):
    for q in np.random.default_rng(42).uniform(-1, 1, (20, 7)):
        actual_p, actual_r = model.forward(q)
        expected_p, expected_r = serial_arm(q)
        np.testing.assert_allclose(actual_p, expected_p, atol=1e-8)
        assert np.linalg.norm((actual_r * expected_r.inv()).as_rotvec()) < 1e-8


@pytest.mark.parametrize("xy,yaw", [([1, 0], 0), ([-1, 0], 0), ([0, 1], 0), ([0, -1], 0), ([0.4, -0.6], 1.5)])
def test_reachable_wrist_targets_generate_base_placements_in_every_direction(model, xy, yaw):
    planner = WristBasePlanner(model, 0.72, 0)
    arm_p, arm_r = model.forward(model.nominal)
    rotation = Rotation.from_rotvec([0, 0, yaw])
    target = np.r_[np.r_[xy, 0.72] + rotation.apply(arm_p), (rotation * arm_r).as_quat()]
    placement = planner.solve(target)
    p, r = model.forward(placement.arm_joints)
    base_r = Rotation.from_rotvec([0, 0, placement.xy_yaw[2]])
    actual = np.r_[placement.xy_yaw[:2], 0.72] + base_r.apply(p)
    np.testing.assert_allclose(actual, target[:3], atol=0.005)
    assert np.linalg.norm((Rotation.from_quat(target[3:]) * (base_r * r).inv()).as_rotvec()) < 0.025
    np.testing.assert_allclose(placement.xy_yaw[:2], xy, atol=0.025)


def test_command_uses_measured_heading_and_holds_a_fixed_world_position(model):
    planner = WristBasePlanner(model, 0.72, np.pi / 2)
    p, r = model.forward(model.nominal)
    yaw = Rotation.from_rotvec([0, 0, np.pi / 2])
    target = np.r_[np.array([0, 0, 0.72]) + yaw.apply(p), (yaw * r).as_quat()]
    root = np.r_[0.1, 0, 0.72, yaw.as_quat()]
    command = planner.command(target, target, root, 0.02)
    # Robot faces +world Y. Correcting a +world X drift means +body Y.
    assert abs(command[0]) < 0.015
    assert command[1] > 0.08
    assert abs(command[2]) < 0.03


def test_impossible_height_is_rejected_instead_of_projecting_target(model):
    planner = WristBasePlanner(model, 0.72, 0)
    with pytest.raises(ValueError, match="No feasible local"):
        planner.solve([0.3, -0.25, 3, 0, 0, 0, 1])


def test_quaternion_sign_and_heading_wrap_do_not_reverse_turn(model):
    planner = WristBasePlanner(model, 0.72, np.pi - 0.01)
    p, r = model.forward(model.nominal)
    rotation = Rotation.from_rotvec([0, 0, -np.pi + 0.01])
    target = np.r_[np.array([0, 0, 0.72]) + rotation.apply(p), (rotation * r).as_quat()]
    a = planner.solve(target)
    target[3:] *= -1
    b = planner.solve(target)
    np.testing.assert_allclose(a.xy_yaw, b.xy_yaw, atol=1e-5)
    assert abs(wrap_angle(a.xy_yaw[2] - (np.pi - 0.01))) < 0.05


def test_combined_reverse_lateral_command_respects_gait_envelope(model):
    planner = WristBasePlanner(model, 0.72, 0)
    p, r = model.forward(model.nominal)
    target = np.r_[np.array([-1, -1, 0.72]) + p, r.as_quat()]
    root = np.array([0, 0, 0.72, 0, 0, 0, 1])
    command = planner.command(target, target, root, 0.02)
    assert command[0] < 0 and command[1] < 0
    assert np.linalg.norm(command[:2] / [0.25, 0.22]) <= 1 + 1e-8
    assert planner.command_limited
