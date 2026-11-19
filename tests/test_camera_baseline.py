"""CPU tests for world trajectories, quaternion conventions, and singular IK solves."""

import math

import pytest
import torch

from agile.controllers.camera_reference import demo_pose, level_recording_pose
from agile.controllers.pose_trajectory import WorldPoseTrajectory, damped_solve, nullspace_posture_step, rotation_error


def test_world_pose_endpoints_and_out_of_range_hold():
    trajectory = WorldPoseTrajectory([0, 2], [[1, 2, 3], [3, 2, 4]], [[0, 0, 0, 1], [0, 0, 1, 0]])
    pose, twist = trajectory.sample([-1, 0, 2, 3])
    torch.testing.assert_close(pose[:, :3], torch.tensor([[1.0, 2, 3], [1, 2, 3], [3, 2, 4], [3, 2, 4]]))
    torch.testing.assert_close(twist, torch.zeros(4, 6))


def test_world_twist_matches_pose_finite_difference():
    trajectory = WorldPoseTrajectory(
        [0, 2], [[0, 0, 0], [1, 2, 3]], [[0, 0, 0, 1], [0, math.sin(0.4), 0, math.cos(0.4)]]
    )
    pose, twist = trajectory.sample([0.8])
    next_pose, _ = trajectory.sample([0.8001])
    numerical = torch.cat(
        ((next_pose[:, :3] - pose[:, :3]) / 0.0001, rotation_error(next_pose[:, 3:], pose[:, 3:]) / 0.0001), -1
    )
    torch.testing.assert_close(numerical, twist, atol=0.003, rtol=0.003)


def test_opposite_quaternion_signs_do_not_spin():
    trajectory = WorldPoseTrajectory([0, 1], [[0, 0, 0], [0, 0, 0]], [[0, 0, 0, 1], [0, 0, 0, -1]])
    pose, twist = trajectory.sample([0.2, 0.5, 0.8])
    torch.testing.assert_close(pose[:, 3:], torch.tensor([[0.0, 0, 0, 1]]).expand(3, -1))
    torch.testing.assert_close(twist, torch.zeros(3, 6))


@pytest.mark.parametrize(
    "times,quats", [([0, 0], [[0, 0, 0, 1]] * 2), ([0, 1], [[0, 0, 0, 0]] * 2), ([1, 2], [[0, 0, 0, 1]] * 2)]
)
def test_bad_planner_inputs_rejected(times, quats):
    with pytest.raises(ValueError):
        WorldPoseTrajectory(times, [[0, 0, 0]] * 2, quats)


def test_damped_solve_handles_singularity_and_batching():
    jacobian = torch.zeros(2, 6, 7)
    jacobian[0, :, :6] = torch.eye(6)
    twist = torch.ones(2, 6)
    solution = damped_solve(jacobian, twist, 0.1)
    torch.testing.assert_close(solution[0, :6], torch.full((6,), 1 / 1.01))
    torch.testing.assert_close(solution[1], torch.zeros(7))
    assert torch.isfinite(solution).all()


def test_known_world_orientation_error_is_xyzw():
    identity = torch.tensor([[0.0, 0, 0, 1]])
    target = torch.tensor([[0.0, 0, math.sin(math.pi / 4), math.cos(math.pi / 4)]])
    torch.testing.assert_close(rotation_error(target, identity), torch.tensor([[0.0, 0, math.pi / 2]]))


def test_damped_posture_projection_limits_task_leakage_and_uses_redundancy():
    jacobian = torch.zeros(2, 6, 7)
    jacobian[0, :, :6] = torch.eye(6)
    requested = torch.full((2, 7), 0.05)
    step = nullspace_posture_step(jacobian, requested)
    assert torch.isfinite(step).all()
    assert (jacobian @ step.unsqueeze(-1)).norm() < 2e-5
    torch.testing.assert_close(step[:, 6], requested[:, 6])
    torch.testing.assert_close(step[1], requested[1])


def test_posture_regularized_ik_satisfies_optimality_and_keeps_free_joint_near_prior():
    rng = torch.Generator().manual_seed(8)
    jacobian = torch.randn(2, 6, 7, generator=rng, dtype=torch.float64)
    jacobian[:, :, 6] = 0
    task = torch.randn(2, 6, generator=rng, dtype=torch.float64) * 0.02
    prior = torch.randn(2, 7, generator=rng, dtype=torch.float64) * 0.1
    damping = 0.05
    step = damped_solve(jacobian, task, damping) + nullspace_posture_step(jacobian, prior, damping)
    residual = (jacobian @ step.unsqueeze(-1)).squeeze(-1) - task
    gradient = (jacobian.transpose(-1, -2) @ residual.unsqueeze(-1)).squeeze(-1)
    gradient += damping**2 * (step - prior)
    torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=1e-12, rtol=0)
    torch.testing.assert_close(step[:, 6], prior[:, 6])


def test_recording_pose_is_level_at_fixed_height_and_respects_initial_heading():
    # The base has moved, turned 90 degrees, and pitched 20 degrees. Only its
    # heading should affect the target; body lean must not tilt the horizon.
    half_pitch = math.radians(10)
    s = math.sqrt(0.5)
    root = torch.tensor(
        [
            [
                2.0,
                3.0,
                0.73,
                -s * math.sin(half_pitch),
                s * math.sin(half_pitch),
                s * math.cos(half_pitch),
                s * math.cos(half_pitch),
            ]
        ]
    )
    pose = level_recording_pose(root)
    torch.testing.assert_close(pose[:, :3], torch.tensor([[2.25, 3.30, 0.95]]))
    torch.testing.assert_close(pose[:, 3:], torch.tensor([[0.0, 0, s, s]]))
    root[:, :3] += 1
    torch.testing.assert_close(pose[:, :3], torch.tensor([[2.25, 3.30, 0.95]]))


@pytest.mark.parametrize("mode", ["hold", "sweep"])
def test_recording_demo_preserves_level_horizon_and_small_lateral_vertical_motion(mode):
    initial = level_recording_pose(torch.tensor([[0.0, 0, 0.78, 0, 0, 0, 1]]))
    for t in torch.linspace(0, 20, 101).tolist():
        pose = demo_pose(t, initial, mode)
        torch.testing.assert_close(pose[:, 3:5], torch.zeros(1, 2))
        torch.testing.assert_close(pose[:, 3:].norm(dim=-1), torch.ones(1))
        assert -0.25001 <= pose[0, 1] <= -0.20999
        assert 0.94999 <= pose[0, 2] <= 0.99001
        assert rotation_error(pose[:, 3:], initial[:, 3:]).norm() <= 0.10001
