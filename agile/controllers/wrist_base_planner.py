"""Kinematic base placement for a world-space wrist trajectory (no simulator imports).

The base can translate on a plane and turn. Height, balance, contacts and obstacle
avoidance remain outside this planner; a kinematic solution is not a dynamics proof.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def wrap_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def heading(quaternion):
    x, y, z, w = quaternion
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class ArmKinematics:
    """Exact serial-chain FK calibrated from a reference pose and geometric Jacobian.

    All calibration quantities are in the pelvis frame. Space screws at the
    reference configuration give product-of-exponentials FK at other arm angles.
    Waist configuration is frozen in this placement model, not in the simulator.
    """

    def __init__(self, joint_positions, wrist_pose, jacobian, limits, nominal):
        self.reference = np.asarray(joint_positions, dtype=float).copy()
        self.pose = np.asarray(wrist_pose, dtype=float).copy()
        jacobian = np.asarray(jacobian, dtype=float)
        self.axes = jacobian[3:].T.copy()
        self.points = self.pose[:3] + np.cross(self.axes, jacobian[:3].T)
        self.limits = np.asarray(limits, dtype=float).copy()
        self.nominal = np.clip(nominal, self.limits[:, 0] + 0.03, self.limits[:, 1] - 0.03)
        if self.reference.shape != (7,) or jacobian.shape != (6, 7):
            raise ValueError("Expected seven arm joints and a 6x7 geometric Jacobian")
        if not np.allclose(np.linalg.norm(self.axes, axis=1), 1, atol=1e-4):
            raise ValueError("Arm Jacobian must contain unit revolute-joint axes")

    def forward(self, joints):
        matrix = np.eye(3)
        translation = np.zeros(3)
        for angle, axis, point in zip(np.asarray(joints) - self.reference, self.axes, self.points, strict=True):
            rotation = Rotation.from_rotvec(axis * angle).as_matrix()
            translation += matrix @ (point - rotation @ point)
            matrix = matrix @ rotation
        return (
            translation + matrix @ self.pose[:3],
            Rotation.from_matrix(matrix) * Rotation.from_quat(self.pose[3:]),
        )

    def as_dict(self):
        return {
            "reference_joints_rad": self.reference.tolist(),
            "reference_wrist_pose_pelvis_xyzw": self.pose.tolist(),
            "axes_pelvis": self.axes.tolist(),
            "axis_points_pelvis_m": self.points.tolist(),
            "joint_limits_rad": self.limits.tolist(),
            "nominal_joints_rad": self.nominal.tolist(),
        }


@dataclass
class BasePlacement:
    xy_yaw: np.ndarray
    arm_joints: np.ndarray
    position_residual: float
    orientation_residual: float


class WristBasePlanner:
    """Choose base XY/yaw and a feasible arm posture, then follow with bounded velocity.

    Pose constraints dominate weak posture regularization. XY placement is solved
    exactly from wrist translation, so lateral/backward paths need no special mode.
    Rotation/height are solved jointly with seven arm angles and base heading.
    A preview supplies feedforward; measured base pose supplies position feedback.
    """

    def __init__(self, model, base_height, initial_heading, preview_s=0.3):
        self.model = model
        self.base_height = float(base_height)
        self.preview_s = preview_s
        self.last = np.r_[initial_heading, model.nominal]
        self.placement = None
        self.feedforward = np.zeros(3)
        self.command_limited = False
        self.max_velocity = np.array([0.30, 0.22, 0.35])
        self.max_reverse_velocity = 0.25
        self.feedback_gain = np.array([1.5, 1.5, 1.5])
        self.velocity_damping = np.array([0.45, 0.45, 0.4])
        self.filtered_velocity = np.zeros(3)
        self.previous_base = None
        self.update_period_s = 0.1
        self.elapsed = np.inf
        self.nominal_wrist_yaw = heading(model.forward(model.nominal)[1].as_quat())

    def solve(self, target, seed=None):
        target = np.asarray(target, dtype=float)
        if target.shape != (7,) or not np.isfinite(target).all() or abs(np.linalg.norm(target[3:]) - 1) > 1e-3:
            raise ValueError("Expected finite world wrist pose xyz + unit quaternion xyzw")
        seed = self.last if seed is None else np.asarray(seed)
        target_rotation = Rotation.from_quat(target[3:])
        # Near a vertical viewing direction, yaw is poorly defined; retain the seed.
        forward = target_rotation.as_matrix()[:2, 0]
        preferred = seed[0]
        if np.linalg.norm(forward) > 0.15:
            preferred += wrap_angle(np.arctan2(forward[1], forward[0]) - self.nominal_wrist_yaw - preferred)

        def residual(x):
            arm_position, arm_rotation = self.model.forward(x[1:])
            world_rotation = Rotation.from_rotvec([0, 0, x[0]]) * arm_rotation
            return np.r_[
                self.base_height + arm_position[2] - target[2],
                0.3 * (target_rotation * world_rotation.inv()).as_rotvec(),
                0.008 * (x[1:] - self.model.nominal),
                0.012 * wrap_angle(x[0] - preferred),
            ]

        lower = np.r_[seed[0] - np.pi, self.model.limits[:, 0] + 0.03]
        upper = np.r_[seed[0] + np.pi, self.model.limits[:, 1] - 0.03]
        result = least_squares(residual, np.clip(seed, lower + 1e-6, upper - 1e-6), bounds=(lower, upper), max_nfev=45)
        x = result.x
        arm_position, arm_rotation = self.model.forward(x[1:])
        yaw_rotation = Rotation.from_rotvec([0, 0, x[0]])
        position_residual = abs(self.base_height + arm_position[2] - target[2])
        orientation_residual = np.linalg.norm((target_rotation * (yaw_rotation * arm_rotation).inv()).as_rotvec())
        if position_residual > 0.025 or orientation_residual > 0.12:
            raise ValueError(
                "No feasible local wrist/base placement at the walking policy's fixed height: "
                f"position residual {position_residual:.3f} m, orientation residual {orientation_residual:.3f} rad"
            )
        base_xy = target[:2] - yaw_rotation.apply(arm_position)[:2]
        return BasePlacement(np.r_[base_xy, x[0]], x[1:], position_residual, orientation_residual)

    def command(self, target, preview, measured_base_pose, dt):
        actual_yaw = heading(measured_base_pose[3:])
        actual = np.r_[measured_base_pose[:2], actual_yaw]
        if self.previous_base is not None:
            displacement = actual - self.previous_base
            displacement[2] = wrap_angle(displacement[2])
            alpha = dt / (0.2 + dt)
            self.filtered_velocity += alpha * (displacement / dt - self.filtered_velocity)
        self.previous_base = actual
        self.elapsed += dt
        if self.elapsed >= self.update_period_s - 1e-8:
            self.placement = self.solve(target)
            self.last = np.r_[self.placement.xy_yaw[2], self.placement.arm_joints]
            future = self.solve(preview, seed=self.last)
            change = future.xy_yaw - self.placement.xy_yaw
            change[2] = wrap_angle(change[2])
            self.feedforward = change / self.preview_s
            self.elapsed = 0.0
        desired = self.placement.xy_yaw + self.elapsed * self.feedforward
        error = desired - actual
        error[2] = wrap_angle(error[2])
        velocity = (
            self.feedforward
            + self.feedback_gain * error
            - self.velocity_damping * (self.filtered_velocity - self.feedforward)
        )
        c, s = np.cos(actual_yaw), np.sin(actual_yaw)
        command = np.r_[c * velocity[0] + s * velocity[1], -s * velocity[0] + c * velocity[1], velocity[2]]
        limited = np.clip(command, -self.max_velocity, self.max_velocity)
        limited[0] = max(limited[0], -self.max_reverse_velocity)
        # Do not ask the learned gait for maximal longitudinal and lateral speed
        # simultaneously. Reverse walking has a smaller validated speed envelope.
        longitudinal = self.max_reverse_velocity if limited[0] < 0 else self.max_velocity[0]
        scale = max(1.0, np.linalg.norm(limited[:2] / [longitudinal, self.max_velocity[1]]))
        limited[:2] /= scale
        self.command_limited = bool(np.any(limited != command))
        return limited

    def as_dict(self):
        return {
            "type": "kinematic_wrist_to_base_xy_yaw",
            "base_height_m": self.base_height,
            "preview_s": self.preview_s,
            "update_period_s": self.update_period_s,
            "max_base_velocity": self.max_velocity.tolist(),
            "max_reverse_velocity_mps": self.max_reverse_velocity,
            "position_heading_gain_per_s": self.feedback_gain.tolist(),
            "velocity_damping": self.velocity_damping.tolist(),
            "velocity_filter_time_s": 0.2,
            "translation_speed_envelope": "elliptical",
            "model": self.model.as_dict(),
        }
