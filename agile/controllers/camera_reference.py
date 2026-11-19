"""Level, world-frame recording references for the unloaded G1 demo."""

import torch

from .pose_trajectory import quat_multiply, rotation_error


def maneuver_waypoints(initial_pose, duration_s):
    """A closed wrist-only path: forward, lateral, backward, return, plus 3D rotation."""
    import numpy as np
    from scipy.spatial.transform import Rotation

    initial = initial_pose[0].cpu().numpy()
    offsets = np.array([[0, 0, 0], [0.6, 0, 0.02], [0.6, 0.5, 0.04], [0, 0.5, 0.02], [0, 0, 0]])
    angles = [[0, 0, 0], [0.04, 0.06, 0.1], [-0.05, 0.10, 0.55], [0.05, -0.06, 0.55], [0, 0, 0]]
    rotations = Rotation.from_euler("xyz", angles) * Rotation.from_quat(initial[3:])
    return {
        "frame": "world",
        "quaternion_order": "xyzw",
        "times_s": np.linspace(0, duration_s, 5).tolist(),
        "positions_m": (offsets + initial[:3]).tolist(),
        "quaternions_xyzw": rotations.as_quat().tolist(),
    }


def level_recording_pose(root_pose_w: torch.Tensor) -> torch.Tensor:
    """Anchor once: 30 cm forward, 25 cm right of the base, 95 cm above the floor.

    The wrist's +X points forward horizontally, +Y left, and +Z up. This is a
    provisional recording-frame convention until the camera extrinsic is known.
    Only the initial base heading is used; the returned target stays in world space.
    """
    x, y, z, w = root_pose_w[:, 3:].unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    pose = torch.zeros_like(root_pose_w)
    pose[:, 0] = root_pose_w[:, 0] + 0.30 * yaw.cos() + 0.25 * yaw.sin()
    pose[:, 1] = root_pose_w[:, 1] + 0.30 * yaw.sin() - 0.25 * yaw.cos()
    pose[:, 2] = 0.95
    pose[:, 5] = torch.sin(yaw / 2)
    pose[:, 6] = torch.cos(yaw / 2)
    return pose


def demo_pose(t, initial_pose, mode):
    """Hold or gently sweep around a level world pose."""
    if mode not in ("hold", "sweep"):
        raise ValueError(f"Unsupported local recording reference: {mode}")
    pose = initial_pose.clone()
    if mode == "hold":
        return pose
    phase = torch.full((len(pose),), t * 0.6, device=pose.device, dtype=pose.dtype)
    pose[:, 1] += 0.02 * (1 - torch.cos(phase))
    pose[:, 2] += 0.02 * (1 - torch.cos(phase * 0.8))
    pose[:, 0] += 0.025 * (1 - torch.cos(phase))
    # Keep the horizon level: intentional panning about world Z only.
    half_yaw = 0.05 * torch.sin(phase * 0.9)
    delta = torch.zeros_like(pose[:, 3:])
    delta[:, 2], delta[:, 3] = half_yaw.sin(), half_yaw.cos()
    pose[:, 3:] = quat_multiply(delta, initial_pose[:, 3:])
    return pose


def demo_reference(t, initial_pose, mode):
    pose = demo_pose(t, initial_pose, mode)
    next_pose = demo_pose(t + 0.001, initial_pose, mode)
    twist = torch.cat(
        ((next_pose[:, :3] - pose[:, :3]) / 0.001, rotation_error(next_pose[:, 3:], pose[:, 3:]) / 0.001), -1
    )
    return pose, twist
