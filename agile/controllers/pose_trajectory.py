"""Simulator-independent world-pose trajectories and bounded damped IK helpers.

All quaternions use Isaac Lab 3's (x, y, z, w) convention.
"""

import json
from pathlib import Path

import torch


def quat_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    av, bv = a[..., :3], b[..., :3]
    return torch.cat(
        (
            a[..., 3:] * bv + b[..., 3:] * av + torch.linalg.cross(av, bv),
            a[..., 3:] * b[..., 3:] - (av * bv).sum(-1, keepdim=True),
        ),
        -1,
    )


def rotation_error(target: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
    """Shortest rotation vector from actual to target, expressed in world axes."""
    inverse = torch.cat((-actual[..., :3], actual[..., 3:]), -1)
    delta = quat_multiply(target, inverse)
    delta = torch.where(delta[..., 3:] < 0, -delta, delta)
    norm = delta[..., :3].norm(dim=-1, keepdim=True)
    angle = 2 * torch.atan2(norm, delta[..., 3:].clamp_min(0))
    return delta[..., :3] * torch.where(norm > 1e-7, angle / norm.clamp_min(1e-7), 2.0)


def damped_solve(jacobian: torch.Tensor, twist: torch.Tensor, damping: float) -> torch.Tensor:
    """Damped least-squares solve; finite even for a rank-deficient Jacobian."""
    if damping <= 0:
        raise ValueError("IK damping must be positive")
    jt = jacobian.transpose(-1, -2)
    eye = torch.eye(jacobian.shape[-2], device=jacobian.device, dtype=jacobian.dtype)
    return (jt @ torch.linalg.solve(jacobian @ jt + damping**2 * eye, twist.unsqueeze(-1))).squeeze(-1)


def nullspace_posture_step(jacobian: torch.Tensor, joint_step: torch.Tensor, damping: float = 0.01) -> torch.Tensor:
    """Apply a damped null-space projection to a joint-posture correction.

    Damping keeps the projection finite near singularities but allows some task
    motion. Adding this to a task solve with the SAME damping is equivalent to
    regularizing that solve toward joint_step instead of toward zero joint motion.
    """
    task_step = (jacobian @ joint_step.unsqueeze(-1)).squeeze(-1)
    return joint_step - damped_solve(jacobian, task_step, damping)


class WorldPoseTrajectory:
    """One absolute world-frame path, sampled for one or more query times.

    Minimum-jerk position interpolation and shortest-arc quaternion interpolation
    stop at every waypoint. Position and orientation have zero velocity and
    acceleration at knots. This is a baseline waypoint interface, not a replacement
    for a planner's continuous velocity/acceleration-aware spline.
    """

    def __init__(self, times, positions, quaternions, device="cpu"):
        self.times = torch.as_tensor(times, dtype=torch.float32, device=device)
        self.positions = torch.as_tensor(positions, dtype=torch.float32, device=device)
        self.quaternions = torch.as_tensor(quaternions, dtype=torch.float32, device=device)
        n = self.times.numel()
        if self.times.ndim != 1 or n < 2 or self.positions.shape != (n, 3) or self.quaternions.shape != (n, 4):
            raise ValueError("Expected times [K], positions [K,3], quaternions_xyzw [K,4], K >= 2")
        if not all(torch.isfinite(x).all() for x in (self.times, self.positions, self.quaternions)):
            raise ValueError("Trajectory must contain only finite values")
        if self.times[0] != 0 or not torch.all(self.times[1:] > self.times[:-1]):
            raise ValueError("Times must start at zero and be strictly increasing")
        norms = self.quaternions.norm(dim=-1, keepdim=True)
        if not torch.allclose(norms, torch.ones_like(norms), atol=1e-3, rtol=0):
            raise ValueError("Quaternions must be unit length and use xyzw order")
        self.quaternions = self.quaternions / norms

    @classmethod
    def from_json(cls, path: str | Path, device="cpu"):
        with open(path) as stream:
            data = json.load(stream)
        if data.get("frame") != "world" or data.get("quaternion_order") != "xyzw":
            raise ValueError("Trajectory JSON must declare frame='world' and quaternion_order='xyzw'")
        return cls(data["times_s"], data["positions_m"], data["quaternions_xyzw"], device)

    def sample(self, time_s) -> tuple[torch.Tensor, torch.Tensor]:
        """Return pose [N,7] and world twist [N,6]; hold endpoints outside the path."""
        t = torch.as_tensor(time_s, device=self.times.device, dtype=self.times.dtype).reshape(-1)
        t = t.clamp(self.times[0], self.times[-1])
        i = (torch.searchsorted(self.times, t, right=True) - 1).clamp(0, len(self.times) - 2)
        duration = self.times[i + 1] - self.times[i]
        u = (t - self.times[i]) / duration
        s = (10 * u**3 - 15 * u**4 + 6 * u**5).unsqueeze(-1)
        ds = (30 * u**2 * (1 - u) ** 2 / duration).unsqueeze(-1)
        displacement = self.positions[i + 1] - self.positions[i]
        position = self.positions[i] + s * displacement
        q0, q1 = self.quaternions[i], self.quaternions[i + 1]
        dot = (q0 * q1).sum(-1, keepdim=True)
        q1 = torch.where(dot < 0, -q1, q1)
        dot = dot.abs().clamp(0, 1)
        angle = torch.acos(dot)
        denominator = torch.sin(angle).clamp_min(1e-7)
        quaternion = (torch.sin((1 - s) * angle) * q0 + torch.sin(s * angle) * q1) / denominator
        quaternion = torch.where(dot > 0.9995, (1 - s) * q0 + s * q1, quaternion)
        quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True)
        twist = torch.cat((ds * displacement, ds * rotation_error(q1, q0)), -1)
        return torch.cat((position, quaternion), -1), twist
