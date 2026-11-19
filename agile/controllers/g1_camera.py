"""Pretrained G1 locomotion plus world-frame, right-wrist differential IK.

Import this module only after creating an Isaac Lab AppLauncher.
"""

import re
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from isaaclab.managers import CommandTerm, CommandTermCfg, EventTermCfg
from isaaclab.utils.configclass import configclass

from agile.controllers.pose_trajectory import damped_solve, nullspace_posture_step, rotation_error
from agile.controllers.wrist_base_planner import ArmKinematics, WristBasePlanner
from agile.rl_env import mdp
from agile.rl_env.rsl_rl_observations import flatten_observation_group
from agile.rl_env.tasks.locomotion.g1.velocity_history_env_cfg import (
    CONTROLLED_JOINT_NAMES,
    G1LowerVelocityHistoryEnvCfg,
)

RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parents[1] / "data/policy/velocity_g1/unitree_g1_velocity_history_torchscript.pt"
)


class ExternalVelocityCommand(CommandTerm):
    """Explicit base-frame [vx, vy, yaw_rate], never overwritten by a random sampler."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._command = torch.zeros(env.num_envs, 3, device=env.device)

    @property
    def command(self):
        return self._command

    def _resample_command(self, env_ids):
        self._command[env_ids] = 0

    def _update_command(self):
        pass

    def _update_metrics(self):
        pass


@configclass
class ExternalVelocityCommandCfg(CommandTermCfg):
    class_type: type = ExternalVelocityCommand
    resampling_time_range: tuple[float, float] = (1e9, 1e9)


@configclass
class ResetEventsCfg:
    reset_scene = EventTermCfg(func=mdp.reset_scene_to_default, mode="reset")


def make_camera_baseline_cfg(num_envs=1, device="cuda:0", duration_s=30.0, locomotion_policy=None):
    """Keep locomotion observations/scaling intact; add disjoint right-arm actions."""
    if num_envs != 1:
        raise ValueError(
            "The camera baseline currently supports one robot; parallel rollout stability is not validated"
        )
    cfg = G1LowerVelocityHistoryEnvCfg()
    cfg.eval()
    cfg.seed = 42
    cfg.scene.num_envs = num_envs
    cfg.sim.device = device
    cfg.episode_length_s = duration_s + 10.0
    cfg.events = ResetEventsCfg()
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    # Isaac Lab beta2's plane lacks the legacy material prims.
    cfg.scene.terrain.physics_material = None
    cfg.scene.terrain.visual_material = None
    cfg.scene.sky_light.spawn.texture_file = None
    # Keep the right wrist and target axes legible as the robot walks.
    cfg.viewer.origin_type = "asset_root"
    cfg.viewer.eye = (1.6, -2.0, 0.9)
    cfg.viewer.lookat = (0.15, -0.05, 0.10)
    cfg.scene.robot.init_state.pos = (0.0, 0.0, 0.78)
    cfg.scene.robot.init_state.joint_pos.update(
        {
            # Bent elbow, wrist in front of the chest, approximately level.
            # FK seed for (0.30, -0.25, +0.22) m relative to the upright pelvis.
            "right_shoulder_pitch_joint": -0.6284,
            "right_shoulder_roll_joint": -0.4951,
            "right_shoulder_yaw_joint": -0.16,
            "right_elbow_joint": 0.3135,
            "right_wrist_roll_joint": 0.3768,
            "right_wrist_pitch_joint": 0.3904,
            "right_wrist_yaw_joint": 0.1229,
            "left_shoulder_roll_joint": 0.15,
            "left_elbow_joint": 0.6,
        }
    )
    # The locomotion asset's 4 Nm/rad wrist gains sag noticeably in a horizontal
    # hold. Increase only the right wrist's gains; preserve its effort limits.
    arm_actuator = cfg.scene.robot.actuators["arms"]
    for gains, right_value in ((arm_actuator.stiffness, 12.0), (arm_actuator.damping, 0.6)):
        gains["left_wrist_.*_joint"] = gains.pop(".*_wrist_.*_joint")
        gains["right_wrist_.*_joint"] = right_value
    cfg.commands.base_velocity = ExternalVelocityCommandCfg()
    cfg.observations.policy.enable_corruption = False
    cfg.observations.policy.actions.params = {"action_name": "joint_pos"}
    cfg.actions.random_pos.joint_names_exclude = CONTROLLED_JOINT_NAMES + RIGHT_ARM_JOINTS
    cfg.actions.random_pos.randomize = False
    cfg.actions.right_arm = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=RIGHT_ARM_JOINTS,
        preserve_order=True,
        scale=1.0,
        use_default_offset=False,
    )
    if locomotion_policy is not None:
        if len(locomotion_policy.controlled_names) != 14 or not all(
            any(re.fullmatch(pattern, name) for pattern in CONTROLLED_JOINT_NAMES)
            for name in locomotion_policy.controlled_names
        ):
            raise ValueError("External locomotion policy must control only the same 14 lower-body joints")
        if not np.isclose(cfg.sim.dt * cfg.decimation, locomotion_policy.control_dt):
            raise ValueError("Simulation control period differs from the policy contract")
        # The deployment graph already applies its learned action scale and default pose.
        cfg.actions.joint_pos.joint_names = list(locomotion_policy.controlled_names)
        cfg.actions.joint_pos.preserve_order = True
        cfg.actions.joint_pos.scale = 1.0
        cfg.actions.joint_pos.use_default_offset = False
        cfg.actions.joint_pos.offset = 0.0
        cfg.actions.joint_pos.clip = None
    return cfg


class G1CameraController:
    """50 Hz layered baseline. Targets and twists are for right_wrist_yaw_link's origin.

    The frozen policy owns 12 leg joints plus waist roll/pitch. IK owns exactly
    seven right-arm joints. Left arm and waist yaw stay at their default posture.
    enable_base_tracking() adds kinematic base placement from the wrist target.
    Explicit base commands can override automatic placement for controlled comparisons.
    """

    def __init__(
        self,
        env,
        checkpoint=DEFAULT_CHECKPOINT,
        damping=0.05,
        gain=0.8,
        max_joint_speed=1.5,
        max_joint_acceleration=8.0,
        response_time=0.08,
        max_base_acceleration=(0.25, 0.25, 0.5),
        posture_gain=0.5,
        locomotion_policy=None,
    ):
        if min(damping, gain, max_joint_speed, max_joint_acceleration, response_time) <= 0:
            raise ValueError("Controller gains and bounds must be positive")
        if env.num_envs != 1:
            raise ValueError("G1CameraController is currently validated for one robot only")
        self.env = env
        self.robot = env.scene["robot"]
        self.locomotion_policy = locomotion_policy
        self.policy = None if locomotion_policy else torch.jit.load(str(checkpoint), map_location=env.device).eval()
        if locomotion_policy is not None:
            self.policy_joint_ids, names = self.robot.find_joints(locomotion_policy.joint_names, preserve_order=True)
            if names != locomotion_policy.joint_names:
                raise ValueError("Robot joint ordering does not match the locomotion contract")
            self._policy_gains_checked = False
        self.wrist_id = self.robot.find_bodies("right_wrist_yaw_link")[0][0]
        self.arm_ids, names = self.robot.find_joints(RIGHT_ARM_JOINTS, preserve_order=True)
        if names != RIGHT_ARM_JOINTS:
            raise ValueError(f"Unexpected right-arm joint ordering: {names}")
        self.damping, self.gain = damping, gain
        self.max_joint_speed, self.max_joint_acceleration = max_joint_speed, max_joint_acceleration
        self.response_time = response_time
        if not np.isfinite(posture_gain) or posture_gain < 0:
            raise ValueError("Posture gain must be finite and nonnegative")
        self.posture_gain = posture_gain
        self.max_base_acceleration = tuple(max_base_acceleration)
        if (
            len(self.max_base_acceleration) != 3
            or not np.isfinite(self.max_base_acceleration).all()
            or min(self.max_base_acceleration) <= 0
        ):
            raise ValueError("Expected three finite positive base command acceleration limits")
        self.dt = env.step_dt
        self.arm_target = self.robot.data.joint_pos.torch[:, self.arm_ids].clone()
        self.arm_velocity = torch.zeros_like(self.arm_target)
        self.base_velocity = torch.zeros(env.num_envs, 3, device=env.device)
        self.base_planner = None
        self.limit_saturated = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.action_slices = {}
        start = 0
        for name, size in zip(env.action_manager.active_terms, env.action_manager.action_term_dim, strict=True):
            self.action_slices[name] = slice(start, start + size)
            start += size
        if env.action_manager.get_term("joint_pos").action_dim != 14 or start != 21:
            raise ValueError("Expected 14 locomotion actions and 7 right-arm actions")

    @property
    def wrist_pose_w(self):
        return self.robot.data.body_link_pose_w.torch[:, self.wrist_id]

    @property
    def wrist_twist_w(self):
        return self.robot.data.body_link_vel_w.torch[:, self.wrist_id]

    @property
    def arm_jacobian_w(self):
        body_index = self.wrist_id - (1 if self.robot.is_fixed_base else 0)
        columns = [i + self.robot.num_base_dofs for i in self.arm_ids]
        return self.robot.data.body_link_jacobian_w.torch[:, body_index, :, columns]

    def enable_base_tracking(self):
        """Calibrate arm FK from the live asset after settling; no copied URDF needed."""
        root = self.robot.data.root_link_pose_w.torch[0].cpu().numpy()
        wrist = self.wrist_pose_w[0].cpu().numpy()
        rotation = Rotation.from_quat(root[3:])
        wrist_local = np.r_[
            rotation.inv().apply(wrist[:3] - root[:3]), (rotation.inv() * Rotation.from_quat(wrist[3:])).as_quat()
        ]
        jacobian = self.arm_jacobian_w[0].cpu().numpy()
        jacobian_local = np.vstack(
            (rotation.inv().as_matrix() @ jacobian[:3], rotation.inv().as_matrix() @ jacobian[3:])
        )
        model = ArmKinematics(
            self.robot.data.joint_pos.torch[0, self.arm_ids].cpu().numpy(),
            wrist_local,
            jacobian_local,
            self.robot.data.soft_joint_pos_limits.torch[0, self.arm_ids].cpu().numpy(),
            self.robot.data.default_joint_pos.torch[0, self.arm_ids].cpu().numpy(),
        )
        from agile.controllers.wrist_base_planner import heading

        self.base_planner = WristBasePlanner(model, root[2], heading(root[3:]))
        return self.base_planner

    def reset(self):
        self.arm_target.copy_(self.robot.data.joint_pos.torch[:, self.arm_ids])
        self.arm_velocity.zero_()
        self.base_velocity.zero_()
        self.base_planner = None

    @torch.inference_mode()
    def compute(
        self, target_pose_w, target_twist_w=None, base_velocity_b=None, *, arm_enabled=True, preview_pose_w=None
    ):
        """Return env actions. Inputs: [N,7] xyzw pose, [N,6] world twist, [N,3] base command.

        Reference orientation/position remain in world coordinates even while the
        base moves. Measured non-arm wrist motion is cancelled in the IK solve.
        """
        n = self.env.num_envs
        target = torch.as_tensor(target_pose_w, dtype=torch.float32, device=self.env.device)
        if target.shape != (n, 7) or not torch.isfinite(target).all():
            raise ValueError(f"Target pose must be finite with shape ({n}, 7)")
        if not torch.allclose(target[:, 3:].norm(dim=-1), torch.ones(n, device=self.env.device), atol=1e-3):
            raise ValueError("Target orientation must be a unit xyzw quaternion")
        twist = (
            torch.zeros(n, 6, device=self.env.device)
            if target_twist_w is None
            else torch.as_tensor(target_twist_w, dtype=torch.float32, device=self.env.device)
        )
        automatic_placement = self.base_planner is not None and base_velocity_b is None
        if automatic_placement:
            current = target[0].cpu().numpy()
            if preview_pose_w is None:
                velocity_w = twist[0].cpu().numpy()
                horizon = self.base_planner.preview_s
                preview = np.r_[
                    current[:3] + horizon * velocity_w[:3],
                    (Rotation.from_rotvec(horizon * velocity_w[3:]) * Rotation.from_quat(current[3:])).as_quat(),
                ]
            else:
                preview = torch.as_tensor(preview_pose_w)[0].cpu().numpy()
            base_velocity_b = [
                self.base_planner.command(
                    current, preview, self.robot.data.root_link_pose_w.torch[0].cpu().numpy(), self.dt
                ).tolist()
            ]
        velocity = (
            torch.zeros_like(self.base_velocity)
            if base_velocity_b is None
            else torch.as_tensor(base_velocity_b, dtype=torch.float32, device=self.env.device)
        )
        if (
            twist.shape != (n, 6)
            or velocity.shape != (n, 3)
            or not torch.isfinite(twist).all()
            or not torch.isfinite(velocity).all()
        ):
            raise ValueError("Expected finite target twist [N,6] and base velocity [N,3]")
        bounds = velocity.new_tensor([0.5, 0.3, 0.5])
        velocity = velocity.clamp(-bounds, bounds)
        # Bound command acceleration; never step directly from standing to full speed.
        change = velocity - self.base_velocity
        max_change = velocity.new_tensor(self.max_base_acceleration) * self.dt
        self.base_velocity.add_(change.clamp(-max_change, max_change))
        self.env.command_manager.get_command("base_velocity").copy_(self.base_velocity)
        if self.locomotion_policy is None:
            observations = self.env.observation_manager.compute(update_history=False)
            policy_obs = flatten_observation_group(observations["policy"])
            if policy_obs.shape != (n, 255):
                raise ValueError(f"Walking policy needs [N,255], got {tuple(policy_obs.shape)}")
            legs = self.policy(policy_obs)
        else:
            data = self.robot.data
            targets, kp, kd = self.locomotion_policy.step(
                data.root_link_pose_w.torch[:, 3:].cpu().numpy(),
                data.root_ang_vel_b.torch.cpu().numpy(),
                self.base_velocity.cpu().numpy(),
                data.joint_pos.torch[:, self.policy_joint_ids].cpu().numpy(),
                data.joint_vel.torch[:, self.policy_joint_ids].cpu().numpy(),
            )
            if not self._policy_gains_checked:
                # Explicit DC motors own PD gains; simulator drive gains may be zero.
                gains = {
                    name: (float(actuator.stiffness[0, i]), float(actuator.damping[0, i]))
                    for actuator in self.robot.actuators.values()
                    for i, name in enumerate(actuator.joint_names)
                }
                actual = np.array([gains[name] for name in self.locomotion_policy.controlled_names])
                if not np.allclose(actual, np.stack((kp[0], kd[0]), axis=-1)):
                    raise ValueError("Robot actuator PD gains differ from the locomotion policy export")
                self._policy_gains_checked = True
            legs = torch.as_tensor(targets, device=self.env.device)
        actions = torch.zeros(n, 21, device=self.env.device)
        actions[:, self.action_slices["joint_pos"]] = legs

        q = self.robot.data.joint_pos.torch[:, self.arm_ids]
        if arm_enabled:
            pose = self.wrist_pose_w
            error = torch.cat((target[:, :3] - pose[:, :3], rotation_error(target[:, 3:], pose[:, 3:])), -1)
            jacobian = self.arm_jacobian_w
            qdot = self.robot.data.joint_vel.torch[:, self.arm_ids]
            disturbance = self.wrist_twist_w - (jacobian @ qdot.unsqueeze(-1)).squeeze(-1)
            # Solve a position correction each tick instead of integrating task-space
            # velocity indefinitely against the compliant arm's position actuators.
            desired_delta = self.gain * error + self.dt * (twist - disturbance)
            # Metres and radians have different scales; use a 0.25 m rotational characteristic length.
            weights = q.new_tensor([1, 1, 1, 0.25, 0.25, 0.25])
            joint_delta = damped_solve(jacobian * weights[None, :, None], desired_delta * weights, self.damping)
            if automatic_placement and self.base_planner.placement is not None:
                # Share the placement planner's preferred arm posture. Using the
                # SAME damping as the task solve gives the regularized objective
                # ||J*dq - dx||^2 + damping^2*||dq - secondary||^2.
                # This is a soft posture/pose tradeoff, not an exact null-space
                # hierarchy. Like the task gain, posture_gain is per IK correction.
                posture = q.new_tensor(self.base_planner.placement.arm_joints)[None]
                secondary = self.posture_gain * (posture - q)
                joint_delta += nullspace_posture_step(jacobian * weights[None, :, None], secondary, self.damping)
            # Low-pass the IK position target; acceleration limiting alone can
            # create a bang-bang limit cycle around a rapidly changing IK solution.
            velocity = (q + joint_delta - self.arm_target) / max(self.response_time, self.dt)
            velocity = velocity.clamp(-self.max_joint_speed, self.max_joint_speed)
            delta = (velocity - self.arm_velocity).clamp(
                -self.max_joint_acceleration * self.dt, self.max_joint_acceleration * self.dt
            )
            self.arm_velocity.add_(delta)
            proposed = self.arm_target + self.dt * self.arm_velocity
            limits = self.robot.data.soft_joint_pos_limits.torch[:, self.arm_ids]
            # Anti-windup: keep commanded joints near measured joints as well as within soft limits.
            lower = torch.maximum(limits[..., 0], q - 0.3)
            upper = torch.minimum(limits[..., 1], q + 0.3)
            self.arm_target = proposed.clamp(lower, upper)
            self.limit_saturated = (self.arm_target != proposed).any(-1)
        else:
            self.arm_velocity.zero_()
        actions[:, self.action_slices["right_arm"]] = self.arm_target
        if not torch.isfinite(actions).all():
            raise RuntimeError("Non-finite controller output")
        return actions
