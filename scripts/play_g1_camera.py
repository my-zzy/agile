#!/usr/bin/env python3
"""Run G1 world-pose IK with tracking metrics and optional video/reports."""

import argparse
import csv
import hashlib
import json
import math
import os
import shlex
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mode", choices=("hold", "sweep", "maneuver"), default="maneuver")
parser.add_argument("--trajectory", type=Path, help="Absolute world-frame planner trajectory JSON (one environment).")
parser.add_argument(
    "--base-control",
    choices=("auto", "fixed"),
    default="auto",
    help="auto: derive base XY/yaw from wrist poses; fixed: command zero base velocity.",
)
parser.add_argument(
    "--base-velocity",
    type=float,
    nargs=3,
    metavar=("VX", "VY", "WZ"),
    help="Override automatic placement with a separate base-frame walking command.",
)
parser.add_argument(
    "--duration", type=float, help="Rollout seconds (default: maneuver 36, hold/sweep 20, or trajectory duration)."
)
parser.add_argument(
    "--settle", type=float, default=2.0, help="Standing settling period before starting the trajectory."
)
parser.add_argument(
    "--prepare",
    type=float,
    default=3.0,
    help="Seconds to smoothly approach and check the first target before playback (default: 3).",
)
parser.add_argument("--num_envs", type=int, choices=(1,), default=1, help="This baseline currently supports one robot.")
parser.add_argument("--checkpoint", type=Path, help="Walking TorchScript, or ONNX with --locomotion-contract.")
parser.add_argument("--locomotion-contract", type=Path, help="Named stateful ONNX velocity-policy YAML contract.")
parser.add_argument(
    "--output", type=Path, help="New experiment directory under outputs/ (default: timestamped g1_camera run)."
)
parser.add_argument("--disable-ik", action="store_true", help="Hold the arm posture for a tracking-error comparison.")
parser.add_argument("--real-time", action="store_true")
parser.add_argument("--video", action="store_true", help="Record an observer video (off by default).")
parser.add_argument("--report", action="store_true", help="Write English and Chinese reports (off by default).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs < 1 or args.settle < 0 or args.prepare < 1 or (args.duration is not None and args.duration <= 0):
    parser.error("Use num_envs >= 1, settle >= 0, prepare >= 1, and duration > 0")
if args.trajectory and args.num_envs != 1:
    parser.error("Absolute world-frame trajectory files currently require --num_envs 1")
if args.locomotion_contract and (not args.checkpoint or args.checkpoint.suffix != ".onnx"):
    parser.error("--locomotion-contract requires an ONNX --checkpoint")

if args.video:
    args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG

import agile.isaaclab_extras.monkey_patches  # noqa: F401
from agile.controllers.camera_reference import demo_pose, demo_reference, level_recording_pose, maneuver_waypoints
from agile.controllers.experiment_artifacts import (
    ExperimentVideo,
    create_experiment_directory,
    write_experiment_data,
    write_experiment_reports,
)
from agile.controllers.g1_camera import DEFAULT_CHECKPOINT, G1CameraController, make_camera_baseline_cfg
from agile.controllers.pose_trajectory import WorldPoseTrajectory, rotation_error
from agile.controllers.wrist_base_planner import heading, wrap_angle
from agile.isaaclab_extras.record_video import _update_tracked_asset_recording_camera


def close_env(env):
    # Beta2 retains a camera tracking callback during teardown unless explicitly unsubscribed.
    camera = getattr(env, "viewport_camera_controller", None)
    if camera is not None:
        camera.__del__()
    env.close()


def main():
    trajectory = WorldPoseTrajectory.from_json(args.trajectory, args.device) if args.trajectory else None
    duration = args.duration or (
        float(trajectory.times[-1]) if trajectory else (36.0 if args.mode == "maneuver" else 20.0)
    )
    output = create_experiment_directory(REPO_ROOT, args.output, "planner" if trajectory else args.mode)
    run_config = {
        "started_at": datetime.now().astimezone().isoformat(),
        "command": shlex.join(["uv", "run", "--frozen", "scripts/play_g1_camera.py", *sys.argv[1:]]),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "output_directory": str(output),
    }
    run_config["source_sha256"] = {}
    for source in [Path(__file__), *REPO_ROOT.joinpath("agile/controllers").glob("*.py")]:
        relative = source.relative_to(REPO_ROOT)
        content = source.read_bytes()
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        run_config["source_sha256"][str(relative)] = hashlib.sha256(content).hexdigest()
    locomotion_policy = None
    if args.locomotion_contract:
        from agile.controllers.velocity_policy import OnnxVelocityPolicy

        locomotion_policy = OnnxVelocityPolicy(args.checkpoint, args.locomotion_contract)
        run_config["locomotion_policy"] = locomotion_policy.as_dict()
        policy_directory = output / "locomotion_policy"
        policy_directory.mkdir()
        for source in (args.checkpoint, args.locomotion_contract):
            (policy_directory / source.name).write_bytes(source.read_bytes())
    cfg = make_camera_baseline_cfg(
        args.num_envs, args.device, duration + args.settle + args.prepare, locomotion_policy=locomotion_policy
    )
    # Automatic initial relocation may require more time than the old arm-only preparation.
    cfg.episode_length_s += 600
    env = None
    control_dt = cfg.sim.dt * cfg.decimation
    capture_every = 2
    video = ExperimentVideo(output / "video.mp4", fps=1 / (control_dt * capture_every)) if args.video else None
    simulated_steps = 0
    rows = []
    status = "incomplete"
    failure_reason = None
    controller = None
    initial_pose = None
    preparation_error = None
    effective_prepare = args.prepare
    base_rows = []
    tracking_failure_steps = 0
    automatic_base = args.base_control == "auto" and args.base_velocity is None
    if args.trajectory:
        (output / "trajectory.json").write_text(args.trajectory.read_text())

    def capture_step(phase):
        nonlocal simulated_steps
        if video is None:
            return
        simulated_steps += 1
        if simulated_steps % capture_every == 0:
            _update_tracked_asset_recording_camera(env)
            frame = env.render()
            # The RTX recorder can return a blank first frame while streaming
            # assets. Warm up rendering without advancing physics or video time.
            if video.frames == 0:
                for _ in range(10):
                    if frame is not None and frame.size and frame.any():
                        break
                    frame = env.render()
                else:
                    raise RuntimeError("Video renderer did not produce a nonblack initial frame")
            video.append(frame, phase)

    try:
        env = ManagerBasedRLEnv(cfg=cfg, render_mode="rgb_array" if args.video else None)
        with torch.inference_mode():
            env.reset()
            controller = G1CameraController(
                env, checkpoint=args.checkpoint or DEFAULT_CHECKPOINT, locomotion_policy=locomotion_policy
            )
            foot_ids = controller.robot.find_bodies(
                ["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
            )[0]
            initial_pose = controller.wrist_pose_w.clone()
            print(f"[BASELINE] Frozen walking policy: {args.checkpoint or DEFAULT_CHECKPOINT}", flush=True)
            print(f"[BASELINE] Settling for {args.settle}s; right wrist uses world-frame xyzw poses.", flush=True)
            for _ in range(round(args.settle / env.step_dt)):
                action = controller.compute(initial_pose, arm_enabled=False)
                _, _, terminated, truncated, _ = env.step(action)
                capture_step("settle")
                if (terminated | truncated).any():
                    raise RuntimeError("G1 terminated during settling")
            controller.reset()
            if automatic_base:
                planner = controller.enable_base_tracking()
                run_config["base_planner"] = planner.as_dict()
            preview_s = planner.preview_s if automatic_base else 0.3
            initial_pose = (
                trajectory.sample([0.0])[0]
                if trajectory
                else level_recording_pose(controller.robot.data.root_link_pose_w.torch)
            )
            if args.mode == "maneuver" and trajectory is None:
                path_data = maneuver_waypoints(initial_pose, duration)
                (output / "trajectory.json").write_text(json.dumps(path_data, indent=2) + "\n")
                trajectory = WorldPoseTrajectory.from_json(output / "trajectory.json", env.device)
            if automatic_base:
                placement = planner.solve(initial_pose[0].cpu().numpy())
                root = controller.robot.data.root_link_pose_w.torch[0].cpu().numpy()
                distance = float(((placement.xy_yaw[:2] - root[:2]) ** 2).sum() ** 0.5)
                turn = abs(float(wrap_angle(placement.xy_yaw[2] - heading(root[3:]))))
                effective_prepare = max(args.prepare, 2 + 1.875 * distance / 0.14, 2 + 1.875 * turn / 0.25)
                if effective_prepare > 600:
                    raise ValueError("Initial relocation exceeds 600 seconds; start closer to the first wrist target")

            def reference(time_s):
                return trajectory.sample([time_s]) if trajectory else demo_reference(time_s, initial_pose, args.mode)

            markers = None
            # Include the axes in offscreen recordings as well as the live viewer.
            if env.render_mode == "rgb_array" or env.sim.has_gui or env.sim.has_active_visualizers():
                marker_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/CameraWristTarget")
                marker_cfg.markers = {"frame": marker_cfg.markers["frame"]}
                # frame_prim.usd's positive axes extend 1.5 asset units.
                marker_cfg.markers["frame"].scale = (0.20 / 1.5,) * 3
                markers = VisualizationMarkers(marker_cfg)
                markers.visualize(initial_pose[:, :3], initial_pose[:, 3:])
                print("[BASELINE] Target axes: RED +X forward, GREEN +Y left, BLUE +Z up (20 cm).", flush=True)
            print(f"[BASELINE] Initial target world pose [x,y,z,qx,qy,qz,qw]: {initial_pose.tolist()}", flush=True)
            print(
                f"[BASELINE] Approaching the first target over {effective_prepare:.2f}s; base control={args.base_control}.",
                flush=True,
            )
            approach = WorldPoseTrajectory(
                [0.0, effective_prepare * 2 / 3],
                torch.cat((controller.wrist_pose_w[:, :3], initial_pose[:, :3])),
                torch.cat((controller.wrist_pose_w[:, 3:], initial_pose[:, 3:])),
                env.device,
            )
            # Establish the same recording posture even for the fixed-arm comparison.
            minimum_prepare_steps = round(effective_prepare / env.step_dt)
            maximum_prepare_steps = minimum_prepare_steps + (round(12 / env.step_dt) if automatic_base else 0)
            ready_steps = 0
            for step in range(maximum_prepare_steps):
                start = time.perf_counter()
                target, twist = approach.sample([step * env.step_dt])
                preview = approach.sample([step * env.step_dt + preview_s])[0]
                actions = controller.compute(target, twist, preview_pose_w=preview)
                _, _, terminated, truncated, _ = env.step(actions)
                if markers:
                    markers.visualize(target[:, :3], target[:, 3:])
                capture_step("prepare")
                if (terminated | truncated).any():
                    raise RuntimeError("G1 terminated while approaching the first target")
                if automatic_base and step + 1 >= minimum_prepare_steps:
                    root = controller.robot.data.root_link_pose_w.torch[0].cpu().numpy()
                    base_error = float(((planner.placement.xy_yaw[:2] - root[:2]) ** 2).sum() ** 0.5)
                    yaw_error = abs(float(wrap_angle(planner.placement.xy_yaw[2] - heading(root[3:]))))
                    wrist_error = (initial_pose[:, :3] - controller.wrist_pose_w[:, :3]).norm().item()
                    rotation_error_now = (
                        rotation_error(initial_pose[:, 3:], controller.wrist_pose_w[:, 3:]).norm().item()
                    )
                    ready = base_error < 0.12 and yaw_error < 0.26 and wrist_error < 0.08 and rotation_error_now < 0.26
                    ready_steps = ready_steps + 1 if ready else 0
                    if ready_steps >= 10:
                        break
                if args.real_time:
                    time.sleep(max(0.0, env.step_dt - (time.perf_counter() - start)))
            effective_prepare = (step + 1) * env.step_dt
            error_p = (initial_pose[:, :3] - controller.wrist_pose_w[:, :3]).norm(dim=-1).item()
            error_r = rotation_error(initial_pose[:, 3:], controller.wrist_pose_w[:, 3:]).norm(dim=-1).item()
            preparation_error = {"position_m": error_p, "orientation_rad": error_r}
            if automatic_base:
                preparation_error.update(base_position_m=base_error, base_heading_rad=yaw_error)
                if ready_steps < 10:
                    raise RuntimeError("Initial base relocation did not settle within 12 cm / 15 deg before playback")
            print(f"[BASELINE] Initial target check: {error_p:.3f}m, {error_r * 180 / 3.14159265:.1f}deg.", flush=True)
            if error_p > (0.08 if automatic_base else 0.05) or error_r > (0.26 if automatic_base else 0.175):
                raise RuntimeError(
                    "First target preparation failed the pose tolerance (auto: 8 cm / 15 deg; fixed: 5 cm / 10 deg)"
                )
            print(f"[BASELINE] Running {duration}s. Metrics: {output.resolve()}", flush=True)

            # A runnable example in the actual simulator world; replace with your planner's output.
            example_times = [0.0, 3.0, 6.0]
            example_poses = [demo_pose(t, initial_pose[:1], "sweep")[0].tolist() for t in example_times]
            (output / "planner_example.json").write_text(
                json.dumps(
                    {
                        "frame": "world",
                        "quaternion_order": "xyzw",
                        "times_s": example_times,
                        "positions_m": [p[:3] for p in example_poses],
                        "quaternions_xyzw": [p[3:] for p in example_poses],
                    },
                    indent=2,
                )
                + "\n"
            )

            previous_twist = controller.wrist_twist_w.clone()
            for step in range(max(1, round(duration / env.step_dt))):
                start = time.perf_counter()
                if not simulation_app.is_running():
                    break
                t = step * env.step_dt
                target, twist = reference(t)
                base = None if automatic_base else torch.zeros(env.num_envs, 3, device=env.device)
                if args.base_velocity is not None:
                    base[:] = base.new_tensor(args.base_velocity)
                actions = controller.compute(
                    target, twist, base, arm_enabled=not args.disable_ik, preview_pose_w=reference(t + preview_s)[0]
                )
                _, _, terminated, truncated, _ = env.step(actions)
                if (terminated | truncated).any():
                    capture_step("rollout")
                    raise RuntimeError(
                        f"Environment terminated at step {step}; stopping instead of masking a fall with resets"
                    )
                # Compare post-step measurements against the reference at the same timestamp.
                target, twist = reference(t + env.step_dt)
                actual = controller.wrist_pose_w.clone()
                actual_twist = controller.wrist_twist_w.clone()
                position_error = (target[:, :3] - actual[:, :3]).norm(dim=-1)
                orientation_error = rotation_error(target[:, 3:], actual[:, 3:]).norm(dim=-1)
                linear_velocity_error = (twist[:, :3] - actual_twist[:, :3]).norm(dim=-1)
                angular_velocity_error = (twist[:, 3:] - actual_twist[:, 3:]).norm(dim=-1)
                acceleration = (actual_twist - previous_twist) / env.step_dt
                previous_twist = actual_twist
                if markers:
                    markers.visualize(target[:, :3], target[:, 3:])
                capture_step("rollout")
                values = (
                    torch.cat(
                        (
                            position_error[:, None],
                            orientation_error[:, None],
                            linear_velocity_error[:, None],
                            angular_velocity_error[:, None],
                            acceleration[:, :3].norm(dim=-1, keepdim=True),
                            acceleration[:, 3:].norm(dim=-1, keepdim=True),
                            controller.limit_saturated[:, None].float(),
                            actual,
                            target,
                        ),
                        -1,
                    )
                    .cpu()
                    .tolist()
                )
                rows.extend([[t + env.step_dt, i, *value] for i, value in enumerate(values)])
                root = controller.robot.data.root_link_pose_w.torch[0].cpu().numpy()
                actual_base = [float(root[0]), float(root[1]), float(heading(root[3:]))]
                desired_base = planner.placement.xy_yaw.tolist() if automatic_base else [float("nan")] * 3
                feet = controller.robot.data.body_link_pose_w.torch[0, foot_ids, :3].cpu().numpy()
                feet_velocity = controller.robot.data.body_link_vel_w.torch[0, foot_ids, :3].cpu().numpy()
                feet_delta = feet[0] - feet[1]
                yaw = float(heading(root[3:]))
                stance_width = -math.sin(yaw) * feet_delta[0] + math.cos(yaw) * feet_delta[1]
                tilt = math.acos(max(-1.0, min(1.0, 1 - 2 * (root[3] ** 2 + root[4] ** 2))))
                base_rows.append(
                    [
                        t + env.step_dt,
                        *actual_base,
                        *desired_base,
                        *controller.base_velocity[0].cpu().tolist(),
                        int(planner.command_limited) if automatic_base else 0,
                        float(root[2]),
                        tilt,
                        float(stance_width),
                        float((feet_velocity[0] ** 2).sum() ** 0.5),
                        float((feet_velocity[1] ** 2).sum() ** 0.5),
                        float(feet[0, 2]),
                        float(feet[1, 2]),
                        *root[3:].tolist(),
                        *controller.robot.data.joint_pos.torch[0, controller.arm_ids].cpu().tolist(),
                        *(planner.placement.arm_joints.tolist() if automatic_base else [float("nan")] * 7),
                    ]
                )
                bad_tracking = position_error.max().item() > 0.35 or orientation_error.max().item() > 1.2
                tracking_failure_steps = tracking_failure_steps + 1 if bad_tracking else 0
                if tracking_failure_steps * env.step_dt >= 2:
                    raise RuntimeError(
                        "Wrist tracking lost for 2 seconds (>35 cm or >1.2 rad); target timing was not altered"
                    )
                if step % 100 == 0:
                    print(
                        f"[BASELINE] t={t:.1f}s position={position_error.mean().item():.3f}m "
                        f"orientation={torch.rad2deg(orientation_error).mean().item():.1f}deg",
                        flush=True,
                    )
                if args.real_time:
                    time.sleep(max(0, env.step_dt - (time.perf_counter() - start)))
            else:
                status = "complete"
    except BaseException as exc:
        status = "failed"
        failure_reason = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        raise
    finally:
        video_error = None
        video_summary = {"status": "disabled"}
        if video is not None:
            try:
                video_summary = video.close()
                if video_summary["status"] != "saved":
                    raise RuntimeError("Experiment produced no video frames")
            except Exception as exc:
                video_error = exc
                status = "failed"
                failure_reason = failure_reason or f"Video encoding failed: {exc}"
                video_summary = {"status": "failed", "error": str(exc), "frames": video.frames}
        fields = [
            "time_s",
            "env",
            "position_error_m",
            "orientation_error_rad",
            "linear_velocity_error_mps",
            "angular_velocity_error_radps",
            "linear_acceleration_mps2",
            "angular_acceleration_radps2",
            "joint_target_saturated",
        ]
        fields += [
            f"{prefix}_{name}" for prefix in ("actual", "target") for name in ("x", "y", "z", "qx", "qy", "qz", "qw")
        ]
        with (output / "tracking.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(fields)
            writer.writerows(rows)
        with (output / "base_tracking.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "time_s",
                    "actual_x",
                    "actual_y",
                    "actual_yaw",
                    "desired_x",
                    "desired_y",
                    "desired_yaw",
                    "command_vx",
                    "command_vy",
                    "command_wz",
                    "command_limited",
                    "base_height_m",
                    "base_tilt_rad",
                    "stance_width_m",
                    "left_foot_speed_mps",
                    "right_foot_speed_mps",
                    "left_foot_height_m",
                    "right_foot_height_m",
                    *[f"base_{name}" for name in ("qx", "qy", "qz", "qw")],
                    *[f"arm_joint_{i}" for i in range(7)],
                    *[f"planned_arm_joint_{i}" for i in range(7)],
                ]
            )
            writer.writerows(base_rows)
        summary = {
            "status": status,
            "failure_reason": failure_reason,
            "samples": len(rows),
            "num_envs": args.num_envs,
            "ik_enabled": not args.disable_ik,
            "mode": "planner" if args.trajectory else args.mode,
            "base_control": "override" if args.base_velocity is not None else args.base_control,
            "checkpoint": str(args.checkpoint or DEFAULT_CHECKPOINT),
            "locomotion_policy": run_config.get("locomotion_policy"),
            "recorder_payload_modelled": False,
            "seed": cfg.seed,
            "control_dt_s": control_dt,
            "measured_duration_s": rows[-1][0] if rows else 0.0,
            "video": video_summary,
            "requested_duration_s": duration,
            "settle_s": args.settle,
            "prepare_s": effective_prepare,
            "initial_target_pose_world_xyzw": initial_pose.tolist() if initial_pose is not None else None,
            "preparation_error": preparation_error,
            "base_velocity_override": args.base_velocity,
            "trajectory_file": str(args.trajectory) if args.trajectory else None,
            "trajectory_snapshot": "trajectory.json" if (output / "trajectory.json").exists() else None,
            "base_planner": run_config.get("base_planner"),
        }
        if controller is not None:
            summary["max_base_command_acceleration"] = controller.max_base_acceleration
            summary["ik_parameters"] = {
                name: getattr(controller, name)
                for name in (
                    "damping",
                    "gain",
                    "max_joint_speed",
                    "max_joint_acceleration",
                    "response_time",
                    "posture_gain",
                )
            }
        if rows:
            data = torch.tensor(rows)
            for column in range(2, 8):
                summary[fields[column] + "_rms"] = float(data[:, column].square().mean().sqrt())
                summary[fields[column] + "_p95"] = float(torch.quantile(data[:, column], 0.95))
            summary["joint_target_saturated_fraction"] = float(data[:, 8].mean())
            summary["target_xyz_range_m"] = (data[:, 16:19].max(0).values - data[:, 16:19].min(0).values).tolist()
        if base_rows:
            data = torch.tensor(base_rows)
            summary["base_xy_range_m"] = (data[:, 1:3].max(0).values - data[:, 1:3].min(0).values).tolist()
            summary["base_travel_m"] = float((data[1:, 1:3] - data[:-1, 1:3]).norm(dim=-1).sum())
            if automatic_base:
                summary["base_position_error_m_rms"] = float(
                    (data[:, 1:3] - data[:, 4:6]).square().sum(-1).mean().sqrt()
                )
                summary["base_command_limited_fraction"] = float(data[:, 10].mean())
        if args.report:
            write_experiment_reports(output, summary, run_config)
        else:
            write_experiment_data(output, summary, run_config)
        print(f"[BASELINE] {json.dumps(summary, indent=2)}", flush=True)
        print(f"[BASELINE] Results: {output} (video={args.video}, reports={args.report})", flush=True)
        if env is not None:
            close_env(env)
        if video_error is not None and sys.exc_info()[0] is None:
            raise video_error


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    else:
        simulation_app.close()
