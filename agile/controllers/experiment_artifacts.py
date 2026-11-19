"""Video and bilingual reports for G1 experiments (no simulator imports)."""

import json
import math
from datetime import datetime
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def create_experiment_directory(repo_root, requested=None, mode="walk"):
    """Keep every run under outputs and refuse to overwrite an existing run."""
    root = Path(repo_root).resolve() / "outputs"
    if requested is None:
        output = root / "g1_camera" / f"{datetime.now():%Y-%m-%d_%H-%M-%S_%f}_{mode}"
    else:
        requested = Path(requested)
        if requested.is_absolute():
            output = requested
        elif requested.parts and requested.parts[0] == "outputs":
            output = Path(repo_root) / requested
        else:
            output = root / requested
    output = output.resolve()
    if output == root or not output.is_relative_to(root):
        raise ValueError("Experiment directory must be a subdirectory of this repository's outputs/ folder")
    output.mkdir(parents=True, exist_ok=False)
    return output


class ExperimentVideo:
    """Stream frames to H.264 instead of retaining a whole rollout in memory."""

    def __init__(self, path, fps):
        self.path = Path(path)
        self.fps = fps
        self.frames = 0
        self.phases = []
        self.resolution = None
        self.writer = None

    def append(self, frame, phase):
        frame = np.asarray(frame)
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
            raise ValueError("Video capture must return a nonempty uint8 RGB image")
        resolution = [int(frame.shape[1]), int(frame.shape[0])]
        if self.resolution is not None and resolution != self.resolution:
            raise ValueError("Video frame resolution changed during the experiment")
        if self.writer is None:
            self.writer = imageio.get_writer(
                self.path,
                fps=self.fps,
                codec="libx264",
                quality=7,
                macro_block_size=2,
                pixelformat="yuv420p",
                output_params=["-movflags", "+faststart"],
            )
            self.resolution = resolution
        self.writer.append_data(frame)
        if not self.phases or self.phases[-1]["phase"] != phase:
            self.phases.append({"phase": phase, "start_s": self.frames / self.fps, "frames": 0})
        self.phases[-1]["frames"] += 1
        self.frames += 1

    def close(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        saved = self.frames > 0 and self.path.is_file() and self.path.stat().st_size > 0
        validation = validate_video(self.path, self.frames) if saved else None
        return {
            "status": "saved" if saved else "missing",
            "file": self.path.name if saved else None,
            "fps": self.fps,
            "frames": self.frames,
            "duration_s": self.frames / self.fps,
            "resolution": self.resolution,
            "phases": self.phases,
            "validation": validation,
        }


def validate_video(path, expected_frames):
    """Decode the complete MP4 and identify blank frames before reporting success."""
    reader = imageio.get_reader(path)
    black_frames = []
    count = 0
    try:
        metadata = reader.get_meta_data()
        for i, frame in enumerate(reader):
            if not np.asarray(frame).any():
                black_frames.append(i)
            count += 1
    finally:
        reader.close()
    if count != expected_frames or count == 0:
        raise RuntimeError(f"Video decoded {count} frames, expected {expected_frames}")
    if len(black_frames) == count:
        raise RuntimeError("Every recorded frame is black")
    return {
        "decoded_frames": count,
        "codec": metadata.get("codec"),
        "fps": metadata.get("fps"),
        "duration_s": metadata.get("duration"),
        "entirely_black_frame_indices": black_frames,
    }


def write_experiment_data(output, summary, run_config):
    """Save machine-readable measurements/settings without generating reports."""
    output = Path(output)
    (output / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "run_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def write_experiment_reports(output, summary, run_config):
    """Generate requested bilingual reports, including failed/partial runs."""
    output = Path(output)
    write_experiment_data(output, summary, run_config)
    metrics = [
        ("Wrist position error", "腕部位置误差", "position_error_m", 100, "cm"),
        ("Wrist orientation error", "腕部姿态误差", "orientation_error_rad", 180 / math.pi, "deg"),
        ("Linear velocity tracking error", "线速度跟踪误差", "linear_velocity_error_mps", 1, "m/s"),
        ("Angular velocity tracking error", "角速度跟踪误差", "angular_velocity_error_radps", 1, "rad/s"),
        ("Measured linear acceleration", "实测线加速度", "linear_acceleration_mps2", 1, "m/s²"),
        ("Measured angular acceleration", "实测角加速度", "angular_acceleration_radps2", 1, "rad/s²"),
    ]

    def number(key, factor=1):
        value = summary.get(key)
        return "—" if value is None else f"{value * factor:.3f}"

    video = summary.get("video", {})
    pose = summary.get("initial_target_pose_world_xyzw")
    preparation = summary.get("preparation_error")
    for lang in ("en", "zh"):
        zh = lang == "zh"

        def tr(en, cn, chinese=zh):
            return cn if chinese else en

        status = summary.get("status", "incomplete")
        status_text = (
            {"complete": "完成", "failed": "失败", "incomplete": "未完成"}.get(status, status) if zh else status
        )
        lines = [
            tr("# G1 camera experiment report", "# G1 摄像任务实验报告"),
            "",
            f"**{tr('Experiment', '实验')}**: `{output.name}`  ",
            f"**{tr('Started', '开始时间')}**: {run_config.get('started_at', '—')}  ",
            f"**{tr('Status', '状态')}**: {status_text}",
            "",
            tr("## Experiment setup", "## 实验设置"),
            "",
            tr(
                "Frozen pretrained locomotion policy + seven-joint right-arm IK.",
                "冻结的预训练行走策略 + 右臂七关节逆运动学控制。",
            ),
            "",
            f"- {tr('Mode', '模式')}: `{summary.get('mode', '—')}`; IK: {summary.get('ik_enabled', '—')}",
            f"- {tr('Seed / robots', '随机种子 / 机器人数量')}: {summary.get('seed', '—')} / {summary.get('num_envs', '—')}",
            f"- {tr('Requested rollout / measured rollout', '请求轨迹时长 / 实测轨迹时长')}: "
            f"{summary.get('requested_duration_s', '—')} s / {number('measured_duration_s')} s",
            f"- {tr('Settling / preparation', '站立稳定 / 目标准备')}: {summary.get('settle_s', '—')} s / {summary.get('prepare_s', '—')} s",
            f"- {tr('Control timestep', '控制时间步长')}: {summary.get('control_dt_s', '—')} s",
            f"- {tr('Separate base command', '独立底盘速度指令')}: `{summary.get('base_velocity_override')}`",
            f"- {tr('Base control', '底盘控制')}: `{summary.get('base_control', '—')}`",
            f"- {tr('Checkpoint', '策略权重')}: `{summary.get('checkpoint', '—')}`",
            f"- {tr('Planner file', '规划器轨迹文件')}: `{summary.get('trajectory_file')}`",
            "",
            tr("## Target pose and recording", "## 目标位姿与录像"),
            "",
            tr(
                "World coordinates; metres; quaternion order **xyzw**; controlled frame `right_wrist_yaw_link`.",
                "使用世界坐标系，位置单位为米，四元数顺序为 **xyzw**，控制坐标系为 `right_wrist_yaw_link`。",
            ),
            "",
            tr("Initial target `[x, y, z, qx, qy, qz, qw]`:", "初始目标 `[x, y, z, qx, qy, qz, qw]`："),
            "",
            "```json",
            json.dumps(pose, ensure_ascii=False),
            "```",
            "",
        ]
        if summary.get("base_control") == "auto":
            lines += [
                tr(
                    "Base XY and heading are solved from the wrist trajectory and the live robot's arm kinematics. "
                    "A 0.3 s preview supplies feedforward; measured base position/heading supplies feedback. "
                    "Placement updates at 10 Hz, arm and walking commands at 50 Hz. "
                    "No separate base path is supplied; wrist positions, orientations and timestamps are not modified.",
                    "底盘 XY 位置和朝向由腕部轨迹及当前机器人手臂运动学求解。"
                    "使用 0.3 秒轨迹预览提供前馈，实测底盘位置和朝向提供反馈。"
                    "底盘位置规划频率为 10 Hz，手臂与行走指令频率为 50 Hz。"
                    "无需独立底盘路径，不修改腕部位置、姿态和时间戳。",
                ),
                "",
            ]
            parameters = summary.get("base_planner", {})
            lines += [
                f"- {tr('Forward / lateral / yaw command limits', '前进 / 侧向 / 偏航指令限幅')}: `{parameters.get('max_base_velocity')}` (m/s, m/s, rad/s)",
                f"- {tr('Reverse speed limit', '后退速度限幅')}: `{parameters.get('max_reverse_velocity_mps', 'same as forward')}` m/s",
                f"- {tr('Position / heading feedback gains', '位置 / 朝向反馈增益')}: `{parameters.get('position_heading_gain_per_s', parameters.get('position_gain_per_s'))}`",
                f"- {tr('Velocity damping', '速度阻尼')}: `{parameters.get('velocity_damping', [0, 0, 0])}`",
                f"- {tr('Translation speed envelope', '平移速度约束')}: `{parameters.get('translation_speed_envelope', 'per-axis')}`",
                "",
            ]
        if summary.get("mode") == "maneuver":
            lines += [
                tr(
                    "The wrist-only waypoint path travels 0.60 m forward, 0.50 m sideways, backward and back to its start, "
                    "with height changes up to 0.04 m and roll, pitch and yaw changes. Each waypoint is a smooth stop. "
                    "The exact executed input is saved in trajectory.json.",
                    "仅提供腕部航点路径：前进 0.60 m、侧移 0.50 m、后退并返回起点，同时包含最大 0.04 m 高度变化"
                    "及横滚、俯仰、偏航变化。每个航点平滑停稳，实际输入保存在 trajectory.json 中。",
                ),
                "",
            ]
        elif summary.get("mode") != "planner":
            lines += [
                tr(
                    "The target starts 30 cm forward and 25 cm right of the initial base, at world height 95 cm. "
                    "Roll/pitch are zero. Moving demos add translation and gentle yaw while keeping the target level.",
                    "初始目标位于初始底盘前方 30 cm、右侧 25 cm，世界高度为 95 cm。横滚角和俯仰角为零。"
                    "运动演示叠加平移与轻微偏航，目标保持水平。",
                ),
                "",
            ]
        if preparation:
            lines += [
                tr("First-target error after preparation", "准备阶段结束时的首个目标误差")
                + f": {preparation['position_m'] * 100:.2f} cm / {math.degrees(preparation['orientation_rad']):.2f}°.",
                "",
            ]
        lines += [
            tr(
                "Target axes: **red +X, green +Y, blue +Z**, 20 cm long. The video is an external observer view, "
                "not footage from a wrist-mounted camera.",
                "目标三轴：**红色 +X、绿色 +Y、蓝色 +Z**，长度为 20 cm。录像来自外部观察视角，"
                "不是安装在手腕上的摄像机画面。",
            ),
            "",
        ]
        if video.get("status") == "saved":
            lines += [
                f"[{tr('Watch video', '观看录像')}](video.mp4)",
                "",
                f"{video['resolution'][0]} × {video['resolution'][1]}, {video['fps']:g} fps, "
                f"{video['duration_s']:.2f} s, {video['frames']} {tr('frames', '帧')}.",
                "",
                tr("| Phase | Video start (s) | Duration (s) |", "| 阶段 | 视频起始时间 (s) | 时长 (s) |"),
                "| --- | ---: | ---: |",
            ]
            for phase in video.get("phases", []):
                name = phase["phase"]
                if zh:
                    name = {"settle": "站立稳定", "prepare": "目标准备", "rollout": "轨迹跟踪"}.get(name, name)
                lines += [f"| {name} | {phase['start_s']:.2f} | {phase['frames'] / video['fps']:.2f} |"]
        elif video.get("status") == "disabled":
            lines += [tr("Video recording was not requested.", "本次未请求录制视频。")]
        else:
            lines += [
                tr(
                    "**Video unavailable.** No successful recording is claimed.",
                    "**录像不可用。** 本次未获得有效录像。",
                ),
                str(video.get("error", "")),
            ]
        validation = video.get("validation")
        if validation:
            lines += [
                "",
                tr("Video decoding check", "视频解码检查")
                + f": {validation['decoded_frames']} {tr('frames decoded', '帧解码成功')}; "
                + f"{len(validation['entirely_black_frame_indices'])} {tr('entirely black frames', '全黑帧')}.",
            ]
            if validation["entirely_black_frame_indices"]:
                lines += [
                    tr("Black frame indices (starting at 0)", "全黑帧序号（从 0 开始）")
                    + f": `{validation['entirely_black_frame_indices']}`."
                ]
        lines += [
            "",
            tr("## Measured results", "## 实测结果"),
            "",
            tr(
                "Metrics cover rollout samples only, excluding settling and preparation.",
                "指标仅统计轨迹跟踪阶段，不包含站立稳定和目标准备阶段。",
            ),
            "",
            tr("| Metric | RMS | p95 | Unit |", "| 指标 | 均方根 RMS | 95 分位数 p95 | 单位 |"),
            "| --- | ---: | ---: | --- |",
        ]
        for en, cn, key, factor, unit in metrics:
            lines += [f"| {tr(en, cn)} | {number(key + '_rms', factor)} | {number(key + '_p95', factor)} | {unit} |"]
        lines += [
            "",
            f"{tr('Joint-target saturation', '关节目标限幅比例')}: {number('joint_target_saturated_fraction', 100)}%.",
            "",
            tr("## Interpretation and limitations", "## 结论与局限"),
            "",
        ]
        if "base_travel_m" in summary:
            lines += [
                f"- {tr('Measured base travel', '实测底盘累计行程')}: {number('base_travel_m')} m",
                f"- {tr('Base XY range', '底盘 XY 运动范围')}: `{summary.get('base_xy_range_m')}` m",
                f"- {tr('Target XYZ range', '目标 XYZ 运动范围')}: `{summary.get('target_xyz_range_m')}` m",
                f"- {tr('Base XY tracking RMS', '底盘 XY 跟踪误差 RMS')}: {number('base_position_error_m_rms', 100)} cm",
                f"- {tr('Base command speed saturation', '底盘指令速度限幅比例')}: {number('base_command_limited_fraction', 100)}%",
                "",
            ]
        if summary.get("base_control") == "auto":
            lines += [
                tr(
                    "The placement model assumes flat unobstructed ground and a fixed nominal pelvis height/waist configuration. "
                    "Arm IK compensates actual body motion. Local IK can fail even when another posture is feasible; "
                    "there is no global reachability, collision, footstep or balance guarantee. "
                    "A geometrically reachable path can still be too fast for the walking policy/actuators. "
                    "Tracking loss above 0.35 m or 1.2 rad for 2 s stops the run as failed, without retiming the target.",
                    "底盘规划模型假设地面平坦无障碍，骨盆高度和腰部构型为固定标称值；手臂 IK 补偿实际机体运动。"
                    "局部 IK 可能在其他姿态可行时仍求解失败，不提供全局可达性、避碰、落脚点或平衡保证。"
                    "几何上可达的路径仍可能超过行走策略或执行器速度能力。"
                    "跟踪误差超过 0.35 m 或 1.2 rad 持续 2 秒将终止运行并标记失败，不调整目标时间。",
                ),
                "",
            ]
        if status == "complete":
            lines += [
                tr(
                    "The requested rollout completed without environment termination. Completion is not a filming-quality acceptance test.",
                    "请求的轨迹执行完成，期间未触发环境终止。完成运行不等于达到拍摄画面稳定性的验收标准。",
                )
            ]
        else:
            lines += [
                tr(
                    "The experiment failed or ended early. Any metrics and video describe only the available portion.",
                    "实验失败或提前结束。现有指标和录像仅覆盖实际完成的部分。",
                )
            ]
        if summary.get("failure_reason"):
            lines += ["", f"{tr('Failure detail', '失败详情')}: `{summary['failure_reason']}`"]
        lines += [
            "",
            tr(
                "The recorder payload, grip, optical camera and calibrated wrist-to-camera transform are not modeled. "
                "The reference can be level while the real wrist still has pose error and shake. "
                "This single-seed simulation does not validate arbitrary planner paths or hardware operation.",
                "尚未建模录像设备载荷、抓握、光学摄像机及标定的腕部到摄像机变换。"
                "目标保持水平时，实际腕部仍可能存在位姿误差和抖动。"
                "单一随机种子的仿真结果不能验证任意规划轨迹或实机运行。",
            ),
            "",
            tr(
                "Further validation requires more paths, speeds and orientations, then recorder payload and stabilization tests.",
                "后续需要验证更多轨迹、速度和姿态，再开展录像设备载荷及稳定性测试。",
            ),
            "",
            tr("## Reproduction and files", "## 复现与文件"),
            "",
            "```bash",
            run_config.get("command", ""),
            "```",
            "",
            tr(
                "Choose a new output directory when repeating this run; existing experiment directories are never overwritten.",
                "复现实验时请使用新的输出目录；已有实验目录不会被覆盖。",
            ),
            "",
            "- [Run configuration / 运行配置](run_config.json)",
            "- [Metrics / 指标](metrics.json)",
            "- [Tracking samples / 跟踪数据](tracking.csv)",
            f"- [{tr('Chinese report', '英文报告')}](report_{'zh' if lang == 'en' else 'en'}.md)",
            "",
        ]
        if (output / "base_tracking.csv").exists():
            lines += ["- [Base tracking / 底盘跟踪数据](base_tracking.csv)"]
        if summary.get("trajectory_snapshot"):
            lines += ["- [Executed wrist trajectory / 实际腕部轨迹](trajectory.json)"]
        (output / f"report_{lang}.md").write_text("\n".join(lines), encoding="utf-8")
