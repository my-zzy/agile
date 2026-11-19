# G1 camera trajectory tracking plan

## Objective and planner contract

Track a higher-level planner's time-indexed **6D right-wrist trajectory in world coordinates**, while walking steadily and minimizing unwanted recorder motion. A stationary world pose is a constant trajectory. Intentional camera translation and panning must remain possible.

- Positions: metres in the simulator world frame.
- Orientations: unit quaternions in **xyzw** order, matching Isaac Lab 3.
- Controlled frame: origin and axes of `right_wrist_yaw_link`.
- Time: seconds from the start of the rollout, after settling and approaching the first target.
- Desired twist: world-frame linear velocity in m/s followed by angular velocity in rad/s.
- A camera-frame planner must convert camera targets to wrist targets using the calibrated wrist-to-camera transform.
- Planner paths must be continuous and within the robot's workspace and speed limits. The default baseline now derives base XY motion and heading from wrist poses; no separate walking path is required.

## Stage 1 — wrist-driven baseline (current implementation)

Use the bundled `unitree_g1_velocity_history_torchscript.pt` unchanged. It controls the twelve leg joints and waist roll/pitch. Preserve the original 255-value, five-step observation history and 14 previous locomotion actions; never append arm actions to that input.

Control the seven right-arm joints with damped differential IK. Use the **world-frame wrist-link-origin Jacobian**, compensate measured wrist motion caused by the floating base and other joints, low-pass the IK joint targets, and limit joint target speed, acceleration, and position. Hold the left arm and waist yaw at their nominal posture.

The nominal right arm starts in a bent-elbow recording posture. In this baseline only, right-wrist stiffness/damping are 12 Nm/rad and 0.6 Nm·s/rad (the locomotion asset's 4 / 0.2 settings allowed noticeable droop in a horizontal hold). Left-arm gains and all actuator effort limits are unchanged.

The controller runs at 50 Hz with 200 Hz physics and currently supports **one robot**. The runner defaults to `--base-control auto`: it automatically repositions and turns the base using only the world-space wrist trajectory. The pretrained walking network remains frozen. `--base-control fixed` commands zero velocity. An explicit `--base-velocity VX VY WZ` overrides automatic placement for controlled comparisons.

### Wrist-driven base placement

After settling, the runner calibrates a seven-joint arm kinematic model from the live asset's measured pose and geometric Jacobian. It does not depend on a separately copied robot description. Product-of-exponentials forward kinematics represents the arm at other joint angles, with the waist configuration fixed at calibration for planning purposes. Actual waist/body motion is still compensated by the measured-Jacobian arm IK.

At 10 Hz a bounded local least-squares solve chooses seven arm angles and a base heading to match target wrist height and orientation, with a weak preference for a comfortable nominal arm posture. Base XY is then determined by subtracting the rotated arm offset from the target's world XY. This supports lateral and backward translation as well as forward motion; the viewing direction need not match the travel direction.

A second placement solve at a 0.3 s preview estimates base feedforward velocity. At 50 Hz, world XY/heading error feedback (gains 1.5 per second) and filtered velocity damping are added, then rotated into body-frame `vx, vy, yaw_rate`. The velocity estimate uses a 0.2 s low-pass filter; damping gains are 0.45, 0.45, 0.4. Commands are bounded at 0.30 m/s forward, 0.25 m/s backward, 0.22 m/s lateral and 0.35 rad/s yaw. An elliptical XY speed envelope prevents simultaneous maximum longitudinal and lateral commands; the existing acceleration limiter remains active. Fixed wrist targets therefore also receive base-position feedback instead of simply sending zero velocity.

Initial relocation happens during preparation, before planner time starts; preparation is extended based on the initial base displacement/turn. Automatic preparation must reach wrist error below 8 cm / 15 degrees and base-placement error below 12 cm / 15 degrees for ten consecutive control ticks. Up to 12 extra seconds are allowed to settle before declaring preparation failed. The old fixed-base wrist check remains 5 cm / 10 degrees. During playback the target pose and timestamps are never projected or slowed. A local placement residual over 2.5 cm / 0.12 rad raises an infeasibility error; actual wrist error over 35 cm or 1.2 rad for 2 seconds ends the run as failed and retains its artifacts.

This is a local kinematic coordination baseline, not a guarantee for every mathematically reachable full-body pose. It assumes flat unobstructed ground, nominal pelvis height, and the capabilities of the fixed locomotion policy. It does not plan obstacles, contacts, crouching or footsteps; local IK may miss another feasible solution. Geometric feasibility does not prove dynamic feasibility or filming-quality tracking.

Entry points:

- `agile/controllers/g1_camera.py`: environment configuration and `G1CameraController`.
- `agile/controllers/pose_trajectory.py`: world-pose waypoint sampling and IK math.
- `agile/controllers/wrist_base_planner.py`: calibrated arm FK, base placement and preview/feedback velocity control.
- `agile/controllers/camera_reference.py`: level recording pose and demo paths.
- `agile/controllers/experiment_artifacts.py`: streaming MP4 recording and bilingual experiment reports.
- `scripts/play_g1_camera.py`: demo, planner-file playback, and metrics collection.

### Run the baseline

```bash
# Wrist-only 6D loop: forward, sideways, backward, return, and turning.
uv run --frozen scripts/play_g1_camera.py --mode maneuver --viz kit --real-time

# Your planner's absolute wrist poses automatically drive the body and arm.
uv run --frozen scripts/play_g1_camera.py --trajectory /path/to/planner.json

# Standing, holding a level recording pose in world coordinates.
uv run --frozen scripts/play_g1_camera.py --mode hold --base-control fixed --viz kit --real-time

# Standing, tracking small position and orientation changes.
uv run --frozen scripts/play_g1_camera.py --mode sweep --base-control fixed --viz kit --real-time

# Headless evaluation and a matched comparison without IK; both record video.
uv run --frozen scripts/play_g1_camera.py --trajectory examples/g1_camera/wrist_6d_relocation.json --output outputs/camera_with_ik
uv run --frozen scripts/play_g1_camera.py --trajectory examples/g1_camera/wrist_6d_relocation.json --disable-ik --output outputs/camera_without_ik
```

Every run automatically saves its artifacts under **`outputs/g1_camera/<timestamp>_<mode>/`**. Recording is enabled for both headless and GUI runs. Use `--output outputs/<new-run-name>` to choose a directory; existing directories are never overwritten. Startup can leave the window blank for tens of seconds while Isaac Sim initializes.

### Required experiment artifacts

After every G1 camera/control experiment, deliver a video and matching English and Chinese reports in `outputs/`:

| File | Contents |
| --- | --- |
| `video.mp4` | H.264 observer video at 1280 × 720, 25 fps, covering settling, preparation, and rollout; includes the target axes after settling |
| `report_en.md` | English setup, target, video timeline, measured results, completion/failure status, interpretation, and limitations |
| `report_zh.md` | Matching Chinese report generated from the same measurements |
| `metrics.json` | Machine-readable metrics, actual initial world target, preparation error, and video metadata |
| `tracking.csv` | Timestamped wrist/target poses and tracking metrics during rollout |
| `base_tracking.csv` | Measured/desired base XY and heading, applied body-frame velocity, command saturation |
| `run_config.json` | Start time, command, and CLI settings for reproduction |
| `source/` | Executed runner/controller source snapshot; hashes in `run_config.json` for new runs |
| `trajectory.json` | Exact executed waypoint input for planner-file and maneuver runs |
| `planner_example.json` | Separate short example anchored to the recording target; **not the executed input** |

Video frames are streamed to disk without keeping the entire recording in RAM. Playback follows simulation time even when rendering runs slower than real time. Metrics exclude settling and preparation; the reports identify each phase's video interval. The observer video is not footage from a wrist-mounted camera.

The recorder warms up rendering before accepting the first frame. At finalization, it decodes the MP4, checks the frame count, and records any completely black frames in both reports.

Failed/interrupted runs retain available frames, partial metrics, and reports explaining the failure until a good result is verified. After a development round, keep only the selected good result unless the user requests the intermediate attempts. A process crash or forced kill can prevent finalization; do not claim a video exists without checking it. For later training experiments, record a representative evaluation rollout and identify its checkpoint and evaluation settings in both reports.

### Default target and visible axes

The built-in demo starts from a level recording pose:

- **Position:** 0.30 m forward and 0.25 m right of the base's position/heading after settling; **world height 0.95 m** above the flat floor.
- **Orientation:** zero roll and pitch, with yaw equal to the base's heading at that instant. Wrist **+X points forward horizontally, +Y left, +Z up**. For a robot at world XY `(0, 0)` facing +X, this is position `(0.30, -0.25, 0.95)` m, quaternion xyzw `(0, 0, 0, 1)`.
- This pose is anchored once in the **world frame**, never re-anchored to the moving torso. The actual world coordinates depend on movement during settling; the script prints them and saves `initial_target_pose_world_xyzw` in `metrics.json`.
- `hold` keeps this pose fixed. `sweep` adds up to 5 cm forward, 4 cm lateral, and 4 cm vertical motion. These modes keep roll/pitch zero and pan by at most ±0.10 rad (5.7 degrees).
- `maneuver` (the default) supplies only a world wrist path around a 0.60 × 0.50 m loop, with 4 cm height changes and roll/pitch/yaw changes; default duration 36 seconds. Automatic body placement follows this path without a separately programmed base route. Its exact waypoints are saved in `trajectory.json`.
- A supplied `--trajectory` provides its own world poses unchanged, including any intentional tilt.

The nominal arm posture is seeded from G1 kinematics with joint-limit margin. After the default 2 s settling period, preparation smoothly approaches the first target and holds it briefly. `--prepare 3` is the minimum automatic preparation duration; body relocation can extend it. The startup pose check is not a collision or full-path reachability guarantee. The fixed-arm comparison also uses IK during preparation, then holds its last joint command during playback.

With `--viz kit`, a **20 cm RGB triad at the commanded wrist target** is enabled automatically: **red = +X, green = +Y, blue = +Z**. It shows the desired orientation, not the measured hand orientation. The viewer follows G1 from the front-right so the arm and marker are easy to inspect.

These axes define a provisional level recording direction at `right_wrist_yaw_link`. The actual recorder's optical axis still requires the wrist-to-camera calibration described in Stage 2.

```bash
uv run --frozen scripts/play_g1_camera.py \
  --trajectory examples/g1_camera/wrist_6d_relocation.json \
  --viz kit --real-time
```

Planner JSON schema:

```json
{
  "frame": "world",
  "quaternion_order": "xyzw",
  "times_s": [0.0, 3.0],
  "positions_m": [[0.3, -0.25, 1.0], [0.35, -0.25, 1.02]],
  "quaternions_xyzw": [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]
}
```

The values above illustrate the schema; they are not a verified G1 pose. Use a planner path consistent with the robot's workspace and initial state. File playback supports one environment and automatically generates base velocities by default. No base fields are required in the JSON. Use `--base-control fixed` to command zero base velocity or `--base-velocity VX VY WZ` to override base placement explicitly.

Waypoint playback uses minimum-jerk position interpolation and shortest-arc orientation interpolation, stopping at each waypoint and holding the final pose. For a continuous planner spline, call the controller directly with sampled poses and twists instead:

```python
controller = G1CameraController(env)
# After settling the robot, calibrate automatic base placement once.
controller.enable_base_tracking()
actions = controller.compute(
    target_pose_w,       # [N, 7], xyz + xyzw, absolute world frame
    target_twist_w,      # [N, 6], world linear + angular velocity
    preview_pose_w=pose_at_t_plus_0_3,  # [N, 7], optional; otherwise extrapolate twist
)
obs, reward, terminated, truncated, info = env.step(actions)
```

Construct `env` using `make_camera_baseline_cfg`; import the simulator-dependent module after `AppLauncher`. On reset, call `controller.reset()`, settle, recalibrate with `enable_base_tracking()`, and replan the world trajectory. An explicit third positional `base_velocity_b` bypasses automatic placement. The standalone demo stops on termination rather than silently hiding a fall with another episode.

### Baseline limits

This stage uses the unloaded G1 model: **recorder mass, inertia, collision geometry, grip, and a recording camera are not yet simulated**. It validates controller layering and world-frame tracking first. Bounded IK does not guarantee reachability, collision avoidance, or filming-quality stabilization. Report position/orientation error and joint-target saturation rather than treating simulation completion as success.

### Automatic base placement validation — 2026-11-18

The final external-JSON evaluation used `examples/g1_camera/wrist_6d_relocation.json` with no base command.
It automatically relocated before playback and completed the 24-second 6D wrist path containing backward/lateral motion,
direction changes, height changes, and roll/pitch/yaw changes. Initial base placement settled to 4.98 cm / 1.29 degrees.
Wrist tracking RMS was **5.12 cm / 10.78 degrees**; p95 was **8.63 cm / 20.16 degrees**.
Joint-target saturation was **11.0%**; base speed-envelope saturation was **23.0%**.
The 32.28-second video contains 807 decodable frames, with no black frames. Robot and target axes were visually checked.
Saved target samples match the original JSON at the original timestamps; no target projection or retiming was used.

The wrist-driven baseline uses filtered velocity damping, an elliptical speed envelope, and checks
both base and wrist settling before playback. Only the final verified experiment is retained.

- [Final video](outputs/g1_camera/2026-11-18_wrist_driven_baseline/video.mp4)
- [English report](outputs/g1_camera/2026-11-18_wrist_driven_baseline/report_en.md)
- [中文报告](outputs/g1_camera/2026-11-18_wrist_driven_baseline/report_zh.md)
- [Retained experiments](outputs/README.md)

Twenty-six CPU tests cover the current implementation, including independent-chain FK comparison, placement in all XY directions, heading-frame
commands, quaternion sign/wrap behavior, impossible-height rejection and combined reverse/lateral speed bounds.
An additional numerical check solved 32 known-reachable poses from the calibrated G1 model; this is a kinematic check,
not a simulator rollout or a global reachability proof. These results establish a functional baseline, not filming-quality tracking.

## Stage 2 — recorder and reference data

Represent a recorder rigidly attached to the hand. Configure mass, center of mass, inertia, geometry, and wrist-to-camera transform. Use **0.5 kg only as a provisional mass** until the actual recorder is specified. Grasp learning is a separate task.

Add planner trajectories with explicit velocity/acceleration information or a continuous interpolation contract. Validate frame conventions, timing, continuity, reachability, and speed limits. Attach a simulated camera for recorded-footage evaluation; image rendering is not necessary for proprioceptive policy training.

## Stage 3 — learned stabilization controller

Build `CameraTracking-G1-v0` and first evaluate learning residual corrections around the IK baseline. Start with the walking policy frozen. The learned policy can adjust the right arm and bounded walking velocity commands. Keep joint ownership disjoint. Consider joint fine-tuning only if the frozen walking policy limits stabilization.

Observations: wrist position/orientation error, target motion and short future preview, joint states, base motion, and previous actions. Identify which measurements are available on the real robot; do not silently rely on simulator-only ground truth for a deployable policy.

Reward accurate pose and velocity tracking, low unwanted acceleration/jerk, smooth actions, balance, moderate effort, and joint-limit compliance. Penalize deviations from requested camera motion instead of penalizing all wrist motion or all panning.

## Stage 4 — curriculum and robustness

1. Standing pose holding.
2. Slow, reachable 6D trajectories while standing.
3. Slow walking and turning while tracking.
4. Representative held-out planner trajectories.
5. Recorder-property variation, actuator delay, observation noise, and mild disturbances.

Run a small PPO smoke test before a long training run. Size the environment batch after measuring memory and throughput on the RTX 5090. Keep the unchanged locomotion checkpoint as a regression reference.

## Stage 5 — evaluation and acceptance

Compare IK, fixed-arm, and learned controllers on the same held-out trajectories and seeds. Report position/orientation RMS and tail errors, linear/angular velocity errors, acceleration and jerk, falls, actuator/position-limit saturation, and camera-view footage.

Choose numerical acceptance thresholds based on the recorder, lens, framing, and planner requirements. A completed rollout or successful short PPO run verifies the pipeline; it does not establish filming-quality stabilization or hardware readiness.
