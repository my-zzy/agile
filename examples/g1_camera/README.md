# Wrist-only planner input

```bash
uv run --frozen scripts/play_g1_camera.py \
  --trajectory examples/g1_camera/wrist_6d_relocation.json
```

Automatic base placement is the default. The robot first moves to a suitable
starting position, then follows the supplied wrist trajectory using the frozen
walking policy and right-arm IK. Add `--viz kit --real-time` for a live view.
Each invocation creates a new directory under `outputs/g1_camera/` with metrics,
the exact input (`trajectory.json`), source snapshot, and wrist/base tracking data.
Video and reports are off by default. Add `--video` only when a recording is
requested, and `--report` when English/Chinese reports are requested.

`wrist_6d_relocation.json` is a 24-second test with an offset first pose, backward
and lateral motion, a direction reversal, height changes, and roll/pitch/yaw
changes. Its absolute coordinates are for this repository's default single-G1
scene with seed 42. Adjust world coordinates when changing the initial scene.

For a level target moving forward, backward, left and right in sequence:

```bash
uv run --frozen scripts/play_g1_camera.py \
  --trajectory examples/g1_camera/wrist_forward_backward_left_right.json
```

This 48-second path moves 0.50 m per direction over four 10-second segments,
with a 2-second hold after each segment so the body can settle before changing
direction. Directions are fixed in the world frame: forward +X,
backward −X, left +Y, right −Y. Height varies between 0.93 and 0.97 m;
the target orientation stays level at quaternion `[0, 0, 0, 1]` (xyzw).
The backward and right segments return along the preceding horizontal path.
Minimum-jerk interpolation stops smoothly at each segment boundary. As with
the other example, its absolute coordinates assume the default seed-42 scene.

The JSON needs only:

- `frame: "world"` and `quaternion_order: "xyzw"`;
- `times_s`: strictly increasing times, starting at zero;
- `positions_m`: one world XYZ wrist position for each time;
- `quaternions_xyzw`: one unit wrist quaternion for each time.

Targets refer to `right_wrist_yaw_link`, not a calibrated camera optical frame.
Interpolation stops smoothly at each waypoint. For continuous streaming, use
`G1CameraController.enable_base_tracking()` after settling, then call `compute`
with world poses/twists and an optional 0.3-second preview pose. See
[the controller plan](../../G1_CAMERA_PLAN.md) for the complete contract.

The default demo is `--mode maneuver`, a multidirectional wrist path.
`--base-control fixed` commands zero base velocity; `--base-velocity VX VY WZ`
explicitly overrides automatic placement for controlled comparisons.

Reachability is limited by arm joints, nominal walking height, locomotion speed,
and balance. The planner assumes flat unobstructed ground; it does not provide
collision avoidance or a guarantee for every globally reachable full-body pose.
