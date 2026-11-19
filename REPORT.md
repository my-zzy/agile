# G1 Camera Tracking — Progress Summary

[中文版](REPORT_ZH.md) · Updated: **2026-11-19**

Goal: track world-frame wrist poses while G1 selects a suitable body position for smooth video recording.

| Milestone | Summary | Details |
| --- | --- | --- |
| G16 + body placement + arm IK | Completed four-direction translation at 1×–4× target speed; tracking lag increases at higher speeds. | [Report, figures, and videos](docs/milestones/2026-11-19_g16_baseline.md) |
| Orientation tracking | Isolated roll/pitch/yaw sweeps completed; combined leftward walking and turning failed at about 52.6 s. | [Results and failure evidence](docs/milestones/2026-11-19_g16_orientation_tracking.md) |

Earlier AGILE wrist-driven control, arm-posture refinement, and speed tests were superseded after body lag, arm-limit proximity, and an unstable 2× final hold exposed their limitations.

Next: diagnose walking/turning instability, then expand 6D and recorder-payload tests. [Full todo list](docs/milestones/2026-11-19_g16_orientation_tracking.md#todos).
