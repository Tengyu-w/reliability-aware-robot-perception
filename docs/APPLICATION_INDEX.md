# Application Index

This page is the supervisor-facing entry point for the repository. The current
primary framing is a VPPV-style perception reliability monitor for surgical
autonomy: a lightweight runtime monitor that checks whether visual parsing,
depth, perceptual state regression, and physical state evidence are trustworthy
enough for downstream policy execution.

##  Summary

This project upgrades an RGB-D reliability workflow into a VPPV visual-front-end
monitor. It distills depth, temporal, embedding, trajectory, calibration, and
coverage-risk evidence into `visual_state_risk`, then routes the system through
`NORMAL`, `SUSPECT`, `RECOVER`, and `HUMAN_REVIEW` states.

## Best Application Framing

Use this project when writing to supervisors in reliable 3D scene
understanding, robot perception, embodied AI safety, human-robot collaboration,
or trustworthy machine learning for robotics.

Suggested wording:

> I built a VPPV-style perception reliability monitor for surgical autonomy.
> The monitor watches whether depth, temporal visual change, embedding shift,
> trajectory residuals, and calibration-style evidence indicate that the visual
> front end is unreliable, then routes the system toward re-perception,
> recovery, replanning, or human review.

## What To Read

| Time budget | File | Purpose |
|---|---|---|
| 2 minutes | [README](../README.md) | Main question, key results, figures, reproduction |
| 3 minutes | [VPPV monitor report](../reports/vppv_perception_reliability_monitor.md) | Main upgraded project narrative |
| 5 minutes | [PhD application brief](phd_application_project_brief.md) | Concise supervisor-facing project narrative |
| 10 minutes | [Application evidence pack](application_evidence_pack.md) | Evidence layer by layer |
| 15 minutes | [Supervisor one-pager](supervisor_one_pager.md) | One-page research summary |
| Deep dive | [Final report PDF](final_report.pdf) | Full report with figures and tables |

## Evidence Snapshot

| Claim | Current evidence | Strength |
|---|---|---|
| VPPV front-end risk can be distilled. | Random Forest teacher ROC-AUC 0.992 from depth/temporal/embedding/trajectory features. | Strong prototype evidence |
| Risk routes to concrete autonomy states. | 1350 NORMAL, 433 SUSPECT, 17 RECOVER, 0 HUMAN_REVIEW. | Auditable runtime evidence |
| Risk is linked to downstream signals. | Spearman risk vs trajectory residual 0.543; top 10% risk captures all RECOVER/HUMAN_REVIEW states. | Decision-relevant proxy evidence |
| Controlled RGB-D corruptions can be detected. | TUM RGB-D, 300 depth files, 1800 samples, source-paired ROC-AUC 1.000. | Strong for controlled corruptions |
| Naive clean-reference distance fails under motion. | TUM scene-conditioned baseline ROC-AUC 0.483. | Useful negative result |
| Temporal normalization improves reliability scoring. | TUM temporal excess ROC-AUC 1.000 with +/- 5 frame window. | Strong within current sample |
| Descriptor choice matters. | Rotation correlation improves from global 0.061 to grid 0.275 to PCA depth 0.540. | Good diagnostic evidence |
| Runtime states are auditable. | 1800 samples mapped to 1350 NORMAL, 423 SUSPECT, 27 RECOVER. | Prototype monitor evidence |
| Action outcomes can be monitored. | Synthetic trajectory residual benchmark ROC-AUC 0.990. | Good smoke test, not robot-log validation |

## What Is Shown

- A working VPPV-style monitor script:
  `modules/run_vppv_perception_monitor.py`.
- Public figures for architecture, dashboard, feature attribution, ablation,
  route policy, and risk traces.
- A working codebase for video embeddings, RGB-D/depth reliability, temporal
  scoring, calibration, and trajectory residual monitoring.
- Public-data TUM RGB-D evidence and compact figures.
- Explicit negative results showing where simple descriptors fail.
- A monitor vocabulary that can connect perception reliability to robot
  decision states.

## What Remains Unproven

- Closed-loop robot control validation.
- Natural task-native failure labels from a robot, SLAM system, or surgical
  tool tracker.
- A general pretrained RGB-D representation.
- Formal safety guarantees.
- Real-world recovery or human-review efficiency.

## Best-Fit Supervisor Directions

| Direction | How to position the project |
|---|---|
| VPPV surgical autonomy | Front-end visual-state reliability monitor for depth, perceptual state, and policy routing. |
| Reliable 3D scene understanding | Depth and point-cloud reliability under corruption and camera motion. |
| Robot perception | Runtime detection of unreliable RGB-D observations. |
| Embodied AI | Perception risk as a trigger for replanning, recovery, or clarification. |
| Human-robot collaboration | Monitor states for slowing down or requesting human review. |
| Surgical robotics | Transfer trajectory residuals to surgical tool drift and tracking quality. |

## Next Upgrade Before Submission

1. Add a VPPV-native failure label from segmentation quality, tool-state
   regression error, simulator rollouts, or surgical-tool tracking.
2. Compare against a stronger pretrained depth/RGB-D descriptor.
3. Add one short public-safe demo figure that shows the monitor timeline from
   input -> score -> state.
4. Keep `AUC=1.000` claims explicitly tied to controlled corruption labels.
