# Application Index

This page is the supervisor-facing entry point for the repository. The primary
framing is broad: a reliability-aware sequential robot perception project that
starts from CNN-LSTM perception, develops temporal and waveform-like reliability
analysis, and then transfers the monitor to downstream autonomy settings.

VPPV is not used as the project name. It is treated as one surgical autonomy
application case because that framework depends on visual parsing, depth,
perceptual state regression, and physical state vectors.

## Summary

This project asks when a robot should stop trusting its current visual state.
It distills depth, temporal, embedding, trajectory, calibration, and
coverage-risk evidence into `visual_state_risk`, then routes the system through
`NORMAL`, `SUSPECT`, `RECOVER`, and `HUMAN_REVIEW` states.

## Best Application Framing

Use this project when writing to supervisors in reliable 3D scene
understanding, robot perception, embodied AI safety, safe RL, human-robot
collaboration, surgical robotics, or trustworthy machine learning.

Suggested broad wording:

> I built a reliability-aware sequential robot perception monitor. It starts
> from CNN-LSTM visual sequence modeling, then analyzes abnormal temporal
> state changes in RGB-D/depth observations, and finally distills these signals
> into a runtime risk score that can trigger re-perception, recovery,
> replanning, or human review.

Suggested surgical-autonomy wording:

> As a transfer case, I adapted the monitor to a VPPV-style surgical autonomy
> front end, where unreliable depth, visual parsing, perceptual state
> regression, or action-outcome consistency could mislead a downstream policy.

## What To Read

| Time budget | File | Purpose |
|---|---|---|
| 2 minutes | [README](../README.md) | Main question, key results, figures, reproduction |
| 5 minutes | [PhD application brief](phd_application_project_brief.md) | Concise supervisor-facing project narrative |
| 10 minutes | [Project overview](project_overview.md) | Full technical narrative |
| 10 minutes | [Application evidence pack](application_evidence_pack.md) | Evidence layer by layer |
| 10 minutes | [Visual evidence index](VISUAL_EVIDENCE_INDEX.md) | Public figures and snapshot tables |
| 15 minutes | [Supervisor one-pager](supervisor_one_pager.md) | One-page research summary |
| Surgical transfer | [VPPV transfer report](../reports/vppv_perception_reliability_monitor.md) | Application case for surgical autonomy |

## Evidence Snapshot

| Claim | Current evidence | Strength |
|---|---|---|
| Visual-state risk can be distilled. | Random Forest teacher ROC-AUC 0.992 from depth/temporal/embedding/trajectory features. | Strong prototype evidence |
| Risk routes to concrete autonomy states. | 1350 NORMAL, 433 SUSPECT, 17 RECOVER, 0 HUMAN_REVIEW. | Auditable runtime evidence |
| Risk is linked to downstream signals. | Spearman risk vs trajectory residual 0.543; top 10% risk captures all RECOVER/HUMAN_REVIEW states. | Decision-relevant proxy evidence |
| Controlled RGB-D corruptions can be detected. | TUM RGB-D, 300 depth files, 1800 samples, source-paired ROC-AUC 1.000. | Strong for controlled corruptions |
| Naive clean-reference distance fails under motion. | TUM scene-conditioned baseline ROC-AUC 0.483. | Useful negative result |
| Temporal normalization improves reliability scoring. | TUM temporal excess ROC-AUC 1.000 with +/- 5 frame window. | Strong within current sample |
| Descriptor choice matters. | Rotation correlation improves from global 0.061 to grid 0.275 to PCA depth 0.540. | Good diagnostic evidence |
| Runtime states are auditable. | 1800 samples mapped to 1350 NORMAL, 423 SUSPECT, 27 RECOVER. | Prototype monitor evidence |
| Action outcomes can be monitored. | Synthetic trajectory residual benchmark ROC-AUC 0.990. | Good smoke test, not robot-log validation |

## What Is Shown

- A CNN-LSTM starting point for sequential visual perception.
- A working codebase for video embeddings, RGB-D/depth reliability, temporal
  scoring, calibration, and trajectory residual monitoring.
- Public-data TUM RGB-D evidence and compact figures.
- Explicit negative results showing where simple descriptors fail.
- A distilled `visual_state_risk` monitor with attribution, ablation, and
  route-state outputs.
- A surgical autonomy transfer case implemented in
  `modules/run_vppv_perception_monitor.py`.

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
| Reliable robot perception | Runtime detection of unreliable RGB-D observations and state jumps. |
| Trustworthy ML | Risk distillation, feature attribution, calibration, and selective prediction. |
| Embodied AI / safe RL | Perception risk as a trigger for replanning, recovery, or clarification. |
| Human-robot collaboration | Monitor states for slowing down or requesting human review. |
| Reliable 3D scene understanding | Depth and point-cloud reliability under corruption and camera motion. |
| Surgical robotics / VPPV | Optional transfer case for front-end visual-state reliability. |

## Next Upgrade Before Submission

1. Keep the public title broad and method-centered rather than VPPV-centered.
2. Add one task-native failure label from robot logs, SLAM tracking, simulator
   rollouts, surgical-tool tracking, or segmentation quality.
3. Compare against a stronger pretrained depth/RGB-D descriptor.
4. Keep `AUC=1.000` claims explicitly tied to controlled corruption labels.
