# PhD Application Project Brief

## Title

VPPV-Style Perception Reliability Monitor for Surgical Autonomy

## One-Paragraph Summary

This project studies when the visual front end of a VPPV-style surgical
autonomy system should not be trusted. It upgrades an RGB-D/depth reliability
workflow into a monitor for visual parsing, depth, perceptual state regression,
physical state evidence, and downstream trajectory consistency. The monitor
distills depth, temporal, embedding, trajectory, calibration, and coverage-risk
signals into `visual_state_risk`, then routes the system to continue,
re-perceive, recover, replan, or request human review.

## Research Motivation

VPPV-style autonomy depends on visual parsing, depth, perceptual regressors, and
physical state vectors. If those front-end states are unreliable, the downstream
policy can be driven by the wrong state. A useful autonomy stack should
therefore monitor not only what it sees, but also whether the current visual
state is stable enough for policy execution.

## Method Overview

The project has six technical layers:

1. Sequential video action baseline and embedding diagnostics.
2. Synthetic depth/point-cloud corruption reliability tests.
3. TUM RGB-D corruption and temporal reliability benchmarks.
4. Pose-aware descriptor comparison: global, local-grid, and PCA depth
   descriptors.
5. VPPV visual-state risk distillation with feature attribution and
   signal-group ablation.
6. Runtime route policy and trajectory residual monitoring.

## Key Evidence

| Component | Evidence |
|---|---|
| VPPV risk distillation | Random Forest teacher ROC-AUC 0.992 |
| Runtime route states | 1350 NORMAL / 433 SUSPECT / 17 RECOVER / 0 HUMAN_REVIEW |
| Outcome-linked validation | Top 10% risk captures 100% RECOVER/HUMAN_REVIEW states |
| Feature attribution | embedding_shift, trajectory_residual, and progress_slope dominate the student model |
| Controlled RGB-D corruption | TUM source-paired ROC-AUC 1.000 over 1800 samples |
| Negative baseline | Scene-conditioned ROC-AUC 0.483 under camera motion |
| Temporal scoring | Temporal excess ROC-AUC 1.000 |
| Descriptor comparison | Rotation correlation: global 0.061, grid 0.275, PCA depth 0.540 |
| Runtime monitor | 1350 NORMAL / 423 SUSPECT / 27 RECOVER over 1800 samples |
| Trajectory residual | Synthetic action-outcome failure detection ROC-AUC 0.990 |

## Strongest Current Result

The strongest research message is that VPPV visual reliability cannot be judged
only by distance from a global clean reference. In surgery, camera, tool, and
tissue motion are normal. The monitor should ask whether the current visual
state change exceeds the normal variation in a local temporal window. This
turns the project into a VPPV front-end reliability monitor rather than a
generic RGB-D corruption detector.

## Limitations

- Controlled corruptions are not equivalent to natural robot failures.
- Current labels are VPPV-style proxies, not paired surgical policy rollouts.
- Segmentation-mask reliability is framed as a VPPV dependency but is not yet
  directly measured.
- Runtime monitor states are transparent engineering rules, not formal safety
  guarantees.
- The trajectory residual benchmark is synthetic.
- Stronger pretrained RGB-D descriptors and task-native failure labels are
  needed before making broad real-world claims.

## Suitable Application Framing

> I developed a VPPV-style visual-front-end reliability monitor for surgical
> autonomy. The project uses depth validity, temporal visual change, embedding
> shift, calibration-style evidence, and trajectory residuals to decide whether
> the system should continue policy execution, re-perceive, recover, replan, or
> request human review.

## What I Would Improve Next

1. Evaluate the monitor against VPPV simulator rollouts or surgical-tool logs.
2. Replace PCA with a stronger pretrained depth/RGB-D encoder.
3. Add SLAM tracking quality, pose error, or surgical-tool tracking as
   task-native reliability labels.
4. Measure human-review or replanning burden, not only ROC-AUC.
