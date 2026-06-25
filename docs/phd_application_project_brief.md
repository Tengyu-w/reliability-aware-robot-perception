# PhD Application Project Brief

## Title

Reliability-Aware Sequential Robot Perception

## One-Paragraph Summary

This project studies when a robot should stop trusting its current visual
state. It starts from CNN-LSTM sequential perception, then develops RGB-D/depth
reliability analysis, temporal state-change diagnostics, and waveform-like
excess scoring. These signals are distilled into a lightweight
`visual_state_risk` score that can route the system to continue, re-perceive,
recover, replan, or request human review. A VPPV-style surgical autonomy front
end is included as one transfer case, not as the overall project identity.

## Research Motivation

Robot learning systems often treat perception outputs as clean state inputs.
However, visual embeddings, depth maps, and regressed states can become
unreliable under corruption, camera motion, occlusion, calibration drift, or
action-outcome mismatch. A useful autonomy stack should monitor not only what
it sees, but also whether the current visual state is stable enough for policy
execution.

## Method Overview

The project has six technical layers:

1. CNN-LSTM sequential video/action baseline and embedding diagnostics.
2. Synthetic depth/point-cloud corruption reliability tests.
3. TUM RGB-D corruption and temporal reliability benchmarks.
4. Pose-aware descriptor comparison: global, local-grid, and PCA depth
   descriptors.
5. Visual-state risk distillation with feature attribution and signal-group
   ablation.
6. Runtime route policy, trajectory residual monitoring, and surgical-front-end
   transfer.

## Key Evidence

| Component | Evidence |
|---|---|
| Visual-state risk distillation | Random Forest teacher ROC-AUC 0.992 |
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

The strongest research message is that visual reliability cannot be judged only
by distance from a global clean reference. In robot perception, normal camera
and scene motion can make a global reference misleading. The monitor should ask
whether the current visual state change exceeds normal variation inside a local
temporal window. This turns the project into a reliability monitor for dynamic
visual state, rather than only a generic corruption detector.

## VPPV-Style Surgical Transfer

The VPPV-style section is the strongest domain-specific application of the
project. It connects the general monitor to a surgical autonomy stack whose
front end depends on visual parsing, depth maps, perceptual state regression,
and physical state vectors. The project contributes a companion reliability
monitor rather than a replacement for those modules.

In this transfer case, `visual_state_risk` is used as a decision-facing signal:

- `NORMAL`: visual state appears stable enough for continued policy execution.
- `SUSPECT`: visual state is questionable; slow down or re-perceive.
- `RECOVER`: visual abnormality may affect execution; recover, replan, or use
  backup behavior.
- `HUMAN_REVIEW`: state cannot be confirmed; request operator review.

The current evidence supports this mapping with risk distillation, feature
attribution, signal-group ablation, route-state counts, and trajectory-residual
correlation. The limitation is also explicit: these are VPPV-style proxy labels,
not paired VPPV policy rollouts.

## Limitations

- Controlled corruptions are not equivalent to natural robot failures.
- Current labels are proxy reliability labels, not paired closed-loop robot
  rollouts.
- Runtime monitor states are transparent engineering rules, not formal safety
  guarantees.
- The trajectory residual benchmark is synthetic.
- The VPPV-style section is a substantial application transfer case, not a
  claim that this project reproduces or renames VPPV.
- Stronger pretrained RGB-D descriptors and task-native failure labels are
  needed before making broad real-world claims.

## Suitable Application Framing

> I developed a reliability-aware sequential robot perception monitor. The
> project starts from CNN-LSTM visual sequence modeling, then studies RGB-D
> state reliability and local temporal state changes, and finally distills
> these signals into a runtime risk score for re-perception, recovery,
> replanning, or human review.

For surgical robotics only:

> As one transfer case, I apply the same monitor to a VPPV-style surgical
> autonomy front end, where unreliable depth, visual parsing, perceptual state
> regression, or trajectory consistency could mislead a downstream policy.

## What I Would Improve Next

1. Evaluate the monitor against task-native robot logs, simulator rollouts, or
   surgical-tool logs.
2. Replace PCA with a stronger pretrained depth/RGB-D encoder.
3. Add SLAM tracking quality, pose error, segmentation quality, or tool-state
   tracking as task-native reliability labels.
4. Measure human-review or replanning burden, not only ROC-AUC.
