# One-Page Research Summary

## Project Title

VPPV-Style Perception Reliability Monitor for Surgical Autonomy

## Research Motivation

VPPV-style surgical autonomy depends on visual parsing, depth maps, perceptual
regressors, and physical state vectors. If those front-end states are unstable
or corrupted, the downstream policy can be driven by an incorrect state. This
project asks when the autonomy stack should stop trusting its current visual
state and trigger re-perception, recovery, replanning, or human review.

## What I Built

- A VPPV-facing `visual_state_risk` monitor distilled from depth, temporal,
  embedding, trajectory, calibration, and coverage-risk signals.
- A runtime state machine with `NORMAL`, `SUSPECT`, `RECOVER`, and
  `HUMAN_REVIEW` states.
- Feature attribution and high-risk case explanations, so the monitor is
  interpretable rather than only a scalar score.
- Signal-group ablation to compare depth-only, temporal-only, embedding-only,
  trajectory-only, calibration-only, and all-signal variants.
- Supporting TUM RGB-D, pose-aware descriptor, calibration, and trajectory
  residual experiments.

## Key Evidence

| Experiment | Result |
|---|---:|
| VPPV risk distillation | Random Forest teacher ROC-AUC 0.992 |
| Runtime route states | 1350 NORMAL / 433 SUSPECT / 17 RECOVER / 0 HUMAN_REVIEW |
| Top-risk state capture | Top 10% risk captures 100% RECOVER/HUMAN_REVIEW |
| Risk vs trajectory residual | Spearman 0.543 |
| Risk vs temporal excess | Spearman 0.521 |
| TUM scene-conditioned baseline | ROC-AUC 0.483 |
| TUM temporal excess scoring | ROC-AUC 1.000 |
| Trajectory residual failure detection | ROC-AUC 0.990 |

## Main Research Lesson

VPPV visual reliability should not be judged only by distance from a global
clean reference. Surgical scenes contain normal camera, tool, and tissue motion.
The more useful question is whether the current state change exceeds normal
variation in a local temporal window, and whether that abnormal state should
route the system to re-perception, recovery, or human review.

## Fit To Supervisor Directions

- Surgical autonomy / VPPV: visual-state reliability before policy execution.
- Trustworthy ML: risk distillation, attribution, ablation, calibration, and
  selective prediction.
- Runtime assurance: auditable state machine and route policy.
- Robot perception: RGB-D/depth reliability under corruption and camera motion.
- Safe RL: risk scores as recovery triggers, rollout filters, or constraints.

## Limitations

- The current labels are VPPV-style proxies, not paired surgical policy
  rollouts.
- Segmentation-mask quality and perceptual regressor error are framed as VPPV
  dependencies but are not directly measured yet.
- Controlled corruptions are not the same as natural surgical perception
  failures.
- Runtime states are transparent engineering rules, not formal safety proofs.

## Next Step

Evaluate `visual_state_risk` against VPPV-native evidence: segmentation quality,
tool-state regression error, simulator rollouts, surgical-tool tracking logs, or
downstream policy failure labels.

## Further Reading

- Main report: `reports/vppv_perception_reliability_monitor.md`
- Visual evidence: `docs/VISUAL_EVIDENCE_INDEX.md`
- Project overview: `docs/project_overview.md`
- Limitations: `docs/limitations.md`
