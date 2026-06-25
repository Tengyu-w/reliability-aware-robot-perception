# One-Page Research Summary

## Project Title

Reliability-Aware Sequential Robot Perception

## Research Motivation

Robot policies often assume that their visual state is trustworthy. In practice,
visual embeddings, depth maps, and perceived task states can jump because of
camera motion, occlusion, corruption, calibration drift, or action-outcome
mismatch. This project asks when a robot should stop trusting its current visual
state and trigger re-perception, recovery, replanning, or human review.

## What I Built

- A CNN-LSTM-based starting point for sequential visual perception.
- RGB-D/depth reliability experiments under controlled corruption and camera
  motion.
- Temporal and waveform-like excess scores that compare the current visual
  change against a local time window rather than a fixed global reference.
- A distilled `visual_state_risk` monitor using depth, temporal, embedding,
  trajectory, calibration, and coverage-risk signals.
- A runtime state machine with `NORMAL`, `SUSPECT`, `RECOVER`, and
  `HUMAN_REVIEW` states.
- Feature attribution, signal-group ablation, and high-risk case explanations.
- A surgical autonomy transfer case for VPPV-style visual front ends.

## Key Evidence

| Experiment | Result |
|---|---:|
| Visual-state risk distillation | Random Forest teacher ROC-AUC 0.992 |
| Runtime route states | 1350 NORMAL / 433 SUSPECT / 17 RECOVER / 0 HUMAN_REVIEW |
| Top-risk state capture | Top 10% risk captures 100% RECOVER/HUMAN_REVIEW |
| Risk vs trajectory residual | Spearman 0.543 |
| Risk vs temporal excess | Spearman 0.521 |
| TUM scene-conditioned baseline | ROC-AUC 0.483 |
| TUM temporal excess scoring | ROC-AUC 1.000 |
| Trajectory residual failure detection | ROC-AUC 0.990 |

## Main Research Lesson

Visual reliability should not be judged only by distance from a global clean
reference. In robot perception, normal camera and scene motion can make a global
reference misleading. The more useful question is whether the current state
change exceeds normal variation in a local temporal window, and whether that
abnormal state should route the system to re-perception, recovery, or human
review.

## Fit To Supervisor Directions

- Reliable robot perception: RGB-D/depth reliability under corruption and
  camera motion.
- Trustworthy ML: risk distillation, attribution, ablation, calibration, and
  selective prediction.
- Runtime assurance: auditable state machine and route policy.
- Embodied AI / safe RL: risk scores as recovery triggers, rollout filters, or
  constraints.
- Surgical autonomy / VPPV: optional transfer case for visual-state reliability
  before policy execution.

## Limitations

- Current labels are proxy reliability labels, not paired closed-loop robot
  failure labels.
- Controlled corruptions are not the same as natural real-world perception
  failures.
- Runtime states are transparent engineering rules, not formal safety proofs.
- The VPPV-style section is an application transfer case, not a claim that this
  project reproduces or renames VPPV.

## Next Step

Evaluate `visual_state_risk` against task-native evidence: SLAM tracking loss,
segmentation quality, tool-state regression error, simulator rollouts,
robot-log failures, or downstream policy failure labels.

## Further Reading

- Project overview: `docs/project_overview.md`
- Visual evidence: `docs/VISUAL_EVIDENCE_INDEX.md`
- Surgical transfer case: `reports/vppv_perception_reliability_monitor.md`
- Limitations: `docs/limitations.md`
