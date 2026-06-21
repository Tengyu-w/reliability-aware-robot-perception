# One-Page Research Summary

## Project Title

Reliability-Aware Sequential and 3D Robot Perception

## Research Motivation

Modern embodied systems rely on learned perception and action models, but high
task accuracy is not enough for safety-critical robotics. A useful system should
also know when its perception, representation, or action outcome is unreliable
and should trigger recovery, replanning, or human review.

## What I Built

- A CNN-LSTM video action baseline with embedding distribution diagnostics.
- A TUM RGB-D reliability workflow for depth/point-cloud perception.
- Temporal and pose-aware analyses showing when naive embedding distance fails
  under camera motion.
- Descriptor comparison: global point-cloud statistics, local depth-grid
  descriptors, and a learned PCA depth descriptor.
- A runtime assurance monitor that converts reliability scores into NORMAL,
  SUSPECT, RECOVER, and HUMAN_REVIEW states.
- A synthetic trajectory residual benchmark for planned-vs-observed action
  outcome reliability.

## Key Evidence

| Experiment | Result |
|---|---:|
| TUM source-paired corruption detection | ROC-AUC 1.000 |
| TUM scene-conditioned baseline | ROC-AUC 0.483 |
| TUM temporal excess scoring | ROC-AUC 1.000 |
| Global descriptor vs camera rotation | Spearman 0.061 |
| Local grid descriptor vs camera rotation | Spearman 0.275 |
| PCA depth descriptor vs camera rotation | Spearman 0.540 |
| Runtime monitor | 1350 NORMAL / 423 SUSPECT / 27 RECOVER |
| Calibration | ROC-AUC 1.000, ECE-style gap 0.758 |
| Trajectory residual failure detection | ROC-AUC 0.990 |

## Main Research Lesson

Simple global embedding distance can detect controlled corruptions but fails
under normal RGB-D camera motion. Local and learned depth descriptors improve
pose sensitivity, especially for rotation. This motivates pose-normalized or
learned RGB-D representations for SLAM-aware reliability estimation.

## Fit To Supervisor Directions

- Trustworthy ML: calibration, selective prediction, distribution shift,
  transferability of reliability signals.
- Runtime assurance: auditable monitor states and simple safety-property checks.
- Embodied AI / navigation: perception risk as a trigger for replanning or
  clarification.
- Medical/surgical robotics: trajectory deviation, tool drift, and action
  outcome residuals as recovery triggers.
- Safe RL: residual/risk scores as rollout filters, constraints, or recovery
  rewards.

## Limitations

- TUM corruption labels are controlled, not dataset-native SLAM failures.
- PCA descriptors are sequence-fitted baselines, not general pretrained models.
- Runtime monitor rules are transparent prototypes, not formal safety proofs.
- Trajectory residuals are synthetic and should be replaced with simulator or
  robot logs.

## Next Step

Replace the PCA baseline with a pretrained or task-supervised RGB-D/depth
encoder, then evaluate reliability against tracking quality, pose error,
navigation progress, or surgical tool trajectory logs.

## Further Reading

- Full report: `docs/final_report.pdf`
- Experiment order: `docs/experiment_order.md`
- Limitations: `docs/limitations.md`
