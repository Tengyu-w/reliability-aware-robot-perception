# Reliability-Aware Sequential and 3D Perception Evidence Pack

## Positioning

One-page summary: `docs/supervisor_one_pager.md`.

This project is a reliability-aware robot perception prototype. It begins with
CNN-LSTM sequential perception, then develops RGB-D/depth reliability,
temporal-state diagnostics, calibration checks, and trajectory residual
monitoring. The central question is whether visual evidence can identify an
unreliable state before a downstream robot policy is trusted.

The VPPV-style surgical autonomy material is kept as a transfer case, not as
the overall project name.

## Evidence Layers

| Layer | What it demonstrates | Current status |
|---|---|---|
| Video action recognition | Sequential perception embeddings and action-class reliability | Implemented in `modules/main.py` and `modules/embedding_analysis.py` |
| Synthetic 3D perception | Depth-to-point-cloud embedding reliability under controlled corruptions | Implemented and runnable |
| Real depth workflow | Public depth-map preparation, profiling, and corruption benchmark | Implemented; TUM RGB-D sample run completed in current outputs |
| Temporal reliability | Local-window state-change scoring under camera motion | Implemented for TUM RGB-D |
| Runtime risk monitor | Visual-state risk distillation and route-state policy | Implemented in `modules/run_vppv_perception_monitor.py` |
| Surgical autonomy transfer | VPPV-style front-end reliability mapping | Implemented as an application case |

## Current 3D Reliability Results

### Visual-State Risk Distillation

- Samples: 1800
- Distilled risk output: `visual_state_risk`
- Selected model: Random Forest
- Teacher ROC-AUC: 0.992
- Teacher average precision: 0.982
- Runtime states: 1350 NORMAL / 433 SUSPECT / 17 RECOVER / 0 HUMAN_REVIEW
- Spearman risk vs trajectory residual: 0.543
- Spearman risk vs temporal excess: 0.521
- Top 10% risk captures RECOVER/HUMAN_REVIEW states: 1.000
- Interpretation: the monitor provides a front-end reliability score and maps
  high-risk visual states to re-perception, recovery, replanning, or human
  review.

### Multi-Seed Synthetic Depth Benchmark

- Seeds: [1, 2, 3]
- Samples per scene: 8
- Embedding-risk ROC-AUC: 0.804 +/- 0.028
- Coverage-risk AUC: 0.693 +/- 0.022

### Real-Depth Corruption Benchmark

- Source files: 300
- Samples: 1800
- Embedding-risk ROC-AUC: 1.000
- Scene-conditioned ROC-AUC: 0.483
- Coverage-risk AUC: 0.535

### TUM RGB-D Temporal Reliability Benchmark

- Source files: 300
- Samples: 1800
- Temporal window: +/- 5 frames
- Temporal excess ROC-AUC: 1.000
- Temporal local-distance ROC-AUC: 0.483
- Source-paired ROC-AUC: 1.000

### TUM RGB-D Pose-Aware Embedding Shift Analysis

- Adjacent frame pairs: 299
- Global Spearman translation vs embedding shift: 0.046
- Global Spearman rotation vs embedding shift: 0.061
- Grid Spearman translation vs embedding shift: -0.064
- Grid Spearman rotation vs embedding shift: 0.275
- Interpretation: local grid descriptors improve rotation sensitivity over
  global statistics, but neither descriptor is sufficient as a full SLAM-aware
  representation.

### TUM RGB-D Learned PCA Depth Descriptor Baseline

- Adjacent frame pairs: 299
- PCA components: 32
- Explained variance ratio: 0.956
- Spearman translation vs embedding shift: 0.167
- Spearman rotation vs embedding shift: 0.540
- Interpretation: a lightweight learned depth descriptor tracks camera rotation
  better than hand-crafted global/grid descriptors.

### Runtime Assurance Monitor

- Samples: 1800
- NORMAL: 1350
- SUSPECT: 423
- RECOVER: 27
- HUMAN_REVIEW: 0
- Safety-property violations: 0
- Interpretation: reliability scores are converted into auditable autonomy
  states.

### Reliability Score Calibration

- Samples: 1800
- ROC-AUC: 1.000
- Average precision: 1.000
- ECE-style gap: 0.758
- Risk at full coverage: 0.833
- Risk at 80% coverage: 0.792
- Risk at 50% coverage: 0.667
- Interpretation: the score ranks corruptions well, but raw normalized scores
  are not probability-calibrated.

### Trajectory Residual Reliability Demo

- Samples: 400
- Failure types: normal, drift, oscillation, jump, stuck
- ROC-AUC: 0.990
- Average precision: 0.998
- Interpretation: planned-vs-observed trajectory residuals provide a
  control-facing action-outcome reliability score.

### Trajectory Runtime Monitor

- Samples: 400
- NORMAL: 300
- SUSPECT: 60
- RECOVER: 40
- HUMAN_REVIEW: 0

### Video Sequential Perception Benchmark

- Code is implemented; run with real video paths to produce metrics.

## Supervisor Fit

| Supervisor direction | Matching evidence |
|---|---|
| Reliable robot perception | RGB-D/depth reliability, temporal state-change analysis, runtime monitor |
| Trustworthy ML / calibration / robustness | Embedding risk, uncertainty scores, selective prediction, multi-seed reporting |
| Human-robot collaboration / industrial AI | Action recognition reliability and 3D perception screening around tools/workcells |
| Reliable 3D scene understanding | Depth-to-point-cloud embeddings and corruption detection |
| Embodied navigation / autonomous systems | Perception risk as a trigger for abstention, fallback, or human review |
| Medical or clinical trustworthy AI | ECG reliability project plus transferable reliability methodology |
| Surgical autonomy / VPPV | Application case for front-end visual-state reliability |

## VPPV-Style Surgical Autonomy Transfer

The VPPV-style transfer case is included because it makes the reliability
monitor concrete in a high-stakes robot learning setting. A VPPV-style autonomy
pipeline can depend on visual parsing, depth, perceptual regression, physical
state, and policy execution. The project provides a companion monitor for those
front-end states:

| Failure source | Monitor signal |
|---|---|
| Depth estimation error | depth validity, mean depth, depth variance, depth corruption score |
| Visual state jump | temporal local distance, temporal excess score |
| Perceptual state off-distribution | embedding shift |
| Action outcome mismatch | trajectory residual |
| Calibration or observation confidence issue | calibration gap, coverage-risk score |
| Task stagnation | progress slope, progress stagnation score |

The important interpretation is that VPPV-style reliability should be local and
temporal, not only global-reference based. Surgical camera motion, tool motion,
and tissue motion can be expected; the monitor focuses on whether the current
change exceeds the recent local pattern.

## What Is Shown

- The project has working code for video embeddings and 3D geometry embeddings.
- The 3D module can detect controlled perception corruptions using
  embedding-distance risk.
- TUM RGB-D results show that naive global/local distance can fail under camera
  motion, motivating temporal excess scoring.
- Pose-aware analysis shows a limitation: hand-crafted descriptors only weakly
  track frame-to-frame camera motion, though local grids improve rotation
  sensitivity.
- A lightweight learned PCA depth descriptor improves pose-motion sensitivity,
  especially for rotation.
- Runtime monitoring converts reliability scores into
  NORMAL/SUSPECT/RECOVER/HUMAN_REVIEW decisions with a simple property check.
- Calibration analysis separates risk ranking from probability calibration.
- Trajectory residual analysis extends the project from perception reliability
  to action-outcome reliability.
- The surgical transfer case turns RGB-D reliability and trajectory evidence
  into `visual_state_risk` with attribution, ablation, route policy, and
  outcome-linked validation.

## What Remains Unproven

- These results are not closed-loop robot validation.
- The surgical transfer case is not a reproduction of VPPV.
- Segmentation-mask reliability and perceptual regressor error are not yet
  directly measured.
- Synthetic or controlled corruptions do not replace dataset-native failure
  labels.
- Public-data results should be expanded before making strong claims about
  real-world robot perception.

## Next Experiment

Replace proxy residuals with task-native evidence: robot-log failures, SLAM
tracking quality, simulator rollouts, segmentation quality, surgical-tool
tracking, or perceptual regressor errors. Then evaluate whether
`visual_state_risk` predicts downstream policy failures.
