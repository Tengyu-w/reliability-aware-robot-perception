# Reliability-Aware Sequential and 3D Robot Perception

This repository is a research prototype for detecting unreliable perception and
action outcomes in sequential models, RGB-D robot perception, and trajectory
execution. The work began from a CNN-LSTM video action baseline and was extended
into a broader reliability-analysis workflow for trustworthy embodied AI.

The central question is:

> Can embedding-space evidence, temporal consistency, calibration analysis, and
> action residuals identify when a perception-action system is unreliable enough
> to slow down, recover, replan, or request human review?

## Reading Guide

| Document | Purpose |
|---|---|
| [`docs/final_report.pdf`](docs/final_report.pdf) | Full report with the most complete narrative, figures, and tables |
| [`docs/project_overview.md`](docs/project_overview.md) | Short technical overview of the research question and pipeline |
| [`docs/application_evidence_pack.md`](docs/application_evidence_pack.md) | Compact evidence summary for supervisors or reviewers |
| [`docs/experiment_order.md`](docs/experiment_order.md) | Recommended order for reading and rerunning the experiments |
| [`docs/limitations.md`](docs/limitations.md) | Scope, limitations, and next validation steps |
| [`modules/readme.md`](modules/readme.md) | Code map by experimental component |

## What This Project Does

1. Trains a CNN-LSTM video action baseline and extracts validation embeddings.
2. Builds depth and point-cloud reliability diagnostics for RGB-D perception.
3. Runs synthetic and TUM RGB-D corruption benchmarks.
4. Tests temporal reliability under camera motion.
5. Compares global, local-grid, and PCA depth descriptors against pose changes.
6. Converts reliability scores into auditable runtime states:
   `NORMAL`, `SUSPECT`, `RECOVER`, and `HUMAN_REVIEW`.
7. Adds planned-vs-observed trajectory residual monitoring for action outcomes.

## Key Results

| Evidence layer | Setup | Result | Interpretation |
|---|---:|---:|---|
| Synthetic 3D reliability | 3 seeds, 8 samples per scene | ROC-AUC 0.804 +/- 0.028 | Embedding risk gives a reproducible smoke-test signal |
| TUM RGB-D corruption | 300 depth files, 1800 samples | source-paired ROC-AUC 1.000 | Controlled depth corruptions are detectable |
| TUM scene-conditioned baseline | Same TUM run | ROC-AUC 0.483 | Global clean references fail under camera motion |
| TUM temporal reliability | +/- 5 frame window | temporal excess ROC-AUC 1.000 | Local temporal normalization improves reliability scoring |
| Pose-aware global descriptor | 299 adjacent frame pairs | rotation corr. 0.061 | Global statistics are weakly pose-aware |
| Pose-aware grid descriptor | 299 adjacent frame pairs | rotation corr. 0.275 | Local layout improves rotation sensitivity |
| PCA depth descriptor | 32 components | rotation corr. 0.540 | Lightweight learned depth descriptors are more promising |
| Runtime monitor | 1800 TUM temporal samples | 1350 NORMAL / 423 SUSPECT / 27 RECOVER | Scores become auditable runtime states |
| Calibration | TUM temporal risk scores | ROC-AUC 1.000; ECE gap 0.758 | Ranking is strong, raw scores are not calibrated probabilities |
| Trajectory residual | 400 synthetic action-outcome samples | ROC-AUC 0.990 | Planned-vs-observed residuals detect execution failures |

The CSV version of this table is in
[`docs/tables/key_results.csv`](docs/tables/key_results.csv).

## Selected Figures

These figures are copied from local experiment outputs so the repository can
show representative evidence without committing the full `outputs/` directory.

| Temporal reliability | Runtime monitoring |
|---|---|
| ![TUM temporal coverage-risk curve](docs/figures/tum_temporal_coverage_risk.png) | ![TUM runtime monitor trace](docs/figures/tum_runtime_monitor_trace.png) |

| Calibration | Trajectory residuals |
|---|---|
| ![TUM risk calibration](docs/figures/tum_risk_calibration.png) | ![Trajectory residual examples](docs/figures/trajectory_residual_examples.png) |

More selected figures are listed in [`docs/figures/README.md`](docs/figures/README.md).

## Code Layout

```text
modules/
  main.py                              # CNN-LSTM training and prediction entry point
  embedding_analysis.py                # Sequential embedding diagnostics
  robot_3d_reliability.py              # Depth/point-cloud reliability utilities
  run_robot_3d_demo.py                 # Synthetic 3D reliability demo
  run_robot_3d_multiseed.py            # Multi-seed synthetic benchmark
  run_robot_3d_pipeline.py             # Real-depth profile + corruption benchmark
  run_temporal_depth_benchmark.py      # Temporal local-reference reliability
  run_tum_pose_embedding_analysis.py   # Global/grid descriptor pose analysis
  run_tum_pca_depth_descriptor.py      # PCA depth descriptor baseline
  runtime_monitor.py                   # Runtime state monitor
  calibration_analysis.py              # Calibration and coverage-risk analysis
  trajectory_residual_demo.py          # Action-outcome residual reliability demo
  build_evidence_pack.py               # Evidence-pack generator
```

See [`modules/readme.md`](modules/readme.md) and
[`docs/experiment_order.md`](docs/experiment_order.md) for the recommended
reading order.

## Installation

```bash
pip install -r modules/requirements.txt
```

Install a CUDA-enabled PyTorch build if GPU training is needed.

## Minimal Reproduction

The synthetic 3D and trajectory-residual experiments can be run without a robot
dataset:

```bash
python modules/run_robot_3d_demo.py --output-dir outputs/robot_3d_demo
python modules/run_robot_3d_multiseed.py --output-dir outputs/robot_3d_multiseed --seeds 1 2 3 --samples-per-scene 8
python modules/trajectory_residual_demo.py --output-dir outputs/trajectory_residual_demo
```

For the TUM RGB-D workflow:

```bash
python modules/download_tum_rgbd_sample.py \
  --sequence freiburg1_desk \
  --raw-dir data/raw/tum_rgbd \
  --prepared-dir data/prepared_depth/tum_rgbd_freiburg1_desk \
  --max-files 300

python modules/run_temporal_depth_benchmark.py \
  --depth-dir data/prepared_depth/tum_rgbd_freiburg1_desk \
  --output-dir outputs/tum_rgbd_freiburg1_desk_temporal \
  --depth-scale 5000 \
  --max-files 300 \
  --window 5

python modules/runtime_monitor.py \
  --input-csv outputs/tum_rgbd_freiburg1_desk_temporal/temporal_depth_embeddings.csv \
  --output-dir outputs/tum_rgbd_runtime_monitor \
  --score-column temporal_excess_score
```

Raw data, prepared subsets, checkpoints, and generated outputs are intentionally
not tracked by Git. See [`data/README.md`](data/README.md).

## Evidence Pack

After running experiments, rebuild the compact evidence summary with:

```bash
python modules/build_evidence_pack.py \
  --project-root . \
  --output docs/application_evidence_pack.md
```

Dataset card drafts are provided for TUM RGB-D, NYU Depth V2, KITTI depth, and
SUN RGB-D under [`docs/dataset_cards/`](docs/dataset_cards/).

## Scope

This is a prototype baseline, not a finished benchmark or a certified safety
system. The current results support a reliability-analysis workflow, show useful
failure cases, and motivate stronger descriptors and task-native validation.
The main limitations and next experiments are summarized in
[`docs/limitations.md`](docs/limitations.md).
