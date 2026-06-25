# Reliability-Aware Sequential Robot Perception

This repository is a research prototype for reliability monitoring in robot
perception. The project starts from a CNN-LSTM video/action recognition baseline,
then moves into temporal reliability and waveform-like state-change analysis,
and finally shows how the same monitor can be transferred to a surgical autonomy
front end such as a VPPV-style pipeline.

The central question is:

> When should a robot stop trusting its current visual state and trigger
> re-perception, recovery, replanning, or human review?

The main contribution is `visual_state_risk`: a lightweight risk score distilled
from depth, temporal, embedding, trajectory, calibration, and coverage-risk
signals. It is mapped into auditable runtime states: `NORMAL`, `SUSPECT`,
`RECOVER`, and `HUMAN_REVIEW`.

VPPV is treated here as one downstream application context, not as the name of
the project. This keeps the work usable for supervisors in reliable robot
perception, trustworthy ML, embodied AI, safe RL, medical robotics, and surgical
autonomy.

## Surgical Autonomy Transfer: VPPV-Style Front-End Monitoring

The surgical transfer case is intentionally more than a brief mention. VPPV-style
autonomy relies on a visual front end that can include segmentation masks, depth
maps, perceptual state regressors, and physical state vectors. If those states
are unstable or corrupted, a downstream policy may execute from the wrong state.

This project maps its general reliability monitor onto that setting:

| VPPV-style front-end dependency | Monitor evidence in this project |
|---|---|
| Depth map | depth validity, mean depth, depth variance, depth corruption score |
| Perceptual state / embedding | embedding shift and local temporal state change |
| Physical or task progress state | progress slope and progress stagnation score |
| Action outcome consistency | trajectory residual between planned and observed motion |
| Runtime autonomy decision | `NORMAL`, `SUSPECT`, `RECOVER`, `HUMAN_REVIEW` route states |

The main surgical insight is that reliability should not be judged only by
distance from a global clean reference. In laparoscopic or robot-assisted scenes,
camera motion, tool motion, and tissue motion can be normal. The monitor instead
asks whether the current visual-state change exceeds the normal variation inside
a local time window. This is why the VPPV-style case is a strong application of
the broader temporal reliability idea, while the project title remains general.

## Reading Guide

| Document | Purpose |
|---|---|
| [`docs/APPLICATION_INDEX.md`](docs/APPLICATION_INDEX.md) | Supervisor-facing entry point for PhD applications |
| [`docs/phd_application_project_brief.md`](docs/phd_application_project_brief.md) | Concise project brief with fit, evidence, and limits |
| [`docs/project_overview.md`](docs/project_overview.md) | Technical overview of the research question and pipeline |
| [`docs/application_evidence_pack.md`](docs/application_evidence_pack.md) | Compact evidence summary for supervisors or reviewers |
| [`docs/VISUAL_EVIDENCE_INDEX.md`](docs/VISUAL_EVIDENCE_INDEX.md) | Public figure and table index |
| [`reports/vppv_perception_reliability_monitor.md`](reports/vppv_perception_reliability_monitor.md) | Detailed VPPV-style surgical autonomy transfer case |
| [`docs/experiment_order.md`](docs/experiment_order.md) | Recommended order for reading and rerunning the experiments |
| [`docs/limitations.md`](docs/limitations.md) | Scope, limitations, and next validation steps |
| [`modules/readme.md`](modules/readme.md) | Code map by experimental component |

## What This Project Does

1. Starts from a CNN-LSTM sequential perception baseline.
2. Studies whether visual embeddings and temporal state changes can expose
   unreliable perception.
3. Tests RGB-D/depth reliability under controlled corruption and camera motion.
4. Shows a key negative result: global clean-reference distance can fail under
   normal camera motion.
5. Uses local temporal normalization and waveform-like excess scores to detect
   abnormal state changes.
6. Distills multiple reliability signals into `visual_state_risk`.
7. Converts risk scores into auditable runtime states for robot autonomy.
8. Demonstrates a surgical autonomy transfer case where the monitor can sit
   beside a VPPV-style visual front end.

## Key Results

| Evidence layer | Setup | Result | Interpretation |
|---|---:|---:|---|
| Risk distillation | 1800 aligned visual/action samples | Random Forest teacher ROC-AUC 0.992 | Lightweight `visual_state_risk` approximates heavier reliability evidence |
| Runtime route states | Distilled risk trace | 1350 NORMAL / 433 SUSPECT / 17 RECOVER / 0 HUMAN_REVIEW | Visual risk becomes concrete autonomy routing |
| Outcome link | Distilled risk vs residual signals | top 10% risk captures 100% RECOVER/HUMAN_REVIEW | Risk is decision-relevant, not only a teacher-fitting score |
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

| Reliability monitor architecture | Visual risk dashboard |
|---|---|
| ![Reliability monitor architecture](docs/figures/vppv_monitor_architecture.png) | ![Visual risk dashboard](docs/figures/vppv_visual_dashboard.png) |

| Feature attribution | Signal-group ablation |
|---|---|
| ![Feature importance](docs/figures/vppv_feature_importance.png) | ![Signal group ablation](docs/figures/vppv_signal_group_ablation.png) |

| Route policy | Risk trace |
|---|---|
| ![Route policy flow](docs/figures/vppv_route_policy_flow.png) | ![Risk trace](docs/figures/vppv_risk_trace.png) |

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
  run_vppv_perception_monitor.py       # Surgical-autonomy transfer case
  calibration_analysis.py              # Calibration and coverage-risk analysis
  trajectory_residual_demo.py          # Action-outcome residual reliability demo
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

python modules/run_vppv_perception_monitor.py
```

Raw data, prepared subsets, checkpoints, and generated outputs are intentionally
not tracked by Git. See [`data/README.md`](data/README.md).

## Evidence Pack

After running experiments, rebuild the surgical-autonomy transfer report and
visual gallery with:

```bash
python modules/run_vppv_perception_monitor.py
python tools/generate_vppv_visual_gallery.py
```

The public snapshot tables are stored in `docs/tables/vppv_*.csv` and
`docs/tables/vppv_risk_outcome_correlation.json`. The `vppv_` prefix marks the
transfer case artifact names, not the overall project name.

Dataset card drafts are provided for TUM RGB-D, NYU Depth V2, KITTI depth, and
SUN RGB-D under [`docs/dataset_cards/`](docs/dataset_cards/).

## Scope

This is a prototype baseline, not a finished benchmark or a certified safety
system. The current results support a reliability-analysis workflow, show useful
failure cases, and motivate stronger descriptors and task-native validation.
The main limitations and next experiments are summarized in
[`docs/limitations.md`](docs/limitations.md).
