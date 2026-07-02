# Visual-State Reliability for Industrial Action Recognition and Robot Perception

This repository studies how an industrial visual recognition system can be
extended into a reliability-aware robot perception monitor. The project begins
with CNN-LSTM human action recognition from image/video sequences, then asks
whether the same visual pipeline can estimate when its own state estimate is
too unreliable for autonomous use.

The work is a research prototype. It is not a certified safety system, not a
closed-loop robot validation result, and not a reproduction of any surgical
autonomy framework.

## Core Problem

Industrial robot perception cannot stop at recognizing an action or visual
state. A robot also needs to know when the current visual evidence is unstable,
out of distribution, corrupted, or inconsistent with downstream execution.

The central question is:

> Can image-sequence recognition, visual embeddings, depth reliability,
> temporal change, calibration evidence, and action-outcome residuals be turned
> into an auditable monitor that decides when to continue, re-perceive, recover,
> replan, or request human review?

## Research Path

The project is organized around a recognition-to-reliability progression:

```text
industrial image/video action recognition
  -> ResNet18 frame encoder + LSTM temporal classifier
  -> validation embeddings, confidence, entropy, and margins
  -> RGB-D/depth corruption and camera-motion stress tests
  -> local temporal excess scoring for abnormal visual-state changes
  -> trajectory residuals as downstream action-outcome evidence
  -> visual_state_risk distillation
  -> evidence-to-action runtime monitor
  -> VPPV-style surgical front-end transfer case
```

This structure keeps the initial visual-recognition capability visible while
making the reliability upgrade explicit.

## 1. Industrial Action Recognition Baseline

The first system is a CNN-LSTM video classifier for industrial human-action
recognition. It processes AVI videos or image sequences, samples a fixed number
of frames, extracts image features, models temporal motion, and predicts an
action class.

```text
AVI video / image sequence
  -> 16 sampled frames
  -> ResNet18 CNN frame encoder
  -> LSTM temporal model
  -> action-class prediction
  -> training curves, validation accuracy, test predictions
  -> validation embedding diagnostics
```

Relevant code:

- `modules/main.py`: training and prediction entry point.
- `modules/model.py`: ResNet18 + LSTM model.
- `modules/data_loader.py`: frame sampling and video loading.
- `modules/embedding_analysis.py`: validation embeddings, confidence,
  entropy, margins, PCA, and distribution summaries.

This stage establishes that the project can process visual sensor data and
perform temporal action recognition. The later reliability components are built
around the question of when such recognition should or should not be trusted.

## 2. Reliability Question

The reliability layer is motivated by three observations.

First, visual embeddings can reveal instability, but embedding separation is
not by itself a safety guarantee. A representation can look cleaner while the
downstream decision remains fragile.

Second, global clean-reference distance is not a sufficient reliability score
for robot perception. Normal camera motion, object motion, or tool motion can
change the visual state without indicating a perception failure.

Third, perception reliability must be connected to action outcomes. A visual
state is high risk when it can mislead downstream control, recovery, replanning,
or human-review decisions.

## 3. Evidence Families

The project separates reliability evidence into distinct signal families rather
than treating all uncertainty as one scalar.

| Evidence family | Measured signals | Reliability role |
| --- | --- | --- |
| Visual recognition | predicted class, confidence, entropy, probability margin | Identifies uncertain or low-margin action predictions. |
| Embedding geometry | validation embeddings, PCA, embedding shift | Detects representation drift and visual-state conflict. |
| Depth and RGB-D quality | valid-depth ratio, mean depth, depth variance, corruption score | Flags degraded or implausible depth evidence. |
| Temporal state change | local temporal distance, temporal excess score | Separates abnormal state jumps from normal motion. |
| Calibration and coverage | calibration gap, coverage-risk ranking | Distinguishes ranking quality from calibrated probability claims. |
| Action outcome | planned-vs-observed trajectory residual, progress slope | Connects perception reliability to downstream execution. |

This design mirrors the main methodological lesson from the related ECG
reliability project without reusing its naming scheme: diagnostic evidence
should generate testable reliability routes, not serve as proof by itself.

## 4. Evidence-To-Action Monitor

The current runtime design has two layers.

The first layer distills multi-source reliability evidence into
`visual_state_risk`, a lightweight score that can be computed from visual,
temporal, depth, trajectory, progress, calibration, and coverage signals.

The second layer converts risk and dominant evidence type into actions. The
implementation is intentionally transparent: the monitor exposes why a sample
is risky and routes it to a matching response.

```mermaid
flowchart TD
    A["Image sequence / RGB-D state"] --> B["CNN-LSTM and descriptor evidence"]
    B --> C["Reliability evidence table"]

    C --> D1["Recognition confidence"]
    C --> D2["Embedding shift"]
    C --> D3["Depth validity"]
    C --> D4["Temporal excess"]
    C --> D5["Trajectory residual"]
    C --> D6["Calibration / coverage"]

    D1 --> E["visual_state_risk"]
    D2 --> E
    D3 --> E
    D4 --> E
    D5 --> E
    D6 --> E

    E --> F["Runtime route policy"]
    F --> G1["NORMAL: continue"]
    F --> G2["SUSPECT: re-perceive"]
    F --> G3["RECOVER: recover or replan"]
    F --> G4["HUMAN_REVIEW: operator check"]

    E --> H["Reserved-budget route layer"]
    H --> I1["visual-state jump"]
    H --> I2["trajectory residual"]
    H --> I3["depth / signal quality"]
    H --> I4["representation conflict"]
    H --> I5["progress / calibration issue"]
```

The route policy is not a formal safety controller. It is an auditable
research wrapper that turns model evidence into explicit perception actions.

## 5. Main Findings

| Claim | Evidence | Interpretation |
| --- | --- | --- |
| CNN-LSTM provides the initial visual recognition layer. | ResNet18 frame encoder, LSTM sequence model, validation metrics, and exported predictions. | The project begins with image-sequence recognition rather than only post-hoc uncertainty analysis. |
| Controlled depth corruptions are detectable. | TUM RGB-D source-paired corruption benchmark, 300 depth files and 1800 samples. | The reliability pipeline detects designed depth failures, but this remains proxy evidence. |
| Global clean-reference distance can fail under camera motion. | TUM scene-conditioned baseline ROC-AUC 0.483. | Robot visual reliability should not be defined only as distance from a global clean state. |
| Local temporal normalization improves reliability scoring. | Temporal excess benchmark with +/- 5 frame local window, ROC-AUC 1.000. | Abnormal visual-state changes are better evaluated relative to local motion context. |
| Risk ranking is stronger than probability calibration. | TUM calibration analysis: ROC-AUC 1.000 with ECE gap 0.758. | Scores can rank risk well without being calibrated probabilities. |
| Action residuals provide execution-facing evidence. | Synthetic planned-vs-observed trajectory residual demo, ROC-AUC 0.990. | Reliability monitoring should connect perception to downstream outcomes. |
| Multi-source risk can be distilled into a compact runtime score. | VPPV-style risk distillation, Random Forest distillation ROC-AUC 0.992. | `visual_state_risk` approximates heavier reliability evidence. |
| A fixed-budget router can prioritize high-risk states. | 20% action budget captures 66.7% high-risk target cases and 76.5% RECOVER/HUMAN_REVIEW states. | The monitor supports review or recovery triage under limited action budget. |

The CSV version of the result snapshot is stored in
`docs/tables/key_results.csv`.

## 6. Current Evidence Snapshot

| Evidence layer | Setup | Result | Interpretation |
| --- | ---: | ---: | --- |
| CNN-LSTM action recognition | AVI video/image sequences | ResNet18 frame encoder + LSTM temporal classifier | Initial image-sequence recognition capability. |
| Risk distillation | 1800 aligned visual/action samples | Random Forest distillation ROC-AUC 0.992 | Compact risk score approximates richer reliability evidence. |
| Runtime route states | Distilled risk trace | 1350 NORMAL / 433 SUSPECT / 17 RECOVER / 0 HUMAN_REVIEW | Continuous risk becomes auditable autonomy routing. |
| Outcome link | Distilled risk vs residual signals | Top 10% risk captures 100% RECOVER/HUMAN_REVIEW | Risk is decision-relevant, not only fitted to the distillation target. |
| Reserved-budget router | 1800 aligned visual/action samples | 20% budget captures 66.7% high-risk target cases and 76.5% RECOVER/HUMAN_REVIEW | Scalar risk is decomposed into route-specific actions. |
| Synthetic 3D reliability | 3 seeds, 8 samples per scene | ROC-AUC 0.804 +/- 0.028 | Embedding-risk scoring gives a reproducible smoke-test signal. |
| TUM RGB-D corruption | 300 depth files, 1800 samples | Source-paired ROC-AUC 1.000 | Controlled corruptions are detectable in this setup. |
| TUM scene-conditioned baseline | Same TUM run | ROC-AUC 0.483 | Global clean references fail under normal camera motion. |
| TUM temporal reliability | +/- 5 frame window | Temporal excess ROC-AUC 1.000 | Local temporal normalization improves reliability scoring. |
| Pose-aware global descriptor | 299 adjacent frame pairs | Rotation corr. 0.061 | Global statistics are weakly pose-aware. |
| Pose-aware grid descriptor | 299 adjacent frame pairs | Rotation corr. 0.275 | Local layout improves rotation sensitivity. |
| PCA depth descriptor | 32 components | Rotation corr. 0.540 | Lightweight learned depth descriptors are more promising. |
| Calibration | TUM temporal risk scores | ROC-AUC 1.000; ECE gap 0.758 | Ranking is strong, but raw scores are not calibrated probabilities. |
| Trajectory residual | 400 synthetic action-outcome samples | ROC-AUC 0.990 | Planned-vs-observed residuals detect execution failures. |

## 7. Transfer Case: VPPV-Style Front-End Monitoring

The VPPV-style section is a transfer case for high-stakes visual front-end
monitoring. It does not rename the project and does not claim to reproduce or
validate VPPV.

| Front-end dependency | Monitor evidence |
| --- | --- |
| Depth map | valid-depth ratio, mean depth, depth variance, corruption score |
| Perceptual state or embedding | embedding shift, local temporal state change |
| Physical or task progress state | progress slope, progress stagnation |
| Action outcome consistency | planned-vs-observed trajectory residual |
| Runtime autonomy decision | `NORMAL`, `SUSPECT`, `RECOVER`, `HUMAN_REVIEW`, or route-specific review |

The careful claim is that the monitor is compatible with a VPPV-style visual
front end as a reliability wrapper. It has not been validated on paired
surgical robot logs, real segmentation masks, VPPV policy rollouts, or clinical
deployment data.

## 8. Selected Visual Evidence

| Reliability monitor architecture | Visual risk dashboard |
| --- | --- |
| ![Reliability monitor architecture](docs/figures/vppv_monitor_architecture.png) | ![Visual risk dashboard](docs/figures/vppv_visual_dashboard.png) |

| Feature attribution | Signal-group ablation |
| --- | --- |
| ![Feature importance](docs/figures/vppv_feature_importance.png) | ![Signal group ablation](docs/figures/vppv_signal_group_ablation.png) |

| Route policy | Risk trace |
| --- | --- |
| ![Route policy flow](docs/figures/vppv_route_policy_flow.png) | ![Risk trace](docs/figures/vppv_risk_trace.png) |

| Temporal reliability | Runtime monitoring |
| --- | --- |
| ![TUM temporal coverage-risk curve](docs/figures/tum_temporal_coverage_risk.png) | ![TUM runtime monitor trace](docs/figures/tum_runtime_monitor_trace.png) |

| Calibration | Trajectory residuals |
| --- | --- |
| ![TUM risk calibration](docs/figures/tum_risk_calibration.png) | ![Trajectory residual examples](docs/figures/trajectory_residual_examples.png) |

More selected figures are listed in `docs/figures/README.md`.

## 9. Repository Map

```text
modules/
  main.py                              CNN-LSTM training and prediction entry point
  model.py                             ResNet18 + LSTM model
  data_loader.py                       Video loading and frame sampling
  embedding_analysis.py                Embedding, confidence, entropy, PCA diagnostics
  robot_3d_reliability.py              Depth/point-cloud reliability utilities
  run_temporal_depth_benchmark.py      Local temporal-reference reliability
  run_tum_pose_embedding_analysis.py   Global/grid descriptor pose sensitivity
  run_tum_pca_depth_descriptor.py      PCA depth descriptor baseline
  calibration_analysis.py              Calibration and coverage-risk analysis
  trajectory_residual_demo.py          Action-outcome residual reliability demo
  runtime_monitor.py                   Risk scores to runtime states
  run_vppv_perception_monitor.py       Risk distillation and transfer case
  mechanism_router.py                  Reserved-budget route policy

docs/
  project_overview.md                  Technical overview
  mechanism_separated_routing_upgrade.md
  limitations.md                       Evidence boundary and next validation steps
  VISUAL_EVIDENCE_INDEX.md             Public figure and table index

reports/
  vppv_perception_reliability_monitor.md
```

## 10. Reproduce Key Runs

Install dependencies:

```bash
pip install -r modules/requirements.txt
```

Synthetic 3D and trajectory-residual smoke tests:

```bash
python modules/run_robot_3d_demo.py --output-dir outputs/robot_3d_demo
python modules/run_robot_3d_multiseed.py --output-dir outputs/robot_3d_multiseed --seeds 1 2 3 --samples-per-scene 8
python modules/trajectory_residual_demo.py --output-dir outputs/trajectory_residual_demo
```

TUM RGB-D reliability workflow:

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

Risk distillation and route-policy run:

```bash
python modules/run_vppv_perception_monitor.py

python modules/mechanism_router.py \
  --input-csv outputs/vppv_perception_monitor/risk_trace.csv \
  --output-dir outputs/mechanism_router \
  --action-budget 0.20 \
  --residual-reserve 0.20
```

Raw data, prepared subsets, checkpoints, and generated outputs are intentionally
not tracked by Git. See `data/README.md`.

## Evidence Boundary

- Current reliability labels are proxy labels, not natural task-failure labels.
- Controlled corruptions do not replace real perception failures caused by
  lighting, occlusion, deformation, smoke, blur, sensor degradation, or policy
  mistakes.
- Temporal and depth benchmarks show internal evidence, not closed-loop robot
  safety.
- Calibration results support risk ranking, not calibrated probability
  deployment.
- The VPPV-style section is a transfer case, not surgical system validation.
- Runtime states and route policies are transparent research rules, not formal
  safety guarantees.

## Next Validation Step

The strongest next experiment is to replace proxy labels with task-native
failure evidence: robot logs, SLAM tracking loss, segmentation quality,
depth-estimation error, surgical-tool state regression error, simulator
rollouts, or real action-outcome residuals. The key evaluation should ask
whether `visual_state_risk` and the route policy improve failure capture under
a fixed action or human-review budget.
