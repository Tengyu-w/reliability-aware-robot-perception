# Surgical-Autonomy Transfer Case: Perception-State Reliability Monitor

This report is an application case for a VPPV-style surgical autonomy front
end. It is not the overall project title and does not claim to reproduce VPPV.
The broader project is reliability-aware sequential robot perception.

Chinese title: 面向 VPPV 手术自主系统的感知状态可靠性监控器

## Research Question

Can the existing reliability-aware robot perception project serve as a VPPV visual-front-end monitor for depth, regressed perceptual state, physical state, and downstream trajectory consistency?

## VPPV Fit

VPPV-style autonomy depends on segmentation masks, depth maps, perceptual regressors, and physical state vectors. This monitor does not replace those modules. It watches whether their visual-state evidence is stable enough for a downstream RL policy to trust, or whether the system should re-perceive, recover, replan, or request human review.

## Experiment A: Visual-State Risk Distillation

- Samples: 1800
- Split: group_shuffle_by_source_file
- Teacher high-risk threshold: 0.465
- Teacher high-risk rate: 0.250
- Selected model: random_forest
- Runtime output: `visual_state_risk`

| Model | Teacher ROC-AUC | Teacher AP | Brier | Observed-proxy ROC-AUC |
|---|---:|---:|---:|---:|
| logistic_regression | 0.986 | 0.955 | 0.039 | 0.934 |
| random_forest | 0.992 | 0.982 | 0.026 | 0.707 |
| decision_tree | 0.967 | 0.946 | 0.032 | 0.637 |

Input features used by the lightweight student model: `depth_valid_ratio`, `mean_depth`, `depth_std`, `temporal_local_distance`, `embedding_shift`, `trajectory_residual`, and `progress_slope`.

## Experiment A2: Feature Attribution

The monitor does not only output a scalar risk score. It also provides feature-level evidence, indicating whether risk is mainly driven by temporal visual shift, depth corruption, embedding displacement, calibration inconsistency, or trajectory residual.

| Feature | Importance | Signed/Raw Importance |
|---|---:|---:|
| embedding_shift | 0.279 | 0.279 |
| trajectory_residual | 0.279 | 0.279 |
| progress_slope | 0.260 | 0.260 |
| mean_depth | 0.066 | 0.066 |
| depth_valid_ratio | 0.059 | 0.059 |
| depth_std | 0.042 | 0.042 |
| temporal_local_distance | 0.016 | 0.016 |

High-risk samples are exported with one-sentence explanations in `top_risk_cases.csv`, including whether each case is dominated by local temporal excess, trajectory residual, depth evidence, calibration risk, or embedding shift.

## Experiment A3: Signal-Group Ablation

This ablation asks which family of signals is most useful for VPPV front-end risk. It is a diagnostic comparison, not a deployment model choice: some groups include teacher-side audit signals that are heavier than the lightweight runtime input.

| Signal group | Features | Teacher ROC-AUC | Teacher AP | Observed-proxy ROC-AUC |
|---|---:|---:|---:|---:|
| all_signals | 12 | 0.997 | 0.992 | 0.625 |
| all_except_temporal | 10 | 0.997 | 0.992 | 0.662 |
| all_student_features | 7 | 0.992 | 0.982 | 0.707 |
| trajectory_only | 3 | 0.859 | 0.779 | 0.855 |
| temporal_only | 2 | 0.765 | 0.621 | 0.917 |
| calibration_only | 1 | 0.761 | 0.581 | 0.916 |
| all_except_trajectory | 9 | 0.759 | 0.625 | 0.809 |
| embedding_only | 1 | 0.727 | 0.562 | 0.910 |
| coverage_only | 1 | 0.720 | 0.594 | 0.917 |
| depth_only | 4 | 0.698 | 0.525 | 0.635 |

## Experiment B: Normal Camera Motion vs Visual Error

- Global/scene-conditioned clean-reference ROC-AUC: 0.483
- Local temporal-excess ROC-AUC: 1.000
- Pose-aware grid rotation correlation: 0.275

Interpretation: VPPV visual reliability should not be defined only as distance from a global clean reference. In moving laparoscopic scenes, both the camera and tissue can move normally. The useful question is whether the current change exceeds normal variation inside a local temporal window.

## Experiment C: Runtime State Machine

| State | Count | Fraction | Candidate VPPV action |
|---|---:|---:|---|
| NORMAL | 1350 | 0.750 | continue_autonomy |
| SUSPECT | 433 | 0.241 | slow_down_or_reobserve |
| RECOVER | 17 | 0.009 | trigger_recovery_or_replan |
| HUMAN_REVIEW | 0 | 0.000 | request_human_review |

- NORMAL: visual state is stable; continue execution.
- SUSPECT: visual state is abnormal; slow down or re-observe.
- RECOVER: visual abnormality may affect execution; trigger recovery or replanning.
- HUMAN_REVIEW: state cannot be confirmed; request human review.

## Experiment C2: VPPV Route Policy

| Monitor state | 中文 | VPPV action | 中文动作 |
|---|---|---|---|
| NORMAL | 正常 | continue VPPV policy | 继续执行 VPPV policy |
| SUSPECT | 可疑 | slow down / re-perceive | 降低速度，重新运行 visual parsing / depth |
| RECOVER | 需要恢复 | re-estimate state / replan / safe backup | 重新估计状态，重新规划，或切换 recovery |
| HUMAN_REVIEW | 人工复查 | surgeon confirmation / takeover | 请求医生确认或接管 |

In this controlled experiment `HUMAN_REVIEW` is zero because the highest-risk samples are still routed to `RECOVER`. This is not a defect: before irreversible surgical actions such as grasping, clipping, or tissue retraction, the same state machine can use stricter review thresholds.

## Experiment D: VPPV Failure Mapping

| Project signal | Corresponding VPPV risk |
|---|---|
| depth_corruption | depth estimation error |
| temporal_excess | visual-state jump |
| embedding_shift | perceptual state outside the training distribution |
| trajectory_residual | observed action outcome deviates from plan |
| calibration_gap | hand-eye calibration or score calibration drift |
| progress_stagnation | task progress stalls |

## Experiment E: Outcome-Linked Validation

- Spearman risk vs trajectory residual: 0.543
- Spearman risk vs temporal excess: 0.521
- Spearman risk vs embedding shift: 0.542
- Top 10% risk captures RECOVER/HUMAN_REVIEW states: 1.000

This links the distilled risk to downstream monitor states and residual signals, so the score is not only fitting a teacher label. It is also decision-relevant for VPPV-style autonomy.

## Evidence Level

Confirmed: the repository now produces a VPPV-facing risk distillation run, feature attribution, route evaluation, state-count tables, and a risk trace from existing RGB-D temporal reliability and trajectory residual evidence.

Suggested: the same monitor can act as a visual-state reliability score, state-jump detector, re-perception trigger, recovery trigger, and human-review trigger for a VPPV-style pipeline.

Not yet proven: this is not validated on paired surgical robot logs, segmentation masks, VPPV policy rollouts, or real clinical deployment data. The trajectory residual signal is aligned as a deterministic proxy rather than a paired surgical execution trace.

## Email Contribution Wording

This upgrade reframes the project as a front-end reliability monitor for VPPV-style surgical autonomy. Since VPPV relies on segmentation, depth, regressed perceptual state, and physical state, unreliable visual states can mislead the downstream RL policy. The monitor provides a `visual_state_risk` score, explains which reliability signals drive that score, detects abnormal state jumps relative to a local temporal window, and routes risky states to re-perception, recovery/replanning, or human review.

## Output Files

- `outputs/vppv_perception_monitor/risk_distillation_metrics.json`
- `outputs/vppv_perception_monitor/risk_state_counts.csv`
- `outputs/vppv_perception_monitor/risk_trace.png`
- `outputs/vppv_perception_monitor/feature_importance.csv`
- `outputs/vppv_perception_monitor/feature_importance.png`
- `outputs/vppv_perception_monitor/top_risk_cases.csv`
- `outputs/vppv_perception_monitor/signal_group_ablation.csv`
- `outputs/vppv_perception_monitor/signal_group_ablation.png`
- `outputs/vppv_perception_monitor/vppv_route_policy.csv`
- `outputs/vppv_perception_monitor/risk_outcome_correlation.json`
- `reports/vppv_perception_reliability_monitor.md`
