# Visual Evidence Index

This page collects the public visual evidence included in the repository. The
full generated `outputs/` directory is intentionally ignored, so the figures and
tables below are the curated GitHub-facing evidence set.

## Main VPPV Upgrade

| Evidence | File | What it shows |
|---|---|---|
| Monitor architecture | `docs/figures/vppv_monitor_architecture.png` | VPPV visual front end -> reliability signals -> distilled risk -> runtime route |
| One-page dashboard | `docs/figures/vppv_visual_dashboard.png` | Feature attribution, signal ablation, state counts, and outcome correlations in one view |
| Feature attribution | `docs/figures/vppv_feature_importance.png` | `embedding_shift`, `trajectory_residual`, and `progress_slope` dominate the lightweight student model |
| Signal ablation | `docs/figures/vppv_signal_group_ablation.png` | Depth, temporal, embedding, trajectory, calibration, and coverage groups compared by ROC-AUC |
| Route policy | `docs/figures/vppv_route_policy_flow.png` | NORMAL/SUSPECT/RECOVER/HUMAN_REVIEW mapped to VPPV actions |
| Risk trace | `docs/figures/vppv_risk_trace.png` | `visual_state_risk` over 1,800 samples with thresholds and component signals |

## Supporting Reliability Evidence

| Evidence | File | Why it remains useful |
|---|---|---|
| TUM temporal embedding PCA | `docs/figures/tum_temporal_embedding_pca.png` | Shows depth embedding structure for clean/corrupted TUM frames |
| TUM temporal coverage-risk | `docs/figures/tum_temporal_coverage_risk.png` | Shows selective prediction behavior under local temporal scoring |
| TUM runtime monitor trace | `docs/figures/tum_runtime_monitor_trace.png` | Earlier runtime monitor evidence behind the VPPV state machine |
| TUM risk calibration | `docs/figures/tum_risk_calibration.png` | Separates ranking quality from probability calibration |
| PCA depth pose rotation shift | `docs/figures/pca_depth_pose_rotation_shift.png` | Shows why learned depth descriptors are more pose-aware than global descriptors |
| Trajectory residual examples | `docs/figures/trajectory_residual_examples.png` | Provides action-outcome residual examples used by the VPPV outcome link |
| Trajectory risk by type | `docs/figures/trajectory_risk_by_type.png` | Shows residual risk across drift, oscillation, jump, and stuck failures |
| Synthetic depth PCA | `docs/figures/synthetic_depth_embedding_pca.png` | Smoke evidence for geometry embedding risk under controlled corruptions |

## Snapshot Tables

| Table | File |
|---|---|
| VPPV feature importance | `docs/tables/vppv_feature_importance.csv` |
| VPPV signal-group ablation | `docs/tables/vppv_signal_group_ablation.csv` |
| VPPV risk-state counts | `docs/tables/vppv_risk_state_counts.csv` |
| VPPV route policy | `docs/tables/vppv_route_policy.csv` |
| VPPV outcome-linked validation | `docs/tables/vppv_risk_outcome_correlation.json` |
| Legacy/supporting key results | `docs/tables/key_results.csv` |

## Academic Use

The VPPV figures should be used as the primary application evidence. The older
TUM RGB-D, calibration, and trajectory figures are retained because they provide
comparative and methodological support: they show why global clean-reference
distance is insufficient, why local temporal normalization matters, and how
residual monitoring links perception reliability to action outcomes.
