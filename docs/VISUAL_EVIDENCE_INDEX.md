# Visual Evidence Index

This page collects the public visual evidence included in the repository. The
full generated `outputs/` directory is intentionally ignored, so the figures and
tables below are the curated GitHub-facing evidence set.

## Reliability Evidence

| Evidence | File | Why it remains useful |
|---|---|---|
| TUM temporal embedding PCA | `docs/figures/tum_temporal_embedding_pca.png` | Shows depth embedding structure for clean/corrupted TUM frames |
| TUM temporal coverage-risk | `docs/figures/tum_temporal_coverage_risk.png` | Shows selective prediction behavior under local temporal scoring |
| TUM runtime monitor trace | `docs/figures/tum_runtime_monitor_trace.png` | Shows how continuous reliability scores become NORMAL, SUSPECT, and RECOVER states |
| TUM risk calibration | `docs/figures/tum_risk_calibration.png` | Separates ranking quality from probability calibration |
| PCA depth pose rotation shift | `docs/figures/pca_depth_pose_rotation_shift.png` | Shows why learned depth descriptors are more pose-aware than global descriptors |
| Trajectory residual examples | `docs/figures/trajectory_residual_examples.png` | Provides action-outcome residual examples used by the outcome link |
| Trajectory risk by type | `docs/figures/trajectory_risk_by_type.png` | Shows residual risk across drift, oscillation, jump, and stuck failures |
| Synthetic depth PCA | `docs/figures/synthetic_depth_embedding_pca.png` | Smoke evidence for geometry embedding risk under controlled corruptions |

## Snapshot Tables

| Table | File |
|---|---|
| Key results | `docs/tables/key_results.csv` |

## Academic Use

Use these figures as evidence for the reliability-monitoring method, not as
deployment validation. The TUM RGB-D, calibration, and trajectory figures show
why global clean-reference distance is insufficient, why local temporal
normalization matters, and how residual monitoring links perception reliability
to action outcomes.
