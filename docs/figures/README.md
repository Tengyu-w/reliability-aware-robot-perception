# Figures

The figures in this folder are selected from local experiment outputs and copied
here so the public repository can show representative evidence without
including the full generated `outputs/` directory.

| Figure | Source experiment | What to look for |
|---|---|---|
| `vppv_monitor_architecture.png` | Reliability monitor transfer case | How visual parsing/depth/state inputs become reliability signals and runtime routes |
| `vppv_visual_dashboard.png` | Reliability monitor dashboard | One-page view of feature attribution, ablation, states, and outcome correlations |
| `vppv_feature_importance.png` | Visual-state risk distillation | Which lightweight student features drive `visual_state_risk` |
| `vppv_signal_group_ablation.png` | Signal-group ablation | Relative value of depth, temporal, embedding, trajectory, calibration, and coverage signals |
| `vppv_route_policy_flow.png` | Runtime route evaluation | Mapping from NORMAL/SUSPECT/RECOVER/HUMAN_REVIEW to autonomy actions |
| `vppv_risk_trace.png` | Risk trace | Distilled risk over samples with state thresholds and component signals |
| `synthetic_depth_embedding_pca.png` | Synthetic 3D depth reliability demo | Separation structure in point-cloud geometry embeddings |
| `tum_temporal_embedding_pca.png` | TUM RGB-D temporal benchmark | Embedding structure for clean and corrupted depth samples |
| `tum_temporal_coverage_risk.png` | TUM RGB-D temporal benchmark | Selective prediction trade-off under temporal scoring |
| `tum_runtime_monitor_trace.png` | Runtime assurance monitor | Transition of scores into NORMAL, SUSPECT, and RECOVER states |
| `tum_risk_calibration.png` | Calibration analysis | Difference between ranking quality and probability calibration |
| `pca_depth_pose_rotation_shift.png` | PCA depth descriptor analysis | Relationship between pose rotation and learned depth shift |
| `trajectory_residual_examples.png` | Trajectory residual demo | Examples of normal and failed action-outcome traces |
| `trajectory_risk_by_type.png` | Trajectory residual demo | Residual risk by failure type |

