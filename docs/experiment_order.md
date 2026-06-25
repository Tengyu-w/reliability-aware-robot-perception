# Experiment Order

This document lists the intended reading and execution order. The repository is
organized as a general reliability-aware sequential robot perception project,
with a VPPV-style surgical autonomy transfer case near the end.

| Step | Purpose | Main scripts | Main evidence |
|---:|---|---|---|
| 1 | Inspect or train the original sequence baseline | `modules/main.py`, `modules/train.py`, `modules/model.py` | CNN-LSTM perception baseline and embeddings |
| 2 | Prepare or rerun the TUM RGB-D evidence layer | `modules/download_tum_rgbd_sample.py`, `modules/prepare_depth_dataset.py` | Public depth sequence and reproducible subset |
| 3 | Run RGB-D corruption and temporal reliability benchmarks | `modules/run_robot_3d_pipeline.py`, `modules/run_temporal_depth_benchmark.py` | Controlled corruption, temporal excess, and coverage-risk curves |
| 4 | Compare pose-aware depth descriptors | `modules/run_tum_pose_embedding_analysis.py`, `modules/run_tum_pca_depth_descriptor.py` | Global/grid/PCA descriptor relation to camera motion |
| 5 | Evaluate calibration and selective prediction | `modules/calibration_analysis.py` | ROC-AUC, average precision, calibration gap, retained-risk curves |
| 6 | Evaluate trajectory residual evidence | `modules/trajectory_residual_demo.py` | Planned-vs-observed residual risk and failure-type analysis |
| 7 | Run visual-state risk distillation and route evaluation | `modules/run_vppv_perception_monitor.py` | `visual_state_risk`, feature importance, ablation, state counts, route policy |
| 8 | Generate GitHub-facing visual evidence | `tools/generate_vppv_visual_gallery.py` | Architecture, dashboard, route flow, state strip, and risk figures |
| 9 | Read the surgical transfer report | `reports/vppv_perception_reliability_monitor.md` | VPPV-style application case, results, and limitations |
| 10 | Inspect public visual evidence | `docs/VISUAL_EVIDENCE_INDEX.md` | Curated figures and snapshot tables |

## Primary Reproduction

For the compact public evidence layer:

```bash
python modules/run_vppv_perception_monitor.py
python tools/generate_vppv_visual_gallery.py
```

For the broader RGB-D reliability path, run the TUM preparation and temporal
benchmark scripts before the risk-distillation script.

## Research Interpretation

The strongest claim is not that the system is safe for deployment. The claim is
that visual-state reliability can be monitored, explained, and routed:
`NORMAL` continues policy execution, `SUSPECT` triggers re-perception,
`RECOVER` triggers replanning or backup behavior, and `HUMAN_REVIEW` requests an
operator confirmation when the state cannot be trusted.
