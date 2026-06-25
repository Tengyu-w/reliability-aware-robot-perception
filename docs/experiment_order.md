# Experiment Order

This document lists the intended reading and execution order for the current
VPPV-style repository. The older RGB-D, calibration, and trajectory experiments
are retained as supporting evidence, but the primary entry point is now the
VPPV perception monitor.

| Step | Purpose | Main scripts | Main evidence |
|---:|---|---|---|
| 1 | Run the VPPV visual-state risk distillation and route evaluation | `modules/run_vppv_perception_monitor.py` | `visual_state_risk`, feature importance, ablation, state counts, route policy |
| 2 | Generate GitHub-facing visual evidence | `tools/generate_vppv_visual_gallery.py` | VPPV architecture, dashboard, route flow, state strip, and risk figures |
| 3 | Read the main VPPV report | `reports/vppv_perception_reliability_monitor.md` | Research framing, results, limitations, email wording |
| 4 | Inspect public visual evidence | `docs/VISUAL_EVIDENCE_INDEX.md` | Curated figures and snapshot tables |
| 5 | Prepare or rerun the TUM RGB-D evidence layer | `modules/download_tum_rgbd_sample.py`, `modules/prepare_depth_dataset.py` | Public depth sequence and reproducible subset |
| 6 | Run RGB-D corruption and temporal reliability benchmarks | `modules/run_robot_3d_pipeline.py`, `modules/run_temporal_depth_benchmark.py` | Controlled corruption, temporal excess, and coverage-risk curves |
| 7 | Compare pose-aware depth descriptors | `modules/run_tum_pose_embedding_analysis.py`, `modules/run_tum_pca_depth_descriptor.py` | Global/grid/PCA descriptor relation to camera motion |
| 8 | Evaluate calibration and selective prediction | `modules/calibration_analysis.py` | ROC-AUC, average precision, calibration gap, retained-risk curves |
| 9 | Evaluate trajectory residual evidence | `modules/trajectory_residual_demo.py` | Planned-vs-observed residual risk and failure-type analysis |
| 10 | Optional: train the original video baseline | `modules/main.py`, `modules/train.py`, `modules/model.py` | Sequential perception embeddings and action baseline |

## Primary Reproduction

```bash
python modules/run_vppv_perception_monitor.py
python tools/generate_vppv_visual_gallery.py
```

## Research Interpretation

The strongest claim is not that the system is safe for deployment. The claim is
that visual-front-end reliability can be monitored, explained, and routed:
`NORMAL` continues policy execution, `SUSPECT` triggers re-perception,
`RECOVER` triggers replanning or backup behavior, and `HUMAN_REVIEW` requests a
surgeon or operator confirmation when the state cannot be trusted.
