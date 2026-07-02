# Experiment Order

This document lists the intended reading and execution order. The repository is
organized as an industrial visual action recognition project upgraded with
reliability-aware robot perception and route-state monitoring.

| Step | Purpose | Main scripts | Main evidence |
|---:|---|---|---|
| 1 | Inspect or train the original sequence baseline | `modules/main.py`, `modules/train.py`, `modules/model.py` | CNN-LSTM perception baseline and embeddings |
| 2 | Prepare or rerun the TUM RGB-D evidence layer | `modules/download_tum_rgbd_sample.py`, `modules/prepare_depth_dataset.py` | Public depth sequence and reproducible subset |
| 3 | Run RGB-D corruption and temporal reliability benchmarks | `modules/run_robot_3d_pipeline.py`, `modules/run_temporal_depth_benchmark.py` | Controlled corruption, temporal excess, and coverage-risk curves |
| 4 | Compare pose-aware depth descriptors | `modules/run_tum_pose_embedding_analysis.py`, `modules/run_tum_pca_depth_descriptor.py` | Global/grid/PCA descriptor relation to camera motion |
| 5 | Evaluate calibration and selective prediction | `modules/calibration_analysis.py` | ROC-AUC, average precision, calibration gap, retained-risk curves |
| 6 | Evaluate trajectory residual evidence | `modules/trajectory_residual_demo.py` | Planned-vs-observed residual risk and failure-type analysis |
| 7 | Inspect mechanism-separated route logic | `modules/mechanism_router.py` | Boundary-first routing, reserved residual routes, and fixed-budget review logic |
| 8 | Inspect public visual evidence | `docs/VISUAL_EVIDENCE_INDEX.md` | Curated figures and snapshot tables |

## Primary Reproduction

For the RGB-D reliability path:

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
```

For the action-outcome residual path:

```bash
python modules/trajectory_residual_demo.py --output-dir outputs/trajectory_residual_demo
```

For route-policy inspection, pass a prepared risk trace that contains the
visual, temporal, depth, trajectory, progress, calibration, and coverage columns
used by `modules/mechanism_router.py`:

```bash
python modules/mechanism_router.py \
  --input-csv <risk_trace.csv> \
  --output-dir outputs/mechanism_router \
  --action-budget 0.20 \
  --residual-reserve 0.20
```

## Research Interpretation

The strongest claim is not that the system is safe for deployment. The claim is
that visual-state reliability can be monitored, explained, and routed:
`NORMAL` continues execution, `SUSPECT` triggers re-observation, `RECOVER`
triggers pause/replanning/recovery behavior, and `HUMAN_REVIEW` requests
operator confirmation when the state cannot be trusted.
