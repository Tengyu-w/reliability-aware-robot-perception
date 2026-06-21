# Experiment Order

This document lists the intended reading and execution order for the repository.
The scripts are kept as separate entry points so each result can be rerun or
replaced independently.

| Step | Purpose | Main scripts | Main evidence |
|---:|---|---|---|
| 1 | Train a sequential video baseline and export validation embeddings | `modules/main.py`, `modules/train.py`, `modules/model.py` | Video classifier checkpoint, validation metrics, embedding summaries |
| 2 | Analyze embedding-space reliability for sequential perception | `modules/embedding_analysis.py` | PCA plots and class-wise embedding summaries |
| 3 | Build synthetic depth/point-cloud reliability evidence | `modules/run_robot_3d_demo.py`, `modules/run_robot_3d_multiseed.py` | Multi-seed synthetic depth ROC-AUC and coverage-risk results |
| 4 | Prepare reproducible public-depth subsets | `modules/download_tum_rgbd_sample.py`, `modules/prepare_depth_dataset.py` | Dataset manifests and dataset cards |
| 5 | Run real-depth profiling and corruption benchmarks | `modules/run_robot_3d_pipeline.py`, `modules/run_real_depth_analysis.py`, `modules/run_real_depth_corruption_benchmark.py` | TUM RGB-D corruption and scene-conditioned reliability results |
| 6 | Test temporal reliability under camera motion | `modules/run_temporal_depth_benchmark.py` | Temporal excess score and coverage-risk curves |
| 7 | Compare pose-aware depth descriptors | `modules/run_tum_pose_embedding_analysis.py`, `modules/run_tum_pca_depth_descriptor.py` | Global, grid, and PCA descriptor correlation with pose change |
| 8 | Convert reliability scores into runtime states | `modules/runtime_monitor.py` | NORMAL, SUSPECT, RECOVER, HUMAN_REVIEW traces |
| 9 | Separate ranking quality from probability calibration | `modules/calibration_analysis.py` | ROC-AUC, average precision, calibration gap, coverage-risk |
| 10 | Extend reliability analysis to action outcomes | `modules/trajectory_residual_demo.py` | Planned-vs-observed residual risk and failure-type analysis |
| 11 | Build a compact evidence pack | `modules/build_evidence_pack.py` | `docs/application_evidence_pack.md` |

The current repository is a research prototype. The strongest claims are about
reproducible diagnostic workflows, not closed-loop robot safety.

