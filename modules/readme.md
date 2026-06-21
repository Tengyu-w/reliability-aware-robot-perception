# Code Map

The code is organized as small entry points around a common reliability-analysis
theme. The recommended reading order is in `docs/experiment_order.md`.

## Core Video Baseline

- `config.py`: default paths and hyperparameters.
- `data_loader.py`: video loading, frame sampling, and transforms.
- `model.py`: ResNet18 + LSTM classifier.
- `train.py`: train and validation loops.
- `main.py`: command-line entry point for training and prediction.
- `embedding_analysis.py`: validation embedding extraction and diagnostics.

## 3D And RGB-D Reliability

- `robot_3d_reliability.py`: synthetic depth scenes, point-cloud geometry
  embeddings, corruptions, metrics, plots, and reports.
- `run_robot_3d_demo.py`: single synthetic 3D reliability run.
- `run_robot_3d_multiseed.py`: multi-seed synthetic reliability benchmark.
- `run_real_depth_analysis.py`: real depth-map profiling.
- `run_real_depth_corruption_benchmark.py`: controlled real-depth corruption
  benchmark.
- `prepare_depth_dataset.py`: reproducible depth subset preparation.
- `download_tum_rgbd_sample.py`: optional TUM RGB-D sample downloader/preparer.
- `run_robot_3d_pipeline.py`: profile plus corruption benchmark wrapper.

## Temporal, Pose, And Runtime Analysis

- `run_temporal_depth_benchmark.py`: temporal local-reference reliability.
- `run_tum_pose_embedding_analysis.py`: global/grid descriptor pose sensitivity.
- `run_tum_pca_depth_descriptor.py`: lightweight PCA depth descriptor baseline.
- `runtime_monitor.py`: converts reliability scores into runtime states.
- `calibration_analysis.py`: calibration and coverage-risk analysis.
- `trajectory_residual_demo.py`: planned-vs-observed action-outcome residuals.
- `build_evidence_pack.py`: compiles a compact research evidence summary.

