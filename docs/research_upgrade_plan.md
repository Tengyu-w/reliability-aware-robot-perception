# Research Upgrade Plan

This project should be treated as a research prototype until it has stronger
multi-run evidence.

## Target Research Question

Can a CNN-LSTM video classifier learn separable action representations from
short industrial-action videos, and do embedding-space distribution signals
explain when the classifier is likely to be wrong?

## Evidence Produced By The Current Pipeline

- Classification accuracy on a held-out validation split.
- Class-wise precision, recall, and F1 in `metrics.json`.
- Confusion matrix in `metrics.json`.
- Per-video validation embeddings in `val_embeddings.csv`.
- PCA visualization in `embedding_pca.png`.
- Per-class embedding compactness, entropy, confidence, margin, and accuracy in
  `embedding_summary.csv`.

## Distribution Questions To Ask

- Do samples from the same action class form compact clusters?
- Are incorrect predictions near class boundaries in PCA space?
- Do low-margin or high-entropy samples overlap with errors?
- Which classes have high within-class embedding spread?
- Which classes are repeatedly confused with each other?

## Minimum Next Experiments

1. Run at least 3 seeds and report mean/std validation metrics.
2. Increase epochs beyond the current smoke-test default.
3. Add a fixed train/validation split file to avoid accidental cherry-picking.
4. Report class imbalance and per-class sample counts.
5. Compare the CNN-LSTM baseline against at least one simpler baseline.
6. Inspect failure videos for classes with low accuracy or high embedding spread.

## Robot 3D Perception Extension

The `robot_3d_reliability.py` module adds a robotics-facing evidence loop. It
uses depth maps, point-cloud geometry embeddings, and corruption/OOD scoring to
show that the same reliability idea can transfer from ECG and action video to
3D robot perception.

Professor-facing framing:

- For reliable 3D scene understanding: measure whether depth/point-cloud
  embeddings shift under sensor noise, occlusion, dropout, or viewpoint-like
  drift.
- For embodied navigation: abstain or route to a safer policy when the current
  depth observation is far from clean reference geometry.
- For autonomous systems safety: use coverage-risk curves to show the trade-off
  between acting on every perception output and rejecting unreliable inputs.

Current limitation: the included demo is synthetic. It is useful as a
reproducible evidence loop, but a stronger application package should replace
or supplement it with public RGB-D/SLAM/depth data.

Initial smoke result:

- Synthetic clean/corrupted depth samples: 288
- Corruption types: Gaussian depth noise, dropout, quantization, occlusion, and
  tilt-like depth drift
- Embedding-risk ROC-AUC for corruption detection: about 0.79 in the default
  demo setting
- Interpretation: scene-conditioned clean geometry references make corrupted
  depth observations measurably farther in point-cloud embedding space.

Reproducibility upgrade:

- Use `run_robot_3d_multiseed.py` to produce per-seed metrics and mean/std
  summaries.
- Use `run_real_depth_analysis.py` to profile public or lab depth maps without
  changing the core reliability code.
- Use `run_real_depth_corruption_benchmark.py` when real depth maps are
  available but explicit corruption labels are not. It treats original depth
  maps as clean references, applies controlled sensor corruptions, and reports
  embedding-risk ROC-AUC and coverage-risk curves.
- Use `run_temporal_depth_benchmark.py` for RGB-D sequences such as TUM. It
  tests whether corruption-induced embedding shift exceeds local clean-frame
  temporal variation.
- Use `run_tum_pose_embedding_analysis.py` to test whether frame-to-frame camera
  translation/rotation explains depth embedding shifts. Weak correlation is a
  useful limitation that motivates pose-normalized or learned 3D descriptors.
  The script can compare global point-cloud statistics against local depth-grid
  descriptors with `--descriptor both`.
- Use `run_tum_pca_depth_descriptor.py` as a lightweight learned descriptor
  baseline before moving to pretrained RGB-D/depth encoders.
- Treat synthetic results as a smoke test; real RGB-D/depth data should be the
  next validation layer.

Suggested public-data workflow:

1. Download a small RGB-D/depth subset such as TUM RGB-D, NYU Depth V2, KITTI
   depth completion, or another lab-relevant depth source.
2. For TUM RGB-D, optionally run `download_tum_rgbd_sample.py` to fetch and
   prepare `freiburg1_desk`.
3. Arrange depth files by scene or sequence folder.
4. Run `prepare_depth_dataset.py` to create a reproducible subset and manifest.
5. Fill out `docs/dataset_card_template.md` for data provenance and limitations.
6. Run `run_robot_3d_pipeline.py` to produce profiling and reliability metrics.
7. Link `robot_3d_report.md`, PCA, and coverage-risk plots in supervisor
   emails as evidence of method transfer to robot perception.

Best-fit supervisor narrative:

- Bing Wang / Junjie Hu / Bo Yang: reliable 3D perception, depth/SLAM adjacent
  uncertainty, and scene-understanding failure screening.
- Yuxiang Sun / Yiding Ji: perception reliability as a trigger for safer
  navigation, control review, or fallback behavior.
- Pai Zheng: 3D/depth perception reliability can support safer industrial
  human-robot collaboration around tools, workcells, and obstacles.

## Claims That Are Supported

- The project implements a reproducible baseline pipeline.
- The model can produce video-level embeddings for distribution analysis.
- The pipeline can generate validation metrics and embedding diagnostics.
- The project includes a runnable 3D perception reliability smoke demo using
  depth-to-point-cloud embeddings and corruption scoring.

## Claims That Are Not Yet Supported

- The model is robust enough for real deployment.
- The current one-epoch result proves strong action understanding.
- The embedding distribution is stable across seeds or datasets.
- The method generalizes to unseen camera setups or new environments.
- The synthetic 3D demo proves real-world robot perception robustness.
