# KITTI Depth Completion Dataset Card

## Dataset Identity

- Dataset name: KITTI Depth Completion
- Official URL: https://www.cvlibs.net/datasets/kitti/eval_depth.php
- Raw data type: outdoor driving depth maps / sparse LiDAR-derived depth
- Best project use: autonomous driving perception reliability
- Prepared subset location: `data/prepared_depth/kitti_depth`

## Why This Dataset

KITTI is useful when the target narrative is autonomous driving, outdoor robot
perception, or safety-critical mobile systems. It is especially relevant for
Yiding Ji, Bing Wang, Yuxiang Sun, and Lei Zhu if the project is framed around
perception reliability before planning/control.

## Preparation Plan

Recommended structure:

```text
raw/kitti_depth/
  sequence_or_drive_1/
    *.png
  sequence_or_drive_2/
    *.png
```

Preparation command:

```bash
python modules/prepare_depth_dataset.py \
  --input-dir raw/kitti_depth \
  --output-dir data/prepared_depth/kitti_depth \
  --max-files 300 \
  --seed 42
```

Benchmark command:

```bash
python modules/run_robot_3d_pipeline.py \
  --depth-dir data/prepared_depth/kitti_depth \
  --output-dir outputs/kitti_depth_robot_3d \
  --depth-scale 256 \
  --max-files 300
```

Confirm the exact scale convention for the downloaded KITTI depth files before
final reporting.

## Professor-Facing Interpretation

What this can show:

- Outdoor depth reliability under dropout, quantization, and noise.
- Whether embedding-distance risk can flag degraded driving perception inputs.
- A safety-facing bridge from uncertainty estimation to autonomous systems.

What this does not prove:

- End-to-end driving safety.
- 3D detection robustness.
- Closed-loop planning reliability.
