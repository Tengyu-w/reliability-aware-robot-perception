# SUN RGB-D Dataset Card

## Dataset Identity

- Dataset name: SUN RGB-D
- Official URL: https://rgbd.cs.princeton.edu/
- Raw data type: RGB-D indoor scene dataset
- Best project use: scene-understanding reliability and indoor 3D perception
- Prepared subset location: `data/prepared_depth/sun_rgbd`

## Why This Dataset

SUN RGB-D is useful for indoor scene understanding and semantic perception
stories. It is less SLAM-specific than TUM RGB-D, but it helps show that the
geometry-reliability pipeline can scale beyond a tiny smoke demo.

## Preparation Plan

After extracting depth maps into scene/source folders:

```bash
python modules/prepare_depth_dataset.py \
  --input-dir raw/sun_rgbd_depth \
  --output-dir data/prepared_depth/sun_rgbd \
  --max-files 300 \
  --seed 42
```

Benchmark command:

```bash
python modules/run_robot_3d_pipeline.py \
  --depth-dir data/prepared_depth/sun_rgbd \
  --output-dir outputs/sun_rgbd_robot_3d \
  --depth-scale 1000 \
  --max-files 300
```

Confirm exported depth units before final reporting.

## Professor-Facing Interpretation

What this can show:

- Indoor scene geometry reliability under controlled corruptions.
- Embedding distribution shifts across scene categories or sensors.
- Method transfer from ECG/video reliability to RGB-D scene understanding.

What this does not prove:

- Navigation or manipulation success.
- Robustness under real robot motion blur or active sensing.
- Generalization across all RGB-D cameras.
