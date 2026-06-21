# TUM RGB-D Dataset Card

## Dataset Identity

- Dataset name: TUM RGB-D Dataset
- Official URL: https://cvg.cit.tum.de/data/datasets/rgbd-dataset
- Raw data type: RGB-D sequences with camera trajectories
- Best project use: mobile robotics, SLAM-adjacent depth reliability, sequence-level perception drift
- Prepared subset location: `data/prepared_depth/tum_rgbd`

## Why This Dataset

TUM RGB-D is the strongest first choice for a robotics-facing evidence layer
because it is explicitly tied to RGB-D SLAM and camera motion. It is more
directly relevant to Bing Wang, Junjie Hu, Peng Yin, and Yuxiang Sun than a
generic indoor scene dataset.

## Preparation Plan

Recommended structure before preparation:

```text
raw/tum_rgbd/
  freiburg1_room/
    depth/
      *.png
  freiburg2_desk/
    depth/
      *.png
```

Preparation command:

```bash
python modules/download_tum_rgbd_sample.py \
  --sequence freiburg1_desk \
  --raw-dir data/raw/tum_rgbd \
  --prepared-dir data/prepared_depth/tum_rgbd_freiburg1_desk \
  --max-files 300
```

Manual preparation command if the dataset is already downloaded:

```bash
python modules/prepare_depth_dataset.py \
  --input-dir raw/tum_rgbd \
  --output-dir data/prepared_depth/tum_rgbd \
  --max-files 300 \
  --seed 42
```

Benchmark command:

```bash
python modules/run_robot_3d_pipeline.py \
  --depth-dir data/prepared_depth/tum_rgbd_freiburg1_desk \
  --output-dir outputs/tum_rgbd_freiburg1_desk_robot_3d \
  --depth-scale 5000 \
  --max-files 300
```

TUM RGB-D depth images are commonly converted to meters with depth scale 5000.
Confirm the selected sequence documentation before reporting final numbers.

## Professor-Facing Interpretation

What this can show:

- Whether depth geometry embeddings shift under sensor-style corruptions.
- Whether unreliable RGB-D observations can be screened before SLAM/navigation.
- Whether a reliability framework from ECG/video transfers to robot perception.

What this does not prove:

- Closed-loop robot safety.
- Full SLAM robustness.
- Generalization to all RGB-D sensors.
