# NYU Depth V2 Dataset Card

## Dataset Identity

- Dataset name: NYU Depth V2
- Official URL: https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html
- Raw data type: indoor RGB-D scenes
- Best project use: indoor depth/scene-understanding reliability
- Prepared subset location: `data/prepared_depth/nyu_depth_v2`

## Why This Dataset

NYU Depth V2 is useful for showing indoor scene understanding rather than
robot trajectory estimation. It is a good fit for 3D perception, embodied AI,
and reliable scene-understanding narratives.

## Preparation Plan

After extracting depth maps into scene folders:

```bash
python modules/prepare_depth_dataset.py \
  --input-dir raw/nyu_depth_v2_depth \
  --output-dir data/prepared_depth/nyu_depth_v2 \
  --max-files 300 \
  --seed 42
```

Benchmark command:

```bash
python modules/run_robot_3d_pipeline.py \
  --depth-dir data/prepared_depth/nyu_depth_v2 \
  --output-dir outputs/nyu_depth_v2_robot_3d \
  --depth-scale 1000 \
  --max-files 300
```

Confirm the unit/scale of the exported depth maps before final reporting.

## Professor-Facing Interpretation

What this can show:

- Reliability of indoor depth geometry embeddings under corruptions.
- Scene-conditioned clean-reference scoring for embodied indoor perception.
- A bridge from uncertainty analysis to depth-based scene understanding.

What this does not prove:

- Navigation success.
- Multi-view consistency.
- Real-time robot deployment.
