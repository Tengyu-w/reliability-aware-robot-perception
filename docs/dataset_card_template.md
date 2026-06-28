# Depth Dataset Card

Use this template for each public or lab depth dataset subset used in the
robot 3D perception reliability experiments.

## Dataset Identity

- Dataset name:
- Official URL:
- License / terms of use:
- Download date:
- Raw data location outside this repository:
- Prepared subset location:

## Why This Dataset

- Robot perception relevance:
- Scene type:
- Sensor modality:
- Expected connection to target research use cases:

## Preparation

- Preparation command:

```bash
python modules/prepare_depth_dataset.py \
  --input-dir /path/to/raw_depth \
  --output-dir data/prepared_depth/<dataset_name> \
  --max-files 200 \
  --seed 42
```

- Depth scale:
- File formats:
- Scene labels:
- Sampling strategy:
- Excluded files or sequences:

## Reliability Benchmark

- Profiling command:

```bash
python modules/run_real_depth_analysis.py \
  --depth-dir data/prepared_depth/<dataset_name> \
  --output-dir outputs/<dataset_name>/profile
```

- Corruption benchmark command:

```bash
python modules/run_real_depth_corruption_benchmark.py \
  --depth-dir data/prepared_depth/<dataset_name> \
  --output-dir outputs/<dataset_name>/corruption_benchmark \
  --max-files 200
```

## Key Results

- Samples:
- Scenes/sequences:
- Embedding-risk ROC-AUC:
- Coverage-risk AUC:
- Strongest corruption signal:
- Weakest corruption signal:

## Limitations

- Does this dataset include downstream robot task labels?
- Are corruptions synthetic or dataset-native?
- Is the subset balanced across scenes?
- Could train/test leakage occur through adjacent frames or same sequence?
- What real robot claim remains unproven?

## Professor-Facing Interpretation

What this dataset shows:

What this dataset suggests:

What this dataset does not prove:
