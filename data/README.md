# Data

This repository does not include raw datasets, prepared depth subsets, trained
checkpoints, or generated experiment outputs.

The code expects local data under paths such as:

```text
data/raw/
data/prepared_depth/
outputs/
```

These folders are intentionally ignored by Git because they may contain large
files, third-party datasets, generated artifacts, or machine-specific paths.

For the public RGB-D workflow, use the dataset cards in `docs/dataset_cards/`
and the preparation scripts in `modules/` to recreate a local subset before
running the experiments.

