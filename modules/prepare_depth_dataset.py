"""Prepare depth-map folders for 3D reliability experiments.

The script standardizes arbitrary depth-map trees into:

prepared_depth/
  scene_a/
    sample_000.npy or original_depth.png
  scene_b/
    ...
  manifest.csv

It does not download datasets. Keep raw public datasets outside the repository
and use this script to create a small, reproducible experiment subset.
"""

import argparse
import csv
import os
import random
import shutil
from pathlib import Path


SUPPORTED_EXTENSIONS = {".npy", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def iter_depth_files(input_dir):
    for root, _, files in os.walk(input_dir):
        for name in files:
            path = Path(root) / name
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                yield path


def infer_scene(path, input_dir):
    rel_parent = path.parent.relative_to(input_dir)
    if str(rel_parent) == ".":
        return "unknown_scene"
    return rel_parent.parts[0]


def prepare_depth_dataset(input_dir, output_dir, max_files=None, seed=42, copy_files=True, order="random"):
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(iter_depth_files(input_dir))
    if order == "random":
        rng = random.Random(seed)
        rng.shuffle(files)
    elif order == "sorted":
        pass
    else:
        raise ValueError("order must be 'random' or 'sorted'")
    if max_files is not None:
        files = files[:max_files]

    manifest_rows = []
    for idx, src in enumerate(files):
        scene = infer_scene(src, input_dir)
        scene_dir = output_dir / scene
        scene_dir.mkdir(parents=True, exist_ok=True)

        dst_name = f"{idx:06d}_{src.stem}{src.suffix.lower()}"
        dst = scene_dir / dst_name
        if copy_files:
            shutil.copy2(src, dst)
            experiment_path = dst
        else:
            experiment_path = src

        manifest_rows.append({
            "sample_index": idx,
            "scene": scene,
            "source_path": str(src),
            "experiment_path": str(experiment_path),
            "extension": src.suffix.lower(),
        })

    manifest_path = output_dir / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_index", "scene", "source_path", "experiment_path", "extension"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    return {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "samples": len(manifest_rows),
        "scenes": sorted({row["scene"] for row in manifest_rows}),
        "order": order,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare depth maps for reliability experiments.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--order", choices=["random", "sorted"], default="random")
    parser.add_argument(
        "--link-only",
        action="store_true",
        help="Write a manifest pointing to raw files without copying them.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    summary = prepare_depth_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_files=args.max_files,
        seed=args.seed,
        copy_files=not args.link_only,
        order=args.order,
    )
    print("Depth dataset prepared.")
    print(f"Samples: {summary['samples']}")
    print(f"Scenes: {', '.join(summary['scenes'])}")
    print(f"Manifest: {summary['manifest_path']}")


if __name__ == "__main__":
    main()
