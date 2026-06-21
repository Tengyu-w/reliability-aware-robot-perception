"""Analyze relation between TUM pose motion and depth embedding shift."""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from robot_3d_reliability import (
    associate_nearest_pose,
    depth_embedding,
    extract_timestamp_from_path,
    iter_depth_files,
    load_depth_file,
    load_tum_groundtruth,
    pose_delta,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze pose-aware depth embedding shifts for TUM RGB-D."
    )
    parser.add_argument("--depth-dir", required=True)
    parser.add_argument("--groundtruth", required=True)
    parser.add_argument("--output-dir", default="outputs/tum_pose_embedding")
    parser.add_argument("--depth-scale", type=float, default=5000.0)
    parser.add_argument("--descriptor", choices=["global", "grid", "both"], default="both")
    return parser.parse_args()


def load_ordered_depth_metadata(depth_dir):
    rows = []
    for path in sorted(iter_depth_files(depth_dir)):
        timestamp = extract_timestamp_from_path(path)
        if timestamp is None:
            continue
        rows.append({"file": path, "depth_timestamp": timestamp})
    if not rows:
        raise ValueError(f"no timestamped depth files found in {depth_dir}")
    return pd.DataFrame(rows).sort_values("depth_timestamp").reset_index(drop=True)


def compute_descriptor_embeddings(metadata, depth_scale, descriptor):
    embeddings = []
    for path in metadata["file"]:
        depth = load_depth_file(path, depth_scale=depth_scale)
        embeddings.append(depth_embedding(depth, descriptor=descriptor))
    return np.asarray(embeddings)


def analyze_descriptor(metadata, matched_poses, embeddings, descriptor, output_dir):
    rows = []
    for i in range(1, len(metadata)):
        emb_shift = float(np.linalg.norm(embeddings[i] - embeddings[i - 1]))
        translation_delta, rotation_delta_deg = pose_delta(
            matched_poses.iloc[i - 1],
            matched_poses.iloc[i],
        )
        rows.append({
            "descriptor": descriptor,
            "prev_file": metadata.iloc[i - 1]["file"],
            "file": metadata.iloc[i]["file"],
            "prev_timestamp": float(metadata.iloc[i - 1]["depth_timestamp"]),
            "timestamp": float(metadata.iloc[i]["depth_timestamp"]),
            "dt": float(metadata.iloc[i]["depth_timestamp"] - metadata.iloc[i - 1]["depth_timestamp"]),
            "embedding_shift": emb_shift,
            "translation_delta": translation_delta,
            "rotation_delta_deg": rotation_delta_deg,
            "pose_time_delta": float(matched_poses.iloc[i]["pose_time_delta"]),
        })

    shift_df = pd.DataFrame(rows)
    trans_corr = spearmanr(shift_df["translation_delta"], shift_df["embedding_shift"])
    rot_corr = spearmanr(shift_df["rotation_delta_deg"], shift_df["embedding_shift"])

    plt.figure(figsize=(7, 5))
    plt.scatter(shift_df["translation_delta"], shift_df["embedding_shift"], s=20, alpha=0.75)
    plt.xlabel("Frame-to-frame translation delta")
    plt.ylabel(f"{descriptor} embedding shift")
    plt.title(f"TUM Translation vs {descriptor} Embedding Shift")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"pose_translation_vs_{descriptor}_embedding_shift.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.scatter(shift_df["rotation_delta_deg"], shift_df["embedding_shift"], s=20, alpha=0.75)
    plt.xlabel("Frame-to-frame rotation delta (deg)")
    plt.ylabel(f"{descriptor} embedding shift")
    plt.title(f"TUM Rotation vs {descriptor} Embedding Shift")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"pose_rotation_vs_{descriptor}_embedding_shift.png"), dpi=180)
    plt.close()

    return shift_df, {
        f"{descriptor}_spearman_translation_vs_embedding_shift": float(trans_corr.statistic),
        f"{descriptor}_spearman_translation_pvalue": float(trans_corr.pvalue),
        f"{descriptor}_spearman_rotation_vs_embedding_shift": float(rot_corr.statistic),
        f"{descriptor}_spearman_rotation_pvalue": float(rot_corr.pvalue),
        f"{descriptor}_mean_embedding_shift": float(shift_df["embedding_shift"].mean()),
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    metadata = load_ordered_depth_metadata(args.depth_dir)
    poses = load_tum_groundtruth(args.groundtruth)
    matched_poses = associate_nearest_pose(metadata["depth_timestamp"].to_numpy(), poses)

    descriptors = ["global", "grid"] if args.descriptor == "both" else [args.descriptor]
    all_shift_frames = []
    metrics = {
        "task": "tum_pose_embedding_shift_analysis",
        "samples": int(len(metadata)),
        "pairs": int(max(0, len(metadata) - 1)),
        "descriptors": descriptors,
        "mean_translation_delta": None,
        "mean_rotation_delta_deg": None,
        "interpretation": (
            "Pose-aware analysis checks whether normal camera motion explains "
            "depth embedding shifts. Comparing global and local-grid descriptors "
            "tests whether preserving coarse spatial layout improves pose awareness."
        ),
    }

    for descriptor in descriptors:
        embeddings = compute_descriptor_embeddings(metadata, args.depth_scale, descriptor)
        shift_df, descriptor_metrics = analyze_descriptor(
            metadata,
            matched_poses,
            embeddings,
            descriptor,
            args.output_dir,
        )
        all_shift_frames.append(shift_df)
        metrics.update(descriptor_metrics)
        if metrics["mean_translation_delta"] is None:
            metrics["mean_translation_delta"] = float(shift_df["translation_delta"].mean())
            metrics["mean_rotation_delta_deg"] = float(shift_df["rotation_delta_deg"].mean())

    combined = pd.concat(all_shift_frames, ignore_index=True)
    combined.to_csv(os.path.join(args.output_dir, "tum_pose_embedding_shifts.csv"), index=False)

    with open(os.path.join(args.output_dir, "tum_pose_embedding_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    report = [
        "# TUM Pose-Aware Depth Embedding Analysis",
        "",
        "## Research Question",
        "",
        "How much of the normal depth embedding shift is explained by camera translation and rotation?",
        "",
        "## Key Result",
        "",
        f"- Depth frames: {metrics['samples']}",
        f"- Adjacent pairs: {metrics['pairs']}",
    ]
    for descriptor in descriptors:
        report.extend([
            f"- {descriptor} Spearman translation vs embedding shift: {metrics[f'{descriptor}_spearman_translation_vs_embedding_shift']:.3f}",
            f"- {descriptor} Spearman rotation vs embedding shift: {metrics[f'{descriptor}_spearman_rotation_vs_embedding_shift']:.3f}",
        ])
    report.extend([
        "",
        "## Interpretation",
        "",
        "Weak correlation means the descriptor does not yet explain normal camera motion. If the local-grid descriptor improves over global statistics, it is a first step toward pose-aware spatial representations; if not, the next step should be learned local 3D/RGB-D descriptors.",
        "",
        "## Limitations",
        "",
        "- This analyzes adjacent-frame geometry shift, not SLAM failure labels.",
        "- The next step is to combine pose-normalized residuals with corruption or tracking-quality labels.",
        "",
    ])
    with open(os.path.join(args.output_dir, "tum_pose_embedding_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print("TUM pose-aware embedding analysis completed.")
    print(f"Pairs: {metrics['pairs']}")
    for descriptor in descriptors:
        print(
            f"{descriptor} Spearman translation/embedding shift: "
            f"{metrics[f'{descriptor}_spearman_translation_vs_embedding_shift']:.3f}"
        )
        print(
            f"{descriptor} Spearman rotation/embedding shift: "
            f"{metrics[f'{descriptor}_spearman_rotation_vs_embedding_shift']:.3f}"
        )


if __name__ == "__main__":
    main()
