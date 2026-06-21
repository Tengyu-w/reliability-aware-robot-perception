"""Learned PCA depth descriptor baseline for TUM pose-aware analysis."""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from robot_3d_reliability import (
    associate_nearest_pose,
    extract_timestamp_from_path,
    iter_depth_files,
    load_depth_file,
    load_tum_groundtruth,
    pose_delta,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit a lightweight PCA depth descriptor and compare embedding shifts with TUM poses."
    )
    parser.add_argument("--depth-dir", required=True)
    parser.add_argument("--groundtruth", required=True)
    parser.add_argument("--output-dir", default="outputs/tum_pca_depth_descriptor")
    parser.add_argument("--depth-scale", type=float, default=5000.0)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--components", type=int, default=32)
    return parser.parse_args()


def load_ordered_depth_paths(depth_dir):
    rows = []
    for path in sorted(iter_depth_files(depth_dir)):
        timestamp = extract_timestamp_from_path(path)
        if timestamp is None:
            continue
        rows.append({"file": path, "depth_timestamp": timestamp})
    if not rows:
        raise ValueError(f"no timestamped depth files found in {depth_dir}")
    return pd.DataFrame(rows).sort_values("depth_timestamp").reset_index(drop=True)


def depth_to_vector(depth, image_size):
    """Resize depth, fill invalid values, and flatten."""
    valid = np.isfinite(depth) & (depth > 0)
    if np.any(valid):
        fill_value = float(np.median(depth[valid]))
    else:
        fill_value = 0.0
    clean = depth.copy()
    clean[~valid] = fill_value

    image = Image.fromarray(clean.astype(np.float32), mode="F")
    image = image.resize((image_size, image_size), resample=Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32)
    arr = np.clip(arr, 0.0, np.percentile(arr, 99.5))
    return arr.reshape(-1)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    metadata = load_ordered_depth_paths(args.depth_dir)
    vectors = []
    for path in metadata["file"]:
        depth = load_depth_file(path, depth_scale=args.depth_scale)
        vectors.append(depth_to_vector(depth, args.image_size))
    x = np.asarray(vectors, dtype=np.float32)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    n_components = min(args.components, x_scaled.shape[0] - 1, x_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    embeddings = pca.fit_transform(x_scaled)

    poses = load_tum_groundtruth(args.groundtruth)
    matched_poses = associate_nearest_pose(metadata["depth_timestamp"].to_numpy(), poses)

    rows = []
    for i in range(1, len(metadata)):
        emb_shift = float(np.linalg.norm(embeddings[i] - embeddings[i - 1]))
        translation_delta, rotation_delta_deg = pose_delta(
            matched_poses.iloc[i - 1],
            matched_poses.iloc[i],
        )
        rows.append({
            "prev_file": metadata.iloc[i - 1]["file"],
            "file": metadata.iloc[i]["file"],
            "prev_timestamp": float(metadata.iloc[i - 1]["depth_timestamp"]),
            "timestamp": float(metadata.iloc[i]["depth_timestamp"]),
            "dt": float(metadata.iloc[i]["depth_timestamp"] - metadata.iloc[i - 1]["depth_timestamp"]),
            "pca_embedding_shift": emb_shift,
            "translation_delta": translation_delta,
            "rotation_delta_deg": rotation_delta_deg,
            "pose_time_delta": float(matched_poses.iloc[i]["pose_time_delta"]),
        })
    shift_df = pd.DataFrame(rows)
    shift_df.to_csv(os.path.join(args.output_dir, "tum_pca_depth_descriptor_shifts.csv"), index=False)

    trans_corr = spearmanr(shift_df["translation_delta"], shift_df["pca_embedding_shift"])
    rot_corr = spearmanr(shift_df["rotation_delta_deg"], shift_df["pca_embedding_shift"])

    plt.figure(figsize=(7, 5))
    plt.scatter(shift_df["translation_delta"], shift_df["pca_embedding_shift"], s=20, alpha=0.75)
    plt.xlabel("Frame-to-frame translation delta")
    plt.ylabel("PCA depth embedding shift")
    plt.title("TUM Translation vs PCA Depth Descriptor Shift")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "pose_translation_vs_pca_depth_shift.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.scatter(shift_df["rotation_delta_deg"], shift_df["pca_embedding_shift"], s=20, alpha=0.75)
    plt.xlabel("Frame-to-frame rotation delta (deg)")
    plt.ylabel("PCA depth embedding shift")
    plt.title("TUM Rotation vs PCA Depth Descriptor Shift")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "pose_rotation_vs_pca_depth_shift.png"), dpi=180)
    plt.close()

    metrics = {
        "task": "tum_pca_depth_descriptor_pose_analysis",
        "samples": int(len(metadata)),
        "pairs": int(len(shift_df)),
        "image_size": int(args.image_size),
        "components": int(n_components),
        "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "spearman_translation_vs_embedding_shift": float(trans_corr.statistic),
        "spearman_translation_pvalue": float(trans_corr.pvalue),
        "spearman_rotation_vs_embedding_shift": float(rot_corr.statistic),
        "spearman_rotation_pvalue": float(rot_corr.pvalue),
        "mean_embedding_shift": float(shift_df["pca_embedding_shift"].mean()),
        "mean_translation_delta": float(shift_df["translation_delta"].mean()),
        "mean_rotation_delta_deg": float(shift_df["rotation_delta_deg"].mean()),
        "interpretation": (
            "A PCA descriptor is a lightweight learned baseline over depth images. "
            "It tests whether data-driven depth representations track camera motion "
            "better than hand-crafted global/grid descriptors."
        ),
    }
    with open(os.path.join(args.output_dir, "tum_pca_depth_descriptor_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    report = [
        "# TUM PCA Depth Descriptor Pose Analysis",
        "",
        "## Research Question",
        "",
        "Does a lightweight learned depth descriptor track camera motion better than hand-crafted geometry descriptors?",
        "",
        "## Key Result",
        "",
        f"- Depth frames: {metrics['samples']}",
        f"- Adjacent pairs: {metrics['pairs']}",
        f"- PCA components: {metrics['components']}",
        f"- Explained variance ratio: {metrics['explained_variance_ratio_sum']:.3f}",
        f"- Spearman translation vs embedding shift: {metrics['spearman_translation_vs_embedding_shift']:.3f}",
        f"- Spearman rotation vs embedding shift: {metrics['spearman_rotation_vs_embedding_shift']:.3f}",
        "",
        "## Interpretation",
        "",
        "This PCA baseline is not a deep model, but it is a data-driven descriptor. If it improves pose correlation, it supports moving toward learned RGB-D representations.",
        "",
        "## Limitations",
        "",
        "- PCA is fitted on the same sequence and is not a general pretrained descriptor.",
        "- Correlation with pose is not the same as SLAM failure prediction.",
        "- The next step is a pretrained visual/depth encoder or task-supervised descriptor.",
        "",
    ]
    with open(os.path.join(args.output_dir, "tum_pca_depth_descriptor_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print("TUM PCA depth descriptor analysis completed.")
    print(f"Components: {metrics['components']}")
    print(f"Explained variance: {metrics['explained_variance_ratio_sum']:.3f}")
    print(
        "Spearman translation/embedding shift: "
        f"{metrics['spearman_translation_vs_embedding_shift']:.3f}"
    )
    print(
        "Spearman rotation/embedding shift: "
        f"{metrics['spearman_rotation_vs_embedding_shift']:.3f}"
    )


if __name__ == "__main__":
    main()
