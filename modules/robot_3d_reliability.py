"""Lightweight 3D perception reliability diagnostics.

This module is intentionally dataset-agnostic. It can analyze real depth maps
or run a synthetic RGB-D style smoke demo so the repository contains a complete
robot-perception evidence loop without requiring hardware.
"""

import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import auc, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Rotation


def depth_to_point_cloud(depth, fx=1.0, fy=1.0, cx=None, cy=None, stride=4):
    """Convert a depth map to a simple point cloud using pinhole geometry."""
    if depth.ndim != 2:
        raise ValueError("depth must be a 2D array")

    h, w = depth.shape
    cx = (w - 1) / 2.0 if cx is None else cx
    cy = (h - 1) / 2.0 if cy is None else cy

    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    z = depth[ys, xs]
    valid = np.isfinite(z) & (z > 0)
    xs = xs[valid].astype(np.float32)
    ys = ys[valid].astype(np.float32)
    z = z[valid].astype(np.float32)

    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    return np.stack([x, y, z], axis=1)


def point_cloud_embedding(points):
    """Create a compact geometry embedding from a point cloud."""
    if points.size == 0:
        return np.zeros(18, dtype=np.float32)

    centroid = points.mean(axis=0)
    spread = points.std(axis=0)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    extent = maxs - mins

    centered = points - centroid
    cov = np.cov(centered.T) if len(points) > 3 else np.eye(3)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(np.maximum(eigvals, 0.0))[::-1]
    total = eigvals.sum() + 1e-8

    linearity = (eigvals[0] - eigvals[1]) / total
    planarity = (eigvals[1] - eigvals[2]) / total
    scattering = eigvals[2] / total
    density = len(points) / (np.prod(extent + 1e-6))
    z_stats = np.percentile(points[:, 2], [5, 25, 50, 75, 95])

    return np.concatenate([
        centroid,
        spread,
        extent,
        eigvals,
        np.array([linearity, planarity, scattering, density], dtype=np.float32),
        z_stats,
    ]).astype(np.float32)


def local_depth_grid_embedding(depth, grid_size=6):
    """Create a local grid descriptor that preserves coarse spatial layout."""
    if depth.ndim != 2:
        raise ValueError("depth must be a 2D array")

    h, w = depth.shape
    y_edges = np.linspace(0, h, grid_size + 1, dtype=int)
    x_edges = np.linspace(0, w, grid_size + 1, dtype=int)
    features = []

    for yi in range(grid_size):
        for xi in range(grid_size):
            patch = depth[y_edges[yi]:y_edges[yi + 1], x_edges[xi]:x_edges[xi + 1]]
            valid = patch[np.isfinite(patch) & (patch > 0)]
            valid_ratio = len(valid) / max(1, patch.size)
            if len(valid) == 0:
                features.extend([0.0, 0.0, 0.0, 0.0, valid_ratio])
                continue
            features.extend([
                float(valid.mean()),
                float(valid.std()),
                float(np.percentile(valid, 25)),
                float(np.percentile(valid, 75)),
                float(valid_ratio),
            ])

    return np.asarray(features, dtype=np.float32)


def depth_embedding(depth, descriptor="global"):
    """Compute a depth embedding by descriptor name."""
    if descriptor == "global":
        points = depth_to_point_cloud(depth, fx=90, fy=90, stride=3)
        return point_cloud_embedding(points)
    if descriptor == "grid":
        return local_depth_grid_embedding(depth, grid_size=6)
    raise ValueError("descriptor must be 'global' or 'grid'")


def corrupt_depth(depth, kind, severity, rng):
    """Apply simple corruptions that mimic RGB-D/mobile robot perception issues."""
    out = depth.copy()
    if kind == "gaussian_noise":
        out = out + rng.normal(0, 0.03 * severity, size=out.shape)
    elif kind == "dropout":
        mask = rng.random(out.shape) < min(0.08 * severity, 0.7)
        out[mask] = 0.0
    elif kind == "quantization":
        bins = max(4, int(64 / severity))
        out = np.round(out * bins) / bins
    elif kind == "occlusion":
        h, w = out.shape
        occ_h = max(4, int(h * 0.12 * severity))
        occ_w = max(4, int(w * 0.12 * severity))
        y0 = rng.integers(0, max(1, h - occ_h))
        x0 = rng.integers(0, max(1, w - occ_w))
        out[y0:y0 + occ_h, x0:x0 + occ_w] = 0.0
    elif kind == "tilt_shift":
        h, _ = out.shape
        ramp = np.linspace(-1, 1, h).reshape(-1, 1)
        out = out + 0.08 * severity * ramp
    else:
        raise ValueError(f"unknown corruption kind: {kind}")

    return np.clip(out, 0.0, None)


def make_synthetic_depth_scene(scene_type, size=96, rng=None):
    """Generate a small synthetic depth scene with different geometric layouts."""
    rng = np.random.default_rng() if rng is None else rng
    y, x = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
    base = 1.5 + 0.08 * rng.normal(size=(size, size))

    if scene_type == "corridor":
        depth = base + 0.7 * np.abs(x) + 0.15 * y
    elif scene_type == "tabletop":
        depth = base + 0.2 * y
        depth[(np.abs(x) < 0.45) & (np.abs(y) < 0.25)] -= 0.45
    elif scene_type == "obstacle":
        depth = base + 0.15 * x
        obstacle = (x ** 2 + y ** 2) < 0.18
        depth[obstacle] -= 0.7
    elif scene_type == "stairs":
        depth = base + 0.12 * np.floor((y + 1) * 5)
    else:
        raise ValueError(f"unknown scene type: {scene_type}")

    return np.clip(depth, 0.05, None)


def load_depth_file(path, depth_scale=1000.0):
    """Load a depth map from .npy or an image file.

    Image depths are divided by depth_scale. For many RGB-D datasets, 16-bit PNG
    depth is stored in millimeters, so the default converts millimeters to
    meters.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        depth = np.load(path).astype(np.float32)
    elif ext in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
        depth = np.asarray(Image.open(path)).astype(np.float32)
        if depth.ndim == 3:
            depth = depth[..., 0]
        depth = depth / float(depth_scale)
    else:
        raise ValueError(f"unsupported depth file: {path}")

    depth[~np.isfinite(depth)] = 0.0
    return np.clip(depth, 0.0, None)


def iter_depth_files(depth_dir):
    """Yield supported depth files from a directory tree."""
    supported = {".npy", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    for root, _, files in os.walk(depth_dir):
        for name in files:
            if os.path.splitext(name)[1].lower() in supported:
                yield os.path.join(root, name)


def load_depth_dataset(depth_dir, depth_scale=1000.0, label_from_parent=True):
    """Load real depth maps and convert them to geometry embeddings.

    If label_from_parent is true, the parent directory name becomes the scene
    label. This works with folder layouts such as depth/corridor/*.png.
    """
    rows = []
    embeddings = []
    for path in sorted(iter_depth_files(depth_dir)):
        depth = load_depth_file(path, depth_scale=depth_scale)
        points = depth_to_point_cloud(depth, fx=90, fy=90, stride=3)
        embedding = point_cloud_embedding(points)
        scene = os.path.basename(os.path.dirname(path)) if label_from_parent else "unknown"
        valid = depth > 0
        rows.append({
            "scene": scene,
            "sample_id": os.path.splitext(os.path.basename(path))[0],
            "file": path,
            "corruption": "real_input",
            "severity": -1,
            "is_corrupted": False,
            "valid_depth_ratio": float(valid.mean()),
            "depth_mean": float(depth[valid].mean()) if np.any(valid) else 0.0,
            "depth_std": float(depth[valid].std()) if np.any(valid) else 0.0,
        })
        embeddings.append(embedding)

    if not embeddings:
        raise ValueError(f"no supported depth files found in {depth_dir}")

    return np.asarray(embeddings), pd.DataFrame(rows)


def extract_timestamp_from_path(path):
    """Extract a TUM-style floating timestamp from a file path."""
    name = os.path.basename(path)
    match = re.search(r"(\d+\.\d+)", name)
    if not match:
        return None
    return float(match.group(1))


def load_tum_groundtruth(groundtruth_path):
    """Load TUM groundtruth trajectory file."""
    rows = []
    with open(groundtruth_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 8:
                continue
            timestamp, tx, ty, tz, qx, qy, qz, qw = map(float, parts)
            rows.append({
                "timestamp": timestamp,
                "tx": tx,
                "ty": ty,
                "tz": tz,
                "qx": qx,
                "qy": qy,
                "qz": qz,
                "qw": qw,
            })
    if not rows:
        raise ValueError(f"no groundtruth poses found in {groundtruth_path}")
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def associate_nearest_pose(timestamps, poses):
    """Associate each timestamp with the nearest pose row."""
    pose_times = poses["timestamp"].to_numpy()
    indices = np.searchsorted(pose_times, timestamps)
    matched = []
    for ts, idx in zip(timestamps, indices):
        candidates = []
        if idx < len(pose_times):
            candidates.append(idx)
        if idx > 0:
            candidates.append(idx - 1)
        best = min(candidates, key=lambda i: abs(pose_times[i] - ts))
        row = poses.iloc[best].to_dict()
        row["depth_timestamp"] = ts
        row["pose_time_delta"] = abs(row["timestamp"] - ts)
        matched.append(row)
    return pd.DataFrame(matched)


def pose_delta(row_a, row_b):
    """Compute translation and rotation deltas between two TUM poses."""
    ta = np.array([row_a["tx"], row_a["ty"], row_a["tz"]], dtype=np.float64)
    tb = np.array([row_b["tx"], row_b["ty"], row_b["tz"]], dtype=np.float64)
    translation_delta = float(np.linalg.norm(tb - ta))

    qa = np.array([row_a["qx"], row_a["qy"], row_a["qz"], row_a["qw"]], dtype=np.float64)
    qb = np.array([row_b["qx"], row_b["qy"], row_b["qz"], row_b["qw"]], dtype=np.float64)
    ra = Rotation.from_quat(qa)
    rb = Rotation.from_quat(qb)
    rotation_delta_deg = float((ra.inv() * rb).magnitude() * 180.0 / np.pi)
    return translation_delta, rotation_delta_deg


def build_synthetic_3d_dataset(samples_per_scene=24, seed=42):
    """Create clean and corrupted synthetic depth samples for a smoke demo."""
    rng = np.random.default_rng(seed)
    scene_types = ["corridor", "tabletop", "obstacle", "stairs"]
    corruption_types = ["clean", "gaussian_noise", "dropout", "quantization", "occlusion", "tilt_shift"]

    rows = []
    embeddings = []
    for scene in scene_types:
        for i in range(samples_per_scene):
            clean = make_synthetic_depth_scene(scene, rng=rng)
            for corruption in corruption_types:
                severity = 0 if corruption == "clean" else int(rng.integers(1, 5))
                depth = clean if corruption == "clean" else corrupt_depth(clean, corruption, severity, rng)
                points = depth_to_point_cloud(depth, fx=90, fy=90, stride=3)
                embedding = point_cloud_embedding(points)

                rows.append({
                    "scene": scene,
                    "sample_id": f"{scene}_{i:03d}_{corruption}",
                    "corruption": corruption,
                    "severity": severity,
                    "is_corrupted": corruption != "clean",
                    "valid_depth_ratio": float((depth > 0).mean()),
                    "depth_mean": float(depth[depth > 0].mean()) if np.any(depth > 0) else 0.0,
                    "depth_std": float(depth[depth > 0].std()) if np.any(depth > 0) else 0.0,
                })
                embeddings.append(embedding)

    return np.asarray(embeddings), pd.DataFrame(rows)


def fit_clean_reference(embeddings, metadata, n_neighbors=5):
    """Fit nearest-neighbor references from clean embeddings only."""
    clean_idx = metadata.index[metadata["corruption"] == "clean"].to_numpy()
    clean_embeddings = embeddings[clean_idx]
    n_neighbors = min(n_neighbors, len(clean_embeddings))
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(clean_embeddings)
    centroid = clean_embeddings.mean(axis=0)
    return {
        "clean_idx": clean_idx,
        "clean_embeddings": clean_embeddings,
        "nearest_neighbors": nn,
        "centroid": centroid,
    }


def fit_scene_clean_references(embeddings, metadata, n_neighbors=5):
    """Fit clean nearest-neighbor references for each scene type."""
    references = {}
    for scene, group in metadata[metadata["corruption"] == "clean"].groupby("scene"):
        idx = group.index.to_numpy()
        clean_embeddings = embeddings[idx]
        k = min(n_neighbors, len(clean_embeddings))
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(clean_embeddings)
        references[scene] = {
            "clean_idx": idx,
            "clean_embeddings": clean_embeddings,
            "nearest_neighbors": nn,
            "centroid": clean_embeddings.mean(axis=0),
        }
    return references


def score_reliability(embeddings, reference):
    """Compute simple embedding-distance reliability scores."""
    distances, _ = reference["nearest_neighbors"].kneighbors(embeddings)
    knn_distance = distances.mean(axis=1)
    centroid_distance = np.linalg.norm(embeddings - reference["centroid"], axis=1)
    score = 0.65 * knn_distance + 0.35 * centroid_distance
    return score, knn_distance, centroid_distance


def score_scene_conditioned_reliability(embeddings, metadata, references):
    """Score each sample against the clean reference for its own scene type."""
    scores = np.zeros(len(metadata), dtype=np.float32)
    knn_scores = np.zeros(len(metadata), dtype=np.float32)
    centroid_scores = np.zeros(len(metadata), dtype=np.float32)

    for scene, group in metadata.groupby("scene"):
        idx = group.index.to_numpy()
        score, knn, centroid = score_reliability(embeddings[idx], references[scene])
        scores[idx] = score
        knn_scores[idx] = knn
        centroid_scores[idx] = centroid

    return scores, knn_scores, centroid_scores


def score_source_paired_reliability(embeddings, metadata):
    """Score controlled corruptions against their own clean source embedding."""
    clean_lookup = {}
    for idx, row in metadata.iterrows():
        if row["corruption"] == "clean":
            clean_lookup[row["source_file"]] = embeddings[idx]

    scores = np.zeros(len(metadata), dtype=np.float32)
    for idx, row in metadata.iterrows():
        clean_embedding = clean_lookup.get(row["source_file"])
        if clean_embedding is None:
            scores[idx] = np.nan
        else:
            scores[idx] = np.linalg.norm(embeddings[idx] - clean_embedding)
    return np.nan_to_num(scores, nan=0.0)


def score_temporal_local_reliability(embeddings, metadata, window=5):
    """Score samples against clean embeddings from nearby frames in the sequence."""
    clean_rows = metadata[metadata["corruption"] == "clean"].copy()
    clean_rows = clean_rows.sort_values("source_file").reset_index()
    clean_indices = clean_rows["index"].to_numpy()
    source_to_position = {
        source_file: pos for pos, source_file in enumerate(clean_rows["source_file"].tolist())
    }

    scores = np.zeros(len(metadata), dtype=np.float32)
    for idx, row in metadata.iterrows():
        pos = source_to_position.get(row["source_file"])
        if pos is None:
            scores[idx] = np.nan
            continue
        start = max(0, pos - window)
        end = min(len(clean_indices), pos + window + 1)
        local_indices = clean_indices[start:end]
        local_embeddings = embeddings[local_indices]
        distances = np.linalg.norm(local_embeddings - embeddings[idx], axis=1)
        scores[idx] = float(distances.mean())

    return np.nan_to_num(scores, nan=0.0)


def score_temporal_excess_reliability(embeddings, metadata, window=5):
    """Score corruption magnitude relative to natural local temporal variation."""
    clean_rows = metadata[metadata["corruption"] == "clean"].copy()
    clean_rows = clean_rows.sort_values("source_file").reset_index()
    clean_indices = clean_rows["index"].to_numpy()
    source_to_position = {
        source_file: pos for pos, source_file in enumerate(clean_rows["source_file"].tolist())
    }
    source_to_clean_idx = {
        row["source_file"]: int(row["index"]) for _, row in clean_rows.iterrows()
    }

    scores = np.zeros(len(metadata), dtype=np.float32)
    for idx, row in metadata.iterrows():
        pos = source_to_position.get(row["source_file"])
        clean_idx = source_to_clean_idx.get(row["source_file"])
        if pos is None or clean_idx is None:
            scores[idx] = np.nan
            continue

        start = max(0, pos - window)
        end = min(len(clean_indices), pos + window + 1)
        local_indices = clean_indices[start:end]
        local_embeddings = embeddings[local_indices]
        clean_embedding = embeddings[clean_idx]

        corruption_shift = np.linalg.norm(embeddings[idx] - clean_embedding)
        natural_shift = np.linalg.norm(local_embeddings - clean_embedding, axis=1)
        baseline = float(np.median(natural_shift) + np.std(natural_shift) + 1e-6)
        scores[idx] = corruption_shift / baseline

    return np.nan_to_num(scores, nan=0.0)


def coverage_risk_curve(metadata, risk_score):
    """Compute risk among retained samples as high-risk cases are abstained."""
    order = np.argsort(risk_score)
    corrupted = metadata["is_corrupted"].to_numpy(dtype=bool)
    coverages = []
    risks = []
    for keep in range(1, len(order) + 1):
        kept = order[:keep]
        coverages.append(keep / len(order))
        risks.append(float(corrupted[kept].mean()))
    return np.asarray(coverages), np.asarray(risks)


def plot_embedding_distribution(embeddings, metadata, output_path, title):
    """Plot a 2D PCA projection for geometry embeddings."""
    if len(metadata) < 2:
        return False

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(embeddings)
    scene_codes, scene_names = pd.factorize(metadata["scene"])
    corrupted = metadata.get("is_corrupted", pd.Series(False, index=metadata.index)).to_numpy(dtype=bool)

    plt.figure(figsize=(9, 7))
    plt.scatter(
        coords[:, 0],
        coords[:, 1],
        c=scene_codes,
        cmap="tab10",
        s=np.where(corrupted, 58, 30),
        alpha=0.78,
        edgecolors=np.where(corrupted, "black", "none"),
        linewidths=np.where(corrupted, 0.8, 0.0),
    )
    plt.title(title)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=name,
                   markerfacecolor=plt.cm.tab10(i), markersize=8)
        for i, name in enumerate(scene_names[:10])
    ]
    if handles:
        plt.legend(handles=handles, fontsize=8, loc="best")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return True


def run_synthetic_3d_reliability_demo(output_dir, samples_per_scene=24, seed=42):
    """Run a complete synthetic robot 3D perception reliability smoke demo."""
    os.makedirs(output_dir, exist_ok=True)
    embeddings, metadata = build_synthetic_3d_dataset(samples_per_scene, seed)
    references = fit_scene_clean_references(embeddings, metadata)
    risk_score, knn_distance, centroid_distance = score_scene_conditioned_reliability(
        embeddings, metadata, references
    )

    metadata = metadata.copy()
    metadata["embedding_risk_score"] = risk_score
    metadata["knn_clean_distance"] = knn_distance
    metadata["clean_centroid_distance"] = centroid_distance

    emb_cols = [f"geom_emb_{i:02d}" for i in range(embeddings.shape[1])]
    embedding_df = pd.concat([
        metadata.reset_index(drop=True),
        pd.DataFrame(embeddings, columns=emb_cols),
    ], axis=1)
    embedding_df.to_csv(os.path.join(output_dir, "robot_3d_embeddings.csv"), index=False)

    summary = metadata.groupby(["corruption", "severity"]).agg(
        n=("sample_id", "count"),
        mean_risk_score=("embedding_risk_score", "mean"),
        mean_knn_clean_distance=("knn_clean_distance", "mean"),
        mean_valid_depth_ratio=("valid_depth_ratio", "mean"),
        mean_depth_std=("depth_std", "mean"),
    ).reset_index()
    summary.to_csv(os.path.join(output_dir, "robot_3d_reliability_summary.csv"), index=False)

    labels = metadata["is_corrupted"].to_numpy(dtype=int)
    roc_auc = float(roc_auc_score(labels, risk_score))
    coverages, risks = coverage_risk_curve(metadata, risk_score)
    selective_auc = float(auc(coverages, risks))

    plot_embedding_distribution(
        embeddings,
        metadata,
        os.path.join(output_dir, "robot_3d_embedding_pca.png"),
        "Synthetic 3D Perception Embedding Distribution",
    )

    plt.figure(figsize=(7, 5))
    plt.plot(coverages, risks, linewidth=2)
    plt.xlabel("Coverage")
    plt.ylabel("Corruption rate among retained samples")
    plt.title("Selective Perception Risk Curve")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "robot_3d_coverage_risk.png"), dpi=180)
    plt.close()

    metrics = {
        "task": "synthetic_robot_3d_perception_reliability",
        "samples": int(len(metadata)),
        "clean_samples": int((metadata["corruption"] == "clean").sum()),
        "corrupted_samples": int(metadata["is_corrupted"].sum()),
        "embedding_risk_roc_auc_for_corruption_detection": roc_auc,
        "coverage_risk_auc": selective_auc,
        "interpretation": (
            "Higher embedding distance from clean geometry references indicates "
            "potentially unreliable 3D perception input."
        ),
    }
    with open(os.path.join(output_dir, "robot_3d_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def run_multi_seed_synthetic_benchmark(output_dir, seeds, samples_per_scene=24):
    """Run the synthetic 3D reliability demo across multiple seeds."""
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for seed in seeds:
        seed_dir = os.path.join(output_dir, f"seed_{seed}")
        metrics = run_synthetic_3d_reliability_demo(
            output_dir=seed_dir,
            samples_per_scene=samples_per_scene,
            seed=seed,
        )
        rows.append({
            "seed": seed,
            "samples": metrics["samples"],
            "clean_samples": metrics["clean_samples"],
            "corrupted_samples": metrics["corrupted_samples"],
            "embedding_risk_roc_auc": metrics["embedding_risk_roc_auc_for_corruption_detection"],
            "coverage_risk_auc": metrics["coverage_risk_auc"],
        })

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(os.path.join(output_dir, "robot_3d_multiseed_results.csv"), index=False)

    summary = per_seed.agg({
        "embedding_risk_roc_auc": ["mean", "std", "min", "max"],
        "coverage_risk_auc": ["mean", "std", "min", "max"],
    })
    summary.to_csv(os.path.join(output_dir, "robot_3d_multiseed_summary.csv"))

    summary_json = {
        "task": "multi_seed_synthetic_robot_3d_perception_reliability",
        "seeds": [int(s) for s in seeds],
        "samples_per_scene": int(samples_per_scene),
        "embedding_risk_roc_auc_mean": float(per_seed["embedding_risk_roc_auc"].mean()),
        "embedding_risk_roc_auc_std": float(per_seed["embedding_risk_roc_auc"].std(ddof=1)) if len(per_seed) > 1 else 0.0,
        "coverage_risk_auc_mean": float(per_seed["coverage_risk_auc"].mean()),
        "coverage_risk_auc_std": float(per_seed["coverage_risk_auc"].std(ddof=1)) if len(per_seed) > 1 else 0.0,
    }
    with open(os.path.join(output_dir, "robot_3d_multiseed_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    return summary_json


def write_robot_3d_report(output_dir, metrics, title="Robot 3D Perception Reliability Evidence"):
    """Write a compact Markdown report for GitHub and supervisor emails."""
    report_path = os.path.join(output_dir, "robot_3d_report.md")
    is_real_depth = "real_depth" in metrics.get("task", "")
    limitation_head = (
        "The benchmark uses controlled corruptions on real depth maps, not full "
        "closed-loop robot validation."
        if is_real_depth
        else "The synthetic benchmark is a smoke test, not real-world robot validation."
    )
    lines = [
        f"# {title}",
        "",
        "## Research Question",
        "",
        "Can embedding-space distance from clean depth/point-cloud geometry identify unreliable robot 3D perception inputs?",
        "",
        "## Method",
        "",
        "- Convert depth maps to point clouds with a pinhole-style projection.",
        "- Extract compact geometry embeddings from centroid, spread, extent, covariance eigenvalues, and depth statistics.",
        "- Fit clean scene-conditioned references.",
        "- Score new observations by distance from clean geometry references.",
        "- Evaluate whether high-risk scores detect corrupted perception inputs.",
        "",
        "## Key Result",
        "",
    ]

    if "embedding_risk_roc_auc_mean" in metrics:
        lines.extend([
            f"- Seeds: {metrics['seeds']}",
            f"- Embedding-risk ROC-AUC: {metrics['embedding_risk_roc_auc_mean']:.3f} +/- {metrics['embedding_risk_roc_auc_std']:.3f}",
            f"- Coverage-risk AUC: {metrics['coverage_risk_auc_mean']:.3f} +/- {metrics['coverage_risk_auc_std']:.3f}",
        ])
    else:
        lines.extend([
            f"- Samples: {metrics.get('samples', 'n/a')}",
            f"- Embedding-risk ROC-AUC: {metrics.get('embedding_risk_roc_auc_for_corruption_detection', 0):.3f}",
            f"- Scene-conditioned ROC-AUC: {metrics.get('scene_conditioned_roc_auc_for_corruption_detection', 0):.3f}" if "scene_conditioned_roc_auc_for_corruption_detection" in metrics else "- Scene-conditioned ROC-AUC: n/a",
            f"- Coverage-risk AUC: {metrics.get('coverage_risk_auc', 0):.3f}",
        ])

    lines.extend([
        "",
        "## Research Fit",
        "",
        "- Reliable 3D scene understanding: depth observations that drift from clean geometry can be flagged before downstream planning.",
        "- Embodied navigation: high-risk perception frames can trigger abstention, slower control, or human review.",
        "- Industrial HRC: depth/point-cloud reliability can support safer monitoring around tools, workcells, and obstacles.",
        "",
        "## Limitations",
        "",
        f"- {limitation_head}",
        "- The next step is to add dataset-native labels, downstream task errors, or robot navigation outcomes.",
        "- Claims should be framed as reliability evidence and method transfer, not deployment readiness.",
        "",
    ])

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return report_path


def write_temporal_depth_report(output_dir, metrics):
    """Write a Markdown report for temporal depth reliability."""
    report_path = os.path.join(output_dir, "temporal_depth_report.md")
    lines = [
        "# Temporal Depth 3D Perception Reliability Evidence",
        "",
        "## Research Question",
        "",
        "Can local temporal clean references identify unreliable RGB-D/depth frames in a moving sequence?",
        "",
        "## Method",
        "",
        "- Load ordered depth frames from a sequence.",
        "- Generate controlled corruptions for each frame.",
        "- Convert each depth frame to point-cloud geometry embeddings.",
        "- Score each observation against clean embeddings from nearby frames.",
        "- Compare temporal local scoring with a source-paired sanity check.",
        "",
        "## Key Result",
        "",
        f"- Source depth files: {metrics['source_files']}",
        f"- Samples: {metrics['samples']}",
        f"- Temporal window: +/- {metrics['window']} frames",
        f"- Temporal excess ROC-AUC: {metrics['temporal_excess_roc_auc_for_corruption_detection']:.3f}",
        f"- Temporal local-distance ROC-AUC: {metrics['temporal_local_roc_auc_for_corruption_detection']:.3f}",
        f"- Source-paired ROC-AUC: {metrics['source_paired_roc_auc_for_corruption_detection']:.3f}",
        f"- Coverage-risk AUC: {metrics['coverage_risk_auc']:.3f}",
        "",
        "## Interpretation",
        "",
        "Temporal excess scoring asks whether the source-paired corruption shift is larger than the natural clean-frame variation in a local temporal window.",
        "",
        "## Limitations",
        "",
        "- This is still a controlled corruption benchmark, not SLAM failure prediction.",
        "- The next validation layer should use tracking quality, pose error, or downstream navigation instability as labels.",
        "- The method assumes nearby clean frames are available and temporally ordered.",
        "",
    ]
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return report_path


def analyze_real_depth_directory(depth_dir, output_dir, depth_scale=1000.0):
    """Analyze a directory of real depth maps without requiring labels."""
    os.makedirs(output_dir, exist_ok=True)
    embeddings, metadata = load_depth_dataset(depth_dir, depth_scale=depth_scale)

    emb_cols = [f"geom_emb_{i:02d}" for i in range(embeddings.shape[1])]
    embedding_df = pd.concat([
        metadata.reset_index(drop=True),
        pd.DataFrame(embeddings, columns=emb_cols),
    ], axis=1)
    embedding_df.to_csv(os.path.join(output_dir, "real_depth_embeddings.csv"), index=False)

    scene_summary = metadata.groupby("scene").agg(
        n=("sample_id", "count"),
        mean_valid_depth_ratio=("valid_depth_ratio", "mean"),
        mean_depth=("depth_mean", "mean"),
        mean_depth_std=("depth_std", "mean"),
    ).reset_index()
    scene_summary.to_csv(os.path.join(output_dir, "real_depth_scene_summary.csv"), index=False)

    plot_embedding_distribution(
        embeddings,
        metadata,
        os.path.join(output_dir, "real_depth_embedding_pca.png"),
        "Real Depth Geometry Embedding Distribution",
    )

    metrics = {
        "task": "real_depth_geometry_embedding_analysis",
        "depth_dir": depth_dir,
        "samples": int(len(metadata)),
        "scenes": sorted(metadata["scene"].unique().tolist()),
        "interpretation": (
            "This analysis profiles geometry embeddings from real depth maps. "
            "Add clean/corrupted or in-distribution/OOD labels to convert it "
            "into a supervised reliability benchmark."
        ),
    }
    with open(os.path.join(output_dir, "real_depth_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def build_real_depth_corruption_dataset(depth_dir, depth_scale=1000.0, seed=42, max_files=None):
    """Load real depth maps and create controlled corrupted variants."""
    rng = np.random.default_rng(seed)
    corruption_types = ["clean", "gaussian_noise", "dropout", "quantization", "occlusion", "tilt_shift"]
    files = list(sorted(iter_depth_files(depth_dir)))
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise ValueError(f"no supported depth files found in {depth_dir}")

    rows = []
    embeddings = []
    for path in files:
        clean_depth = load_depth_file(path, depth_scale=depth_scale)
        scene = os.path.basename(os.path.dirname(path))
        base_id = os.path.splitext(os.path.basename(path))[0]

        for corruption in corruption_types:
            severity = 0 if corruption == "clean" else int(rng.integers(1, 5))
            depth = clean_depth if corruption == "clean" else corrupt_depth(
                clean_depth, corruption, severity, rng
            )
            points = depth_to_point_cloud(depth, fx=90, fy=90, stride=3)
            embedding = point_cloud_embedding(points)
            valid = depth > 0

            rows.append({
                "scene": scene,
                "sample_id": f"{base_id}_{corruption}",
                "source_file": path,
                "corruption": corruption,
                "severity": severity,
                "is_corrupted": corruption != "clean",
                "valid_depth_ratio": float(valid.mean()),
                "depth_mean": float(depth[valid].mean()) if np.any(valid) else 0.0,
                "depth_std": float(depth[valid].std()) if np.any(valid) else 0.0,
            })
            embeddings.append(embedding)

    return np.asarray(embeddings), pd.DataFrame(rows)


def run_real_depth_corruption_benchmark(
    depth_dir,
    output_dir,
    depth_scale=1000.0,
    seed=42,
    max_files=None,
):
    """Benchmark reliability scores on real depth maps with synthetic corruptions."""
    os.makedirs(output_dir, exist_ok=True)
    embeddings, metadata = build_real_depth_corruption_dataset(
        depth_dir=depth_dir,
        depth_scale=depth_scale,
        seed=seed,
        max_files=max_files,
    )

    references = fit_scene_clean_references(embeddings, metadata)
    scene_risk_score, knn_distance, centroid_distance = score_scene_conditioned_reliability(
        embeddings, metadata, references
    )
    paired_risk_score = score_source_paired_reliability(embeddings, metadata)
    risk_score = paired_risk_score

    metadata = metadata.copy()
    metadata["embedding_risk_score"] = risk_score
    metadata["source_paired_clean_distance"] = paired_risk_score
    metadata["scene_conditioned_risk_score"] = scene_risk_score
    metadata["knn_clean_distance"] = knn_distance
    metadata["clean_centroid_distance"] = centroid_distance

    emb_cols = [f"geom_emb_{i:02d}" for i in range(embeddings.shape[1])]
    embedding_df = pd.concat([
        metadata.reset_index(drop=True),
        pd.DataFrame(embeddings, columns=emb_cols),
    ], axis=1)
    embedding_df.to_csv(os.path.join(output_dir, "real_depth_corruption_embeddings.csv"), index=False)

    summary = metadata.groupby(["corruption", "severity"]).agg(
        n=("sample_id", "count"),
        mean_risk_score=("embedding_risk_score", "mean"),
        mean_knn_clean_distance=("knn_clean_distance", "mean"),
        mean_valid_depth_ratio=("valid_depth_ratio", "mean"),
        mean_depth_std=("depth_std", "mean"),
    ).reset_index()
    summary.to_csv(os.path.join(output_dir, "real_depth_corruption_summary.csv"), index=False)

    labels = metadata["is_corrupted"].to_numpy(dtype=int)
    roc_auc = float(roc_auc_score(labels, risk_score))
    scene_conditioned_roc_auc = float(roc_auc_score(labels, scene_risk_score))
    coverages, risks = coverage_risk_curve(metadata, risk_score)
    coverage_auc = float(auc(coverages, risks))

    plot_embedding_distribution(
        embeddings,
        metadata,
        os.path.join(output_dir, "real_depth_corruption_pca.png"),
        "Real Depth Corruption Embedding Distribution",
    )

    plt.figure(figsize=(7, 5))
    plt.plot(coverages, risks, linewidth=2)
    plt.xlabel("Coverage")
    plt.ylabel("Corruption rate among retained samples")
    plt.title("Real Depth Selective Perception Risk Curve")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "real_depth_corruption_coverage_risk.png"), dpi=180)
    plt.close()

    metrics = {
        "task": "real_depth_corruption_reliability_benchmark",
        "depth_dir": depth_dir,
        "seed": int(seed),
        "source_files": int(metadata["source_file"].nunique()),
        "samples": int(len(metadata)),
        "clean_samples": int((metadata["corruption"] == "clean").sum()),
        "corrupted_samples": int(metadata["is_corrupted"].sum()),
        "embedding_risk_roc_auc_for_corruption_detection": roc_auc,
        "scene_conditioned_roc_auc_for_corruption_detection": scene_conditioned_roc_auc,
        "coverage_risk_auc": coverage_auc,
        "risk_score": "source_paired_clean_embedding_distance",
        "interpretation": (
            "Real depth maps are used as clean references; controlled corruptions "
            "test whether source-paired geometry embedding distance flags unreliable perception inputs."
        ),
    }
    with open(os.path.join(output_dir, "real_depth_corruption_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    write_robot_3d_report(
        output_dir,
        metrics,
        title="Real Depth 3D Perception Reliability Evidence",
    )
    return metrics


def run_temporal_depth_reliability_benchmark(
    depth_dir,
    output_dir,
    depth_scale=1000.0,
    seed=42,
    max_files=None,
    window=5,
):
    """Benchmark local temporal-reference reliability on ordered depth frames."""
    os.makedirs(output_dir, exist_ok=True)
    embeddings, metadata = build_real_depth_corruption_dataset(
        depth_dir=depth_dir,
        depth_scale=depth_scale,
        seed=seed,
        max_files=max_files,
    )

    source_paired_score = score_source_paired_reliability(embeddings, metadata)
    temporal_local_score = score_temporal_local_reliability(embeddings, metadata, window=window)
    temporal_score = score_temporal_excess_reliability(embeddings, metadata, window=window)
    labels = metadata["is_corrupted"].to_numpy(dtype=int)

    temporal_auc = float(roc_auc_score(labels, temporal_score))
    temporal_local_auc = float(roc_auc_score(labels, temporal_local_score))
    source_auc = float(roc_auc_score(labels, source_paired_score))
    coverages, risks = coverage_risk_curve(metadata, temporal_score)
    coverage_auc = float(auc(coverages, risks))

    metadata = metadata.copy()
    metadata["embedding_risk_score"] = temporal_score
    metadata["temporal_excess_score"] = temporal_score
    metadata["temporal_local_distance"] = temporal_local_score
    metadata["source_paired_clean_distance"] = source_paired_score

    emb_cols = [f"geom_emb_{i:02d}" for i in range(embeddings.shape[1])]
    embedding_df = pd.concat([
        metadata.reset_index(drop=True),
        pd.DataFrame(embeddings, columns=emb_cols),
    ], axis=1)
    embedding_df.to_csv(os.path.join(output_dir, "temporal_depth_embeddings.csv"), index=False)

    summary = metadata.groupby(["corruption", "severity"]).agg(
        n=("sample_id", "count"),
        mean_temporal_excess_score=("temporal_excess_score", "mean"),
        mean_temporal_local_distance=("temporal_local_distance", "mean"),
        mean_source_paired_distance=("source_paired_clean_distance", "mean"),
        mean_valid_depth_ratio=("valid_depth_ratio", "mean"),
        mean_depth_std=("depth_std", "mean"),
    ).reset_index()
    summary.to_csv(os.path.join(output_dir, "temporal_depth_reliability_summary.csv"), index=False)

    plot_embedding_distribution(
        embeddings,
        metadata,
        os.path.join(output_dir, "temporal_depth_embedding_pca.png"),
        "Temporal Depth Reliability Embedding Distribution",
    )

    plt.figure(figsize=(7, 5))
    plt.plot(coverages, risks, linewidth=2)
    plt.xlabel("Coverage")
    plt.ylabel("Corruption rate among retained samples")
    plt.title("Temporal Local-Reference Coverage-Risk Curve")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "temporal_depth_coverage_risk.png"), dpi=180)
    plt.close()

    metrics = {
        "task": "temporal_depth_reliability_benchmark",
        "depth_dir": depth_dir,
        "seed": int(seed),
        "window": int(window),
        "source_files": int(metadata["source_file"].nunique()),
        "samples": int(len(metadata)),
        "clean_samples": int((metadata["corruption"] == "clean").sum()),
        "corrupted_samples": int(metadata["is_corrupted"].sum()),
        "temporal_excess_roc_auc_for_corruption_detection": temporal_auc,
        "temporal_local_roc_auc_for_corruption_detection": temporal_local_auc,
        "source_paired_roc_auc_for_corruption_detection": source_auc,
        "coverage_risk_auc": coverage_auc,
        "risk_score": "temporal_excess_over_local_clean_variation",
        "interpretation": (
            "Each depth frame is scored by whether its source-paired embedding "
            "shift exceeds nearby clean-frame temporal variation."
        ),
    }
    with open(os.path.join(output_dir, "temporal_depth_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    write_temporal_depth_report(output_dir, metrics)
    return metrics
