"""Embedding and distribution diagnostics for video classification."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def collect_embeddings(model, loader, device, idx_to_class):
    """Run a model over a loader and collect logits, embeddings, and labels."""
    model.eval()
    rows = []
    embeddings = []
    labels = []
    predictions = []
    confidences = []
    entropies = []
    margins = []

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)
            logits, embedding = model(x, return_embedding=True)
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            sorted_probs, _ = probs.sort(dim=1, descending=True)
            entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)
            margin = sorted_probs[:, 0] - sorted_probs[:, 1]

            embeddings.append(embedding.cpu().numpy())
            labels.extend(y.cpu().tolist())
            predictions.extend(pred.cpu().tolist())
            confidences.extend(conf.cpu().tolist())
            entropies.extend(entropy.cpu().tolist())
            margins.extend(margin.cpu().tolist())

            start = len(rows)
            for i in range(x.size(0)):
                true_idx = int(y[i].cpu().item())
                pred_idx = int(pred[i].cpu().item())
                rows.append({
                    "sample_id": start + i,
                    "batch_id": batch_idx,
                    "true_index": true_idx,
                    "true_class": idx_to_class.get(true_idx, str(true_idx)),
                    "predicted_index": pred_idx,
                    "predicted_class": idx_to_class.get(pred_idx, str(pred_idx)),
                    "confidence": float(conf[i].cpu().item()),
                    "entropy": float(entropy[i].cpu().item()),
                    "margin": float(margin[i].cpu().item()),
                    "correct": bool(pred_idx == true_idx),
                })

    if embeddings:
        embedding_matrix = np.concatenate(embeddings, axis=0)
    else:
        embedding_matrix = np.empty((0, 0))

    meta_df = pd.DataFrame(rows)
    return embedding_matrix, meta_df, labels, predictions, confidences, entropies, margins


def save_embedding_table(embedding_matrix, meta_df, output_path):
    """Save metadata and embedding dimensions to one CSV."""
    if embedding_matrix.size == 0:
        meta_df.to_csv(output_path, index=False)
        return meta_df

    emb_cols = [f"emb_{i:03d}" for i in range(embedding_matrix.shape[1])]
    emb_df = pd.DataFrame(embedding_matrix, columns=emb_cols)
    out_df = pd.concat([meta_df.reset_index(drop=True), emb_df], axis=1)
    out_df.to_csv(output_path, index=False)
    return out_df


def summarize_embedding_distribution(embedding_matrix, meta_df, output_path):
    """Compute simple per-class distribution statistics in embedding space."""
    if embedding_matrix.size == 0 or meta_df.empty:
        summary = pd.DataFrame()
        summary.to_csv(output_path, index=False)
        return summary

    rows = []
    for class_name, group in meta_df.groupby("true_class"):
        idx = group.index.to_numpy()
        class_embeddings = embedding_matrix[idx]
        centroid = class_embeddings.mean(axis=0)
        distances = np.linalg.norm(class_embeddings - centroid, axis=1)
        rows.append({
            "class": class_name,
            "n": int(len(group)),
            "accuracy": float(group["correct"].mean()),
            "mean_confidence": float(group["confidence"].mean()),
            "mean_entropy": float(group["entropy"].mean()),
            "mean_margin": float(group["margin"].mean()),
            "mean_distance_to_centroid": float(distances.mean()),
            "std_distance_to_centroid": float(distances.std()),
        })

    summary = pd.DataFrame(rows).sort_values(["accuracy", "n"], ascending=[True, False])
    summary.to_csv(output_path, index=False)
    return summary


def plot_embedding_pca(embedding_matrix, meta_df, output_path):
    """Save a 2D PCA plot colored by true class and marked by correctness."""
    if embedding_matrix.shape[0] < 2 or embedding_matrix.shape[1] < 2:
        return False

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(embedding_matrix)

    class_codes, class_names = pd.factorize(meta_df["true_class"])
    correct = meta_df["correct"].to_numpy(dtype=bool)

    plt.figure(figsize=(9, 7))
    scatter = plt.scatter(
        coords[:, 0],
        coords[:, 1],
        c=class_codes,
        cmap="tab20",
        s=np.where(correct, 36, 72),
        marker="o",
        alpha=0.8,
        edgecolors=np.where(correct, "none", "black"),
        linewidths=np.where(correct, 0.0, 0.9),
    )
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title("Validation Embedding Distribution")
    plt.grid(True, alpha=0.25)

    handles, _ = scatter.legend_elements(num=min(len(class_names), 12))
    if handles:
        plt.legend(handles, class_names[:len(handles)], loc="best", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return True


def build_metrics_report(labels, predictions, idx_to_class):
    """Build JSON-serializable validation metrics."""
    if not labels:
        return {}

    label_ids = sorted(idx_to_class)
    target_names = [idx_to_class[i] for i in label_ids]
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=label_ids,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=label_ids).tolist(),
    }


def run_embedding_analysis(model, loader, device, idx_to_class, output_dir, split_name="val"):
    """Create embedding tables, distribution summaries, PCA plots, and metrics."""
    os.makedirs(output_dir, exist_ok=True)
    embedding_matrix, meta_df, labels, predictions, _, _, _ = collect_embeddings(
        model, loader, device, idx_to_class
    )

    save_embedding_table(
        embedding_matrix,
        meta_df,
        os.path.join(output_dir, f"{split_name}_embeddings.csv"),
    )
    summarize_embedding_distribution(
        embedding_matrix,
        meta_df,
        os.path.join(output_dir, "embedding_summary.csv"),
    )
    plot_embedding_pca(
        embedding_matrix,
        meta_df,
        os.path.join(output_dir, "embedding_pca.png"),
    )

    return build_metrics_report(labels, predictions, idx_to_class)
