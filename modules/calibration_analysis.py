"""Calibration and selective prediction analysis for reliability scores."""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def normalize_scores(scores):
    scores = np.asarray(scores, dtype=np.float64)
    lo = np.min(scores)
    hi = np.max(scores)
    if hi - lo < 1e-12:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


def calibration_bins(scores, labels, n_bins=10):
    """Compute bin-wise predicted risk and observed risk."""
    scores = normalize_scores(scores)
    labels = np.asarray(labels, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []

    for i in range(n_bins):
        left = edges[i]
        right = edges[i + 1]
        if i == n_bins - 1:
            mask = (scores >= left) & (scores <= right)
        else:
            mask = (scores >= left) & (scores < right)
        if not np.any(mask):
            rows.append({
                "bin": i,
                "left": left,
                "right": right,
                "n": 0,
                "mean_predicted_risk": 0.0,
                "observed_risk": 0.0,
                "abs_gap": 0.0,
            })
            continue

        mean_pred = float(scores[mask].mean())
        observed = float(labels[mask].mean())
        rows.append({
            "bin": i,
            "left": left,
            "right": right,
            "n": int(mask.sum()),
            "mean_predicted_risk": mean_pred,
            "observed_risk": observed,
            "abs_gap": abs(mean_pred - observed),
        })

    return pd.DataFrame(rows)


def expected_calibration_error(bin_df, total):
    if total == 0:
        return 0.0
    return float(((bin_df["n"] / total) * bin_df["abs_gap"]).sum())


def coverage_risk(scores, labels):
    """Risk among retained samples after abstaining on high-risk cases."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    order = np.argsort(scores)
    rows = []
    for keep in range(1, len(order) + 1):
        kept = order[:keep]
        rows.append({
            "coverage": keep / len(order),
            "observed_risk": float(labels[kept].mean()),
            "threshold": float(scores[kept[-1]]),
        })
    return pd.DataFrame(rows)


def run_calibration_analysis(
    input_csv,
    output_dir,
    score_column,
    label_column,
    n_bins=10,
):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_csv)
    if score_column not in df.columns:
        raise ValueError(f"score column '{score_column}' not found")
    if label_column not in df.columns:
        raise ValueError(f"label column '{label_column}' not found")

    scores = df[score_column].astype(float).to_numpy()
    labels = df[label_column].astype(int).to_numpy()
    norm_scores = normalize_scores(scores)

    bin_df = calibration_bins(scores, labels, n_bins=n_bins)
    bin_df.to_csv(os.path.join(output_dir, "calibration_bins.csv"), index=False)

    coverage_df = coverage_risk(norm_scores, labels)
    coverage_df.to_csv(os.path.join(output_dir, "coverage_risk.csv"), index=False)

    ece = expected_calibration_error(bin_df, len(labels))
    roc_auc = float(roc_auc_score(labels, norm_scores)) if len(set(labels)) > 1 else 0.0
    ap = float(average_precision_score(labels, norm_scores)) if len(set(labels)) > 1 else 0.0

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    plt.scatter(
        bin_df["mean_predicted_risk"],
        bin_df["observed_risk"],
        s=np.maximum(bin_df["n"], 1) * 3,
        alpha=0.8,
        label="risk bins",
    )
    plt.xlabel("Mean predicted risk (normalized score)")
    plt.ylabel("Observed risk")
    plt.title("Reliability Score Calibration")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "risk_calibration.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(coverage_df["coverage"], coverage_df["observed_risk"], linewidth=2)
    plt.xlabel("Coverage")
    plt.ylabel("Observed risk among retained samples")
    plt.title("Selective Prediction Coverage-Risk")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "coverage_risk.png"), dpi=180)
    plt.close()

    metrics = {
        "task": "risk_score_calibration_analysis",
        "input_csv": input_csv,
        "score_column": score_column,
        "label_column": label_column,
        "samples": int(len(df)),
        "positive_label_count": int(labels.sum()),
        "positive_label_rate": float(labels.mean()),
        "n_bins": int(n_bins),
        "ece": ece,
        "roc_auc": roc_auc,
        "average_precision": ap,
        "risk_at_full_coverage": float(labels.mean()),
        "risk_at_80pct_coverage": float(
            coverage_df.iloc[max(0, int(0.8 * len(coverage_df)) - 1)]["observed_risk"]
        ),
        "risk_at_50pct_coverage": float(
            coverage_df.iloc[max(0, int(0.5 * len(coverage_df)) - 1)]["observed_risk"]
        ),
        "interpretation": (
            "Calibration checks whether higher reliability scores correspond to "
            "higher observed risk, while coverage-risk measures whether abstaining "
            "on high-risk samples lowers retained risk."
        ),
    }
    with open(os.path.join(output_dir, "calibration_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    report = [
        "# Reliability Score Calibration Report",
        "",
        "## Research Question",
        "",
        "Are reliability scores calibrated enough to support selective prediction or runtime monitoring?",
        "",
        "## Key Result",
        "",
        f"- Samples: {metrics['samples']}",
        f"- Positive labels: {metrics['positive_label_count']}",
        f"- Positive label rate: {metrics['positive_label_rate']:.3f}",
        f"- ROC-AUC: {metrics['roc_auc']:.3f}",
        f"- Average precision: {metrics['average_precision']:.3f}",
        f"- ECE-style gap: {metrics['ece']:.3f}",
        f"- Risk at full coverage: {metrics['risk_at_full_coverage']:.3f}",
        f"- Risk at 80% coverage: {metrics['risk_at_80pct_coverage']:.3f}",
        f"- Risk at 50% coverage: {metrics['risk_at_50pct_coverage']:.3f}",
        "",
        "## Interpretation",
        "",
        "A useful reliability score should both rank risky samples and produce lower observed risk when high-risk samples are rejected.",
        "",
        "## Limitations",
        "",
        "- Normalized scores are not probabilistic without further calibration.",
        "- Current labels are controlled corruptions, not downstream task failures.",
        "- The clean/corrupted class prior can dominate observed bin risks; report positive label rate with calibration results.",
        "- Next step: calibrate on validation data and test on a held-out sequence or dataset.",
        "",
    ]
    with open(os.path.join(output_dir, "calibration_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze reliability score calibration.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", default="outputs/calibration_analysis")
    parser.add_argument("--score-column", required=True)
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--bins", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    metrics = run_calibration_analysis(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        score_column=args.score_column,
        label_column=args.label_column,
        n_bins=args.bins,
    )
    print("Calibration analysis completed.")
    print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
    print(f"ECE-style gap: {metrics['ece']:.3f}")
    print(f"Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
