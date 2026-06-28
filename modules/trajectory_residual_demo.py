"""Synthetic planned-vs-observed trajectory residual benchmark.

This demo supports surgical/autonomous manipulation framing: a robot has a
planned trajectory, but the observed execution may drift, oscillate, jump, or
stall. Residual scores can trigger RECOVER or HUMAN_REVIEW states.
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


FAILURE_TYPES = ["normal", "drift", "oscillation", "jump", "stuck"]


def make_plan(length=80, rng=None):
    """Create a smooth 3D planned trajectory."""
    rng = np.random.default_rng() if rng is None else rng
    t = np.linspace(0, 1, length)
    x = t
    y = 0.25 * np.sin(2 * np.pi * t)
    z = 0.2 + 0.15 * np.cos(np.pi * t)
    plan = np.stack([x, y, z], axis=1)
    plan += rng.normal(0, 0.002, size=plan.shape)
    return plan.astype(np.float32)


def observe_trajectory(plan, failure_type, severity, rng):
    """Generate an observed trajectory from a planned trajectory."""
    obs = plan.copy()
    length = len(plan)
    t = np.linspace(0, 1, length).reshape(-1, 1)
    obs += rng.normal(0, 0.005, size=obs.shape)

    if failure_type == "normal":
        return obs
    if failure_type == "drift":
        direction = rng.normal(size=(1, 3))
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        obs += t * direction * (0.04 * severity)
    elif failure_type == "oscillation":
        axis = rng.normal(size=(1, 3))
        axis = axis / (np.linalg.norm(axis) + 1e-8)
        obs += np.sin(2 * np.pi * (4 + severity) * t) * axis * (0.015 * severity)
    elif failure_type == "jump":
        idx = int(rng.integers(length // 4, 3 * length // 4))
        offset = rng.normal(size=(1, 3))
        offset = offset / (np.linalg.norm(offset) + 1e-8)
        obs[idx:] += offset * (0.05 * severity)
    elif failure_type == "stuck":
        idx = int(rng.integers(length // 3, 2 * length // 3))
        obs[idx:] = obs[idx] + rng.normal(0, 0.004, size=obs[idx:].shape)
    else:
        raise ValueError(f"unknown failure type: {failure_type}")

    return obs.astype(np.float32)


def trajectory_features(plan, obs):
    """Compute residual features between planned and observed trajectories."""
    residual = np.linalg.norm(obs - plan, axis=1)
    final_error = float(np.linalg.norm(obs[-1] - plan[-1]))
    velocity_plan = np.diff(plan, axis=0)
    velocity_obs = np.diff(obs, axis=0)
    velocity_error = np.linalg.norm(velocity_obs - velocity_plan, axis=1)
    path_length_plan = float(np.linalg.norm(velocity_plan, axis=1).sum())
    path_length_obs = float(np.linalg.norm(velocity_obs, axis=1).sum())
    progress = float(np.linalg.norm(obs[-1] - obs[0]) / (np.linalg.norm(plan[-1] - plan[0]) + 1e-8))

    return {
        "mean_residual": float(residual.mean()),
        "max_residual": float(residual.max()),
        "p95_residual": float(np.percentile(residual, 95)),
        "final_error": final_error,
        "mean_velocity_error": float(velocity_error.mean()),
        "max_velocity_error": float(velocity_error.max()),
        "path_length_ratio": path_length_obs / (path_length_plan + 1e-8),
        "progress_ratio": progress,
    }


def residual_risk_score(features):
    """Auditable residual score for trajectory reliability."""
    progress_penalty = max(0.0, 1.0 - features["progress_ratio"])
    path_penalty = abs(features["path_length_ratio"] - 1.0)
    return (
        2.0 * features["mean_residual"]
        + 1.5 * features["p95_residual"]
        + 2.0 * features["final_error"]
        + 1.0 * features["mean_velocity_error"]
        + 0.5 * path_penalty
        + 0.8 * progress_penalty
    )


def build_trajectory_dataset(n_per_type=80, length=80, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    trajectories = {}

    for failure_type in FAILURE_TYPES:
        for i in range(n_per_type):
            severity = 0 if failure_type == "normal" else int(rng.integers(1, 5))
            plan = make_plan(length=length, rng=rng)
            obs = observe_trajectory(plan, failure_type, severity, rng)
            features = trajectory_features(plan, obs)
            score = residual_risk_score(features)
            sample_id = f"{failure_type}_{i:03d}"
            rows.append({
                "sample_id": sample_id,
                "failure_type": failure_type,
                "severity": severity,
                "is_failure": failure_type != "normal",
                "residual_risk_score": score,
                **features,
            })
            trajectories[sample_id] = {"plan": plan, "observed": obs}

    return pd.DataFrame(rows), trajectories


def plot_example_trajectories(df, trajectories, output_dir):
    examples = []
    for failure_type in FAILURE_TYPES:
        row = df[df["failure_type"] == failure_type].sort_values("residual_risk_score").tail(1).iloc[0]
        examples.append(row["sample_id"])

    fig = plt.figure(figsize=(14, 8))
    for idx, sample_id in enumerate(examples, start=1):
        ax = fig.add_subplot(2, 3, idx, projection="3d")
        plan = trajectories[sample_id]["plan"]
        obs = trajectories[sample_id]["observed"]
        ax.plot(plan[:, 0], plan[:, 1], plan[:, 2], label="planned", linewidth=2)
        ax.plot(obs[:, 0], obs[:, 1], obs[:, 2], label="observed", linewidth=2)
        ax.set_title(sample_id)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "trajectory_examples.png"), dpi=180)
    plt.close()


def run_trajectory_residual_demo(output_dir, n_per_type=80, length=80, seed=42):
    os.makedirs(output_dir, exist_ok=True)
    df, trajectories = build_trajectory_dataset(n_per_type=n_per_type, length=length, seed=seed)
    df.to_csv(os.path.join(output_dir, "trajectory_residuals.csv"), index=False)

    labels = df["is_failure"].astype(int).to_numpy()
    scores = df["residual_risk_score"].astype(float).to_numpy()
    roc_auc = float(roc_auc_score(labels, scores))
    ap = float(average_precision_score(labels, scores))

    summary = df.groupby(["failure_type", "severity"]).agg(
        n=("sample_id", "count"),
        mean_score=("residual_risk_score", "mean"),
        mean_residual=("mean_residual", "mean"),
        mean_final_error=("final_error", "mean"),
        mean_progress_ratio=("progress_ratio", "mean"),
    ).reset_index()
    summary.to_csv(os.path.join(output_dir, "trajectory_residual_summary.csv"), index=False)

    plt.figure(figsize=(8, 5))
    df.boxplot(column="residual_risk_score", by="failure_type", grid=False)
    plt.suptitle("")
    plt.title("Trajectory Residual Risk by Failure Type")
    plt.xlabel("failure type")
    plt.ylabel("residual risk score")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "trajectory_risk_by_type.png"), dpi=180)
    plt.close()

    plot_example_trajectories(df, trajectories, output_dir)

    metrics = {
        "task": "trajectory_residual_reliability_demo",
        "samples": int(len(df)),
        "n_per_type": int(n_per_type),
        "length": int(length),
        "seed": int(seed),
        "roc_auc": roc_auc,
        "average_precision": ap,
        "failure_types": FAILURE_TYPES,
        "interpretation": (
            "Planned-vs-observed trajectory residuals can detect execution "
            "failures and provide a control-facing risk score."
        ),
    }
    with open(os.path.join(output_dir, "trajectory_residual_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    report = [
        "# Trajectory Residual Reliability Demo",
        "",
        "## Research Question",
        "",
        "Can planned-vs-observed trajectory residuals detect unreliable robot execution?",
        "",
        "## Failure Modes",
        "",
        "- normal",
        "- drift",
        "- oscillation",
        "- jump",
        "- stuck",
        "",
        "## Key Result",
        "",
        f"- Samples: {metrics['samples']}",
        f"- ROC-AUC: {metrics['roc_auc']:.3f}",
        f"- Average precision: {metrics['average_precision']:.3f}",
        "",
        "## Research Fit",
        "",
        "- Surgical autonomy: tool trajectory deviation and recovery triggers.",
        "- Runtime assurance: residual scores route to RECOVER or HUMAN_REVIEW.",
        "- Safe RL: residual scores can filter bad rollouts or shape recovery rewards.",
        "",
        "## Limitations",
        "",
        "- Synthetic trajectories are a controlled demo, not real surgical robot data.",
        "- Next step: replace synthetic trajectories with tool tracking, robot logs, or simulator rollouts.",
        "",
    ]
    with open(os.path.join(output_dir, "trajectory_residual_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Run synthetic trajectory residual reliability demo.")
    parser.add_argument("--output-dir", default="outputs/trajectory_residual_demo")
    parser.add_argument("--n-per-type", type=int, default=80)
    parser.add_argument("--length", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    metrics = run_trajectory_residual_demo(
        output_dir=args.output_dir,
        n_per_type=args.n_per_type,
        length=args.length,
        seed=args.seed,
    )
    print("Trajectory residual demo completed.")
    print(f"Samples: {metrics['samples']}")
    print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
    print(f"Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
