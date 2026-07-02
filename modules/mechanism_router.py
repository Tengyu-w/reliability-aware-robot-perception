"""Mechanism-separated hierarchical routing for visual-state reliability.

This module adapts the VT/VF project's final lesson to the CNN-LSTM robot
perception project: good embeddings or a high aggregate risk score are not
enough. Reliability evidence should first identify the likely failure
mechanism, then route the sample to a matching action.
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MECHANISM_ACTIONS = {
    "perception_boundary": "re_perceive_or_request_state_set_review",
    "trajectory_residual": "trigger_recovery_or_replan",
    "depth_signal_quality": "refresh_depth_or_visual_parsing",
    "representation_conflict": "request_extra_observation",
    "progress_or_calibration": "slow_down_and_recheck_state",
    "automatic": "continue_autonomy",
}


def minmax(values):
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=float)
    lo = np.nanmin(arr[finite])
    hi = np.nanmax(arr[finite])
    if hi - lo < 1e-12:
        return np.zeros_like(arr, dtype=float)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def get_normalized_column(df, column, default=0.0):
    if column not in df.columns:
        return np.full(len(df), default, dtype=float)
    return minmax(df[column])


def get_numeric_column(df, column, default=0.0):
    if column not in df.columns:
        return np.full(len(df), default, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(default).to_numpy(dtype=float)


def build_mechanism_scores(df):
    """Build interpretable mechanism scores from an existing risk trace table."""
    scores = pd.DataFrame(index=df.index)

    temporal = get_normalized_column(df, "temporal_excess_score")
    temporal_local = get_normalized_column(df, "temporal_local_distance")
    embedding = get_normalized_column(df, "embedding_shift")
    coverage = get_normalized_column(df, "coverage_risk_score")
    visual_risk = get_normalized_column(df, "visual_state_risk")
    calibration = get_normalized_column(df, "calibration_gap_score")

    depth_valid = get_numeric_column(df, "depth_valid_ratio", default=1.0)
    depth_missing = np.clip(1.0 - depth_valid, 0.0, 1.0)
    depth_std = get_normalized_column(df, "depth_std")
    depth_corruption = get_normalized_column(df, "depth_corruption_score")

    trajectory = get_normalized_column(df, "trajectory_residual")
    progress_stagnation = get_normalized_column(df, "progress_stagnation_score")

    scores["perception_boundary_score"] = np.clip(
        0.32 * temporal
        + 0.24 * embedding
        + 0.18 * temporal_local
        + 0.14 * coverage
        + 0.12 * visual_risk,
        0.0,
        1.0,
    )
    scores["trajectory_residual_score"] = np.clip(
        0.72 * trajectory + 0.28 * progress_stagnation,
        0.0,
        1.0,
    )
    scores["depth_signal_quality_score"] = np.clip(
        0.42 * depth_corruption + 0.34 * depth_missing + 0.24 * depth_std,
        0.0,
        1.0,
    )
    scores["representation_conflict_score"] = np.clip(
        0.50 * embedding + 0.30 * coverage + 0.20 * visual_risk,
        0.0,
        1.0,
    )
    scores["progress_or_calibration_score"] = np.clip(
        0.50 * progress_stagnation + 0.30 * calibration + 0.20 * coverage,
        0.0,
        1.0,
    )
    scores["residual_mechanism_score"] = scores[
        [
            "trajectory_residual_score",
            "depth_signal_quality_score",
            "representation_conflict_score",
            "progress_or_calibration_score",
        ]
    ].max(axis=1)
    scores["combined_mechanism_score"] = np.maximum(
        scores["perception_boundary_score"],
        scores["residual_mechanism_score"],
    )
    return scores


def top_indices(values, budget, exclude=None):
    if budget <= 0:
        return []
    exclude = set() if exclude is None else set(exclude)
    candidates = [(idx, float(value)) for idx, value in values.items() if idx not in exclude]
    candidates.sort(key=lambda item: item[1], reverse=True)
    return [idx for idx, _ in candidates[:budget]]


def assign_residual_mechanism(row):
    options = {
        "trajectory_residual": row["trajectory_residual_score"],
        "depth_signal_quality": row["depth_signal_quality_score"],
        "representation_conflict": row["representation_conflict_score"],
        "progress_or_calibration": row["progress_or_calibration_score"],
    }
    return max(options, key=options.get)


def run_mechanism_router(
    input_csv,
    output_dir,
    action_budget=0.20,
    residual_reserve=0.20,
):
    """Run boundary-first, reserved-residual mechanism routing."""
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_csv)
    if df.empty:
        raise ValueError(f"input CSV has no rows: {input_csv}")

    scores = build_mechanism_scores(df)
    out = pd.concat([df.reset_index(drop=True), scores.reset_index(drop=True)], axis=1)

    total_budget = int(np.ceil(len(out) * action_budget))
    residual_budget = int(round(total_budget * residual_reserve))
    boundary_budget = max(total_budget - residual_budget, 0)

    boundary_idx = top_indices(out["perception_boundary_score"], boundary_budget)
    residual_idx = top_indices(
        out["residual_mechanism_score"],
        residual_budget,
        exclude=boundary_idx,
    )

    out["route_stage"] = "automatic"
    out["route_mechanism"] = "automatic"
    out["route_action"] = MECHANISM_ACTIONS["automatic"]
    out["route_selected"] = False

    out.loc[boundary_idx, "route_stage"] = "stage1_boundary_first"
    out.loc[boundary_idx, "route_mechanism"] = "perception_boundary"
    out.loc[boundary_idx, "route_action"] = MECHANISM_ACTIONS["perception_boundary"]
    out.loc[boundary_idx, "route_selected"] = True

    for idx in residual_idx:
        mechanism = assign_residual_mechanism(out.loc[idx])
        out.loc[idx, "route_stage"] = "stage2_reserved_residual"
        out.loc[idx, "route_mechanism"] = mechanism
        out.loc[idx, "route_action"] = MECHANISM_ACTIONS[mechanism]
        out.loc[idx, "route_selected"] = True

    out_csv = os.path.join(output_dir, "mechanism_route_decisions.csv")
    out.to_csv(out_csv, index=False)

    counts = (
        out.groupby(["route_stage", "route_mechanism", "route_action"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["route_stage", "count"], ascending=[True, False])
    )
    counts["fraction"] = counts["count"] / len(out)
    counts_path = os.path.join(output_dir, "mechanism_route_counts.csv")
    counts.to_csv(counts_path, index=False)

    metrics = build_metrics(out, input_csv, action_budget, residual_reserve, total_budget)
    metrics_path = os.path.join(output_dir, "mechanism_route_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plot_mechanism_routes(out, os.path.join(output_dir, "mechanism_route_scores.png"))
    write_report(out, counts, metrics, os.path.join(output_dir, "mechanism_router_report.md"))

    return metrics


def capture_rate(mask_selected, mask_target):
    target_count = int(mask_target.sum())
    if target_count == 0:
        return None
    return float((mask_selected & mask_target).sum() / target_count)


def build_metrics(out, input_csv, action_budget, residual_reserve, total_budget):
    selected = out["route_selected"].to_numpy(dtype=bool)
    metrics = {
        "task": "mechanism_separated_hierarchical_routing",
        "input_csv": input_csv,
        "samples": int(len(out)),
        "action_budget": float(action_budget),
        "reserved_residual_budget_fraction": float(residual_reserve),
        "total_selected": int(selected.sum()),
        "total_budget": int(total_budget),
        "selected_fraction": float(selected.mean()),
        "stage_counts": {
            key: int(value)
            for key, value in out["route_stage"].value_counts().to_dict().items()
        },
        "mechanism_counts": {
            key: int(value)
            for key, value in out["route_mechanism"].value_counts().to_dict().items()
        },
    }

    if "observed_failure_proxy" in out.columns:
        observed = out["observed_failure_proxy"].astype(str).str.lower().isin(["true", "1", "yes"]).to_numpy()
        metrics["observed_failure_proxy_count"] = int(observed.sum())
        metrics["observed_failure_proxy_capture"] = capture_rate(selected, observed)

    if "teacher_high_risk" in out.columns:
        teacher = out["teacher_high_risk"].astype(str).str.lower().isin(["true", "1", "yes"]).to_numpy()
        metrics["teacher_high_risk_count"] = int(teacher.sum())
        metrics["teacher_high_risk_capture"] = capture_rate(selected, teacher)

    if "visual_state_risk" in out.columns:
        visual_risk = pd.to_numeric(out["visual_state_risk"], errors="coerce").fillna(0.0)
        top10 = visual_risk >= visual_risk.quantile(0.90)
        top25 = visual_risk >= visual_risk.quantile(0.75)
        metrics["top10_visual_state_risk_capture"] = capture_rate(selected, top10.to_numpy(dtype=bool))
        metrics["top25_visual_state_risk_capture"] = capture_rate(selected, top25.to_numpy(dtype=bool))

    if "is_depth_corrupted" in out.columns:
        depth = out["is_depth_corrupted"].astype(str).str.lower().isin(["true", "1", "yes"]).to_numpy()
        metrics["depth_corruption_count"] = int(depth.sum())
        metrics["depth_corruption_capture"] = capture_rate(selected, depth)

    if "is_trajectory_failure" in out.columns:
        traj = out["is_trajectory_failure"].astype(str).str.lower().isin(["true", "1", "yes"]).to_numpy()
        metrics["trajectory_failure_count"] = int(traj.sum())
        metrics["trajectory_failure_capture"] = capture_rate(selected, traj)

    if "risk_monitor_state" in out.columns:
        recover = out["risk_monitor_state"].isin(["RECOVER", "HUMAN_REVIEW"]).to_numpy()
        metrics["recover_or_review_state_count"] = int(recover.sum())
        metrics["recover_or_review_state_capture"] = capture_rate(selected, recover)

    metrics["interpretation"] = (
        "The router uses the same reliability evidence as the scalar monitor, "
        "but turns it into mechanism-specific actions: boundary-first visual "
        "state review, then a reserved residual budget for trajectory, depth, "
        "representation, and progress/calibration mechanisms."
    )
    metrics["limitations"] = [
        "The current labels are proxy reliability labels, not task-native robot failure labels.",
        "Mechanism scores are transparent engineering rules and should be validation-tuned.",
        "This is a runtime routing layer, not proof of closed-loop robot safety.",
    ]
    return metrics


def plot_mechanism_routes(out, output_path):
    x = np.arange(len(out))
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(x, out["perception_boundary_score"], label="stage1 boundary", linewidth=1.1)
    axes[0].plot(x, out["residual_mechanism_score"], label="stage2 residual", linewidth=1.1)
    selected = out["route_selected"].to_numpy(dtype=bool)
    axes[0].scatter(
        x[selected],
        out.loc[selected, "combined_mechanism_score"],
        s=12,
        color="black",
        alpha=0.55,
        label="selected route",
    )
    axes[0].set_ylabel("score")
    axes[0].set_title("Mechanism-Separated Reliability Routing")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(ncol=3, fontsize=8)

    mechanism_cols = [
        "trajectory_residual_score",
        "depth_signal_quality_score",
        "representation_conflict_score",
        "progress_or_calibration_score",
    ]
    for col in mechanism_cols:
        axes[1].plot(x, out[col], label=col.replace("_score", ""), linewidth=0.9)
    axes[1].set_xlabel("sample index")
    axes[1].set_ylabel("residual score")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def fmt(value):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def write_report(out, counts, metrics, report_path):
    count_rows = [
        "| {route_stage} | {route_mechanism} | {route_action} | {count} | {fraction:.3f} |".format(**row)
        for row in counts.to_dict(orient="records")
    ]
    lines = [
        "# Mechanism-Separated Hierarchical Reliability Router",
        "",
        "## Research Question",
        "",
        "Can the scalar visual-state monitor be upgraded into mechanism-specific runtime routing for an industrial CNN-LSTM perception front end?",
        "",
        "## Main Idea",
        "",
        "The router keeps the original classifier and risk evidence, but changes the decision layer. Stage 1 prioritizes visual-state boundary evidence: temporal excess, embedding shift, local temporal distance, coverage risk, and distilled visual risk. Stage 2 reserves part of the action budget for residual mechanisms: trajectory residual, depth/signal quality, representation conflict, and progress/calibration inconsistency.",
        "",
        "## Budget",
        "",
        f"- Samples: {metrics['samples']}",
        f"- Action budget: {metrics['action_budget']:.3f}",
        f"- Total selected: {metrics['total_selected']} / {metrics['samples']}",
        f"- Reserved residual budget fraction: {metrics['reserved_residual_budget_fraction']:.3f}",
        "",
        "## Route Counts",
        "",
        "| Stage | Mechanism | Action | Count | Fraction |",
        "|---|---|---|---:|---:|",
        *count_rows,
        "",
        "## Proxy Capture",
        "",
        f"- Observed failure proxy capture: {fmt(metrics.get('observed_failure_proxy_capture'))}",
        f"- High-risk target capture: {fmt(metrics.get('teacher_high_risk_capture'))}",
        f"- Top 10% visual-state-risk capture: {fmt(metrics.get('top10_visual_state_risk_capture'))}",
        f"- Top 25% visual-state-risk capture: {fmt(metrics.get('top25_visual_state_risk_capture'))}",
        f"- Depth corruption capture: {fmt(metrics.get('depth_corruption_capture'))}",
        f"- Trajectory failure capture: {fmt(metrics.get('trajectory_failure_capture'))}",
        f"- RECOVER/HUMAN_REVIEW state capture: {fmt(metrics.get('recover_or_review_state_capture'))}",
        "",
        "## Interpretation",
        "",
        "This is the CNN-LSTM analogue of the VT/VF upgrade: embedding evidence is not treated as the answer by itself. It becomes one mechanism signal inside a hierarchy that can route different reliability failures to different actions.",
        "",
        "## Limitations",
        "",
        "- Current targets are proxy reliability labels, not paired robot task-failure labels.",
        "- Thresholds and budgets should be tuned on validation runs before any deployment claim.",
        "- The router is a decision-support wrapper around perception, not a certified safety controller.",
        "",
    ]
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def parse_args():
    parser = argparse.ArgumentParser(description="Run mechanism-separated visual reliability routing.")
    parser.add_argument(
        "--input-csv",
        default="outputs/visual_state_monitor/risk_trace.csv",
        help="CSV containing visual, temporal, depth, trajectory, progress, calibration, and coverage-risk columns.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/mechanism_router",
        help="Directory for mechanism routing outputs.",
    )
    parser.add_argument("--action-budget", type=float, default=0.20)
    parser.add_argument("--residual-reserve", type=float, default=0.20)
    return parser.parse_args()


def main():
    args = parse_args()
    metrics = run_mechanism_router(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        action_budget=args.action_budget,
        residual_reserve=args.residual_reserve,
    )
    print("Mechanism router completed.")
    print(f"Samples: {metrics['samples']}")
    print(f"Selected: {metrics['total_selected']} / {metrics['samples']}")
    print(f"Observed proxy capture: {fmt(metrics.get('observed_failure_proxy_capture'))}")
    print(f"Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
