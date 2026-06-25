"""Generate visual gallery figures for the VPPV perception monitor."""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "vppv_perception_monitor"
FIG_DIR = OUT_DIR / "visual_gallery"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_data():
    outcome = load_json(OUT_DIR / "risk_outcome_correlation.json")
    feature = pd.read_csv(OUT_DIR / "feature_importance.csv")
    ablation = pd.read_csv(OUT_DIR / "signal_group_ablation.csv")
    state_counts = pd.read_csv(OUT_DIR / "risk_state_counts.csv")
    route_policy = pd.read_csv(OUT_DIR / "vppv_route_policy.csv")
    top_cases = pd.read_csv(OUT_DIR / "top_risk_cases.csv")
    trace = pd.read_csv(OUT_DIR / "risk_trace.csv")
    return outcome, feature, ablation, state_counts, route_policy, top_cases, trace


def save_architecture_diagram(path: Path):
    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.axis("off")
    boxes = [
        ("VPPV visual front end\nsegmentation mask\ndepth map\nregressed state\nphysical state", 0.05, 0.62, 0.28, 0.32, "#d9ecff"),
        ("Reliability signals\ndepth corruption\ntemporal excess\nembedding shift\ntrajectory residual\ncalibration gap", 0.38, 0.62, 0.24, 0.32, "#e8f5e9"),
        ("Distilled model\nLogistic Regression\nRandom Forest\nDecision Tree\noutput: visual_state_risk", 0.68, 0.62, 0.24, 0.32, "#fff3cd"),
        ("Runtime routes\nNORMAL: continue\nSUSPECT: re-perceive\nRECOVER: replan / backup\nHUMAN_REVIEW: surgeon review", 0.38, 0.14, 0.54, 0.28, "#f8d7da"),
    ]
    for text, x, y, w, h, color in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#243447", linewidth=1.5))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=12, linespacing=1.25)

    for start, end in [((0.33, 0.78), (0.38, 0.78)), ((0.62, 0.78), (0.68, 0.78)), ((0.80, 0.62), (0.66, 0.42)), ((0.50, 0.62), (0.50, 0.42))]:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=2, color="#243447"))

    ax.text(
        0.5,
        0.04,
        "Core idea: unreliable visual state triggers re-perception, recovery, or human review before the downstream VPPV policy is misled.",
        ha="center",
        va="center",
        fontsize=11,
        color="#333333",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_dashboard(feature: pd.DataFrame, ablation: pd.DataFrame, state_counts: pd.DataFrame, outcome: dict, path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.ravel()
    f = feature.sort_values("importance", ascending=True)
    axes[0].barh(f["feature"], f["importance"], color="#4c78a8")
    axes[0].set_title("Feature attribution")
    axes[0].set_xlabel("importance")

    a = ablation.sort_values("teacher_roc_auc", ascending=True)
    axes[1].barh(a["signal_group"], a["teacher_roc_auc"], color="#59a14f")
    axes[1].set_title("Signal-group ablation")
    axes[1].set_xlabel("teacher ROC-AUC")
    axes[1].set_xlim(0, 1.05)

    colors = ["#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]
    axes[2].bar(state_counts["monitor_state"], state_counts["count"], color=colors)
    axes[2].set_title("Runtime state counts")
    axes[2].set_ylabel("count")
    axes[2].tick_params(axis="x", rotation=20)

    corr_labels = ["trajectory\nresidual", "temporal\nexcess", "embedding\nshift", "progress\nstagnation", "coverage\nrisk"]
    corr_values = [
        outcome.get("spearman_risk_vs_trajectory_residual", 0.0),
        outcome.get("spearman_risk_vs_temporal_excess_score", 0.0),
        outcome.get("spearman_risk_vs_embedding_shift", 0.0),
        outcome.get("spearman_risk_vs_progress_stagnation_score", 0.0),
        outcome.get("spearman_risk_vs_coverage_risk_score", 0.0),
    ]
    axes[3].bar(corr_labels, corr_values, color="#b07aa1")
    axes[3].axhline(0, color="#333333", linewidth=0.8)
    axes[3].set_title("Outcome-linked correlations")
    axes[3].set_ylabel("Spearman rho")
    axes[3].set_ylim(-0.2, 0.7)

    fig.suptitle("VPPV Perception-State Monitor: What Is Visible At A Glance", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_route_policy_flow(state_counts: pd.DataFrame, route_policy: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.axis("off")
    colors = {"NORMAL": "#d9ead3", "SUSPECT": "#fff2cc", "RECOVER": "#f4cccc", "HUMAN_REVIEW": "#eadcf8"}
    x_positions = [0.04, 0.28, 0.52, 0.76]
    for idx, row in route_policy.iterrows():
        state = row["monitor_state"]
        count_row = state_counts[state_counts["monitor_state"] == state]
        count = int(count_row["count"].iloc[0]) if not count_row.empty else 0
        fraction = float(count_row["fraction"].iloc[0]) if not count_row.empty else 0.0
        x = x_positions[idx]
        ax.add_patch(plt.Rectangle((x, 0.35), 0.20, 0.34, facecolor=colors[state], edgecolor="#333333", linewidth=1.4))
        ax.text(x + 0.10, 0.60, state, ha="center", va="center", fontsize=13, fontweight="bold")
        ax.text(x + 0.10, 0.51, f"n={count} ({fraction:.1%})", ha="center", va="center", fontsize=11)
        action = row["vppv_action"].replace("re-estimate state / replan / safe backup", "re-estimate / replan\nsafe backup")
        action = action.replace("surgeon confirmation / takeover", "surgeon confirmation\ntakeover")
        action = action.replace("slow down / re-perceive", "slow down\nre-perceive")
        ax.text(x + 0.10, 0.42, action, ha="center", va="center", fontsize=10, wrap=True)
        if idx < 3:
            ax.annotate("", xy=(x + 0.235, 0.52), xytext=(x + 0.205, 0.52), arrowprops=dict(arrowstyle="->", lw=1.7))
    ax.text(0.5, 0.15, "High-risk visual state is mapped to a concrete VPPV action.", ha="center", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_top_cases_heatmap(top_cases: pd.DataFrame, path: Path, top_k: int = 18):
    driver_cols = ["depth", "temporal", "embedding", "trajectory", "progress", "calibration", "coverage"]
    rows = top_cases.head(top_k).copy()
    matrix = np.zeros((len(rows), len(driver_cols)))
    for out_i, (_, row) in enumerate(rows.iterrows()):
        for rank in [1, 2, 3]:
            name = row[f"top_driver_{rank}"]
            score = row[f"top_driver_{rank}_score"]
            if name in driver_cols:
                matrix[out_i, driver_cols.index(name)] = score
    fig, ax = plt.subplots(figsize=(11, 6.2))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(driver_cols)))
    ax.set_xticklabels(driver_cols, rotation=30, ha="right")
    labels = [f"{row.monitor_state} | {row.corruption} | {row.trajectory_failure_type}" for row in rows.itertuples()]
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Top High-Risk Cases: Driver Heatmap")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("normalized driver score")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_state_strip(trace: pd.DataFrame, path: Path):
    mapping = {"NORMAL": 0, "SUSPECT": 1, "RECOVER": 2, "HUMAN_REVIEW": 3}
    colors = ["#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]
    arr = trace["risk_monitor_state"].map(mapping).to_numpy()[None, :]
    fig, ax = plt.subplots(figsize=(12, 1.9))
    cmap = mcolors.ListedColormap(colors)
    ax.imshow(arr, aspect="auto", cmap=cmap, vmin=0, vmax=3)
    ax.set_yticks([])
    ax.set_xlabel("sample index")
    ax.set_title("Runtime State Strip Across 1,800 Samples")
    handles = [plt.Line2D([0], [0], marker="s", color="w", label=state, markerfacecolor=colors[idx], markersize=10) for state, idx in mapping.items()]
    ax.legend(handles=handles, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.35), frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    outcome, feature, ablation, state_counts, route_policy, top_cases, trace = load_data()
    save_architecture_diagram(FIG_DIR / "vppv_monitor_architecture.png")
    save_dashboard(feature, ablation, state_counts, outcome, FIG_DIR / "vppv_visual_dashboard.png")
    save_route_policy_flow(state_counts, route_policy, FIG_DIR / "vppv_route_policy_flow.png")
    save_top_cases_heatmap(top_cases, FIG_DIR / "vppv_top_risk_case_heatmap.png")
    save_state_strip(trace, FIG_DIR / "vppv_state_strip.png")
    print(FIG_DIR)


if __name__ == "__main__":
    main()
