"""Runtime assurance monitor for reliability scores.

The monitor turns per-frame/per-sample reliability scores into explicit system
states:

NORMAL -> SUSPECT -> RECOVER -> HUMAN_REVIEW

It is intentionally simple and auditable so it can be discussed with robotics,
safe RL, and surgical autonomy supervisors.
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import pandas as pd


STATES = ["NORMAL", "SUSPECT", "RECOVER", "HUMAN_REVIEW"]


def quantile_thresholds(scores, suspect_q=0.75, recover_q=0.9, review_q=0.97):
    """Compute thresholds from score quantiles."""
    return {
        "suspect": float(scores.quantile(suspect_q)),
        "recover": float(scores.quantile(recover_q)),
        "review": float(scores.quantile(review_q)),
        "suspect_q": suspect_q,
        "recover_q": recover_q,
        "review_q": review_q,
    }


def step_monitor(
    score,
    thresholds,
    previous_state,
    consecutive_recover,
    consecutive_review,
    recover_patience=2,
    review_patience=3,
):
    """Advance the state machine by one score."""
    if score >= thresholds["review"]:
        consecutive_review += 1
        consecutive_recover += 1
    elif score >= thresholds["recover"]:
        consecutive_recover += 1
        consecutive_review = 0
    else:
        consecutive_recover = 0
        consecutive_review = 0

    if consecutive_review >= review_patience:
        state = "HUMAN_REVIEW"
        action = "transfer_to_human_or_safe_controller"
    elif consecutive_recover >= recover_patience:
        state = "RECOVER"
        action = "trigger_recovery_or_replan"
    elif score >= thresholds["suspect"]:
        state = "SUSPECT"
        action = "slow_down_or_request_extra_observation"
    else:
        state = "NORMAL"
        action = "continue_autonomy"

    # Hysteresis: once in human review, require a clearly normal score to exit.
    if previous_state == "HUMAN_REVIEW" and score >= thresholds["suspect"]:
        state = "HUMAN_REVIEW"
        action = "remain_under_human_review"

    return state, action, consecutive_recover, consecutive_review


def run_runtime_monitor(
    input_csv,
    output_dir,
    score_column="embedding_risk_score",
    timestamp_column=None,
    suspect_q=0.75,
    recover_q=0.9,
    review_q=0.97,
    recover_patience=2,
    review_patience=3,
):
    """Run the monitor over a CSV containing reliability scores."""
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_csv)
    if score_column not in df.columns:
        raise ValueError(f"score column '{score_column}' not found in {input_csv}")

    scores = df[score_column].astype(float)
    thresholds = quantile_thresholds(scores, suspect_q, recover_q, review_q)

    state = "NORMAL"
    consecutive_recover = 0
    consecutive_review = 0
    states = []
    actions = []
    recover_counts = []
    review_counts = []

    for score in scores:
        state, action, consecutive_recover, consecutive_review = step_monitor(
            score=score,
            thresholds=thresholds,
            previous_state=state,
            consecutive_recover=consecutive_recover,
            consecutive_review=consecutive_review,
            recover_patience=recover_patience,
            review_patience=review_patience,
        )
        states.append(state)
        actions.append(action)
        recover_counts.append(consecutive_recover)
        review_counts.append(consecutive_review)

    out_df = df.copy()
    out_df["monitor_state"] = states
    out_df["monitor_action"] = actions
    out_df["consecutive_recover_risk"] = recover_counts
    out_df["consecutive_review_risk"] = review_counts
    out_df["property_no_autonomy_under_recover"] = ~(
        out_df["monitor_state"].isin(["RECOVER", "HUMAN_REVIEW"])
        & (out_df["monitor_action"] == "continue_autonomy")
    )
    out_csv = os.path.join(output_dir, "runtime_monitor_decisions.csv")
    out_df.to_csv(out_csv, index=False)

    state_counts = out_df["monitor_state"].value_counts().reindex(STATES, fill_value=0)
    action_counts = out_df["monitor_action"].value_counts()
    property_violations = int((~out_df["property_no_autonomy_under_recover"]).sum())

    plt.figure(figsize=(10, 4))
    x = out_df[timestamp_column] if timestamp_column and timestamp_column in out_df.columns else range(len(out_df))
    plt.plot(x, scores, linewidth=1.4, label=score_column)
    plt.axhline(thresholds["suspect"], color="orange", linestyle="--", label="suspect")
    plt.axhline(thresholds["recover"], color="red", linestyle="--", label="recover")
    plt.axhline(thresholds["review"], color="purple", linestyle="--", label="human review")
    plt.xlabel(timestamp_column if timestamp_column else "sample index")
    plt.ylabel("risk score")
    plt.title("Runtime Monitor Risk Trace")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "runtime_monitor_trace.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    state_counts.plot(kind="bar")
    plt.ylabel("count")
    plt.title("Runtime Monitor State Counts")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "runtime_monitor_state_counts.png"), dpi=180)
    plt.close()

    metrics = {
        "task": "runtime_assurance_monitor",
        "input_csv": input_csv,
        "score_column": score_column,
        "samples": int(len(out_df)),
        "thresholds": thresholds,
        "recover_patience": int(recover_patience),
        "review_patience": int(review_patience),
        "state_counts": {k: int(v) for k, v in state_counts.to_dict().items()},
        "action_counts": {k: int(v) for k, v in action_counts.to_dict().items()},
        "property_checks": {
            "no_continue_autonomy_under_recover_or_review": property_violations == 0,
            "violations": property_violations,
        },
        "interpretation": (
            "The monitor maps continuous reliability scores into auditable "
            "runtime states for autonomy, recovery, and human review."
        ),
    }
    metrics_path = os.path.join(output_dir, "runtime_monitor_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    report = [
        "# Runtime Assurance Monitor Report",
        "",
        "## Research Question",
        "",
        "Can reliability scores be converted into auditable autonomy states for safer robot operation?",
        "",
        "## Monitor States",
        "",
        "- NORMAL: continue autonomy.",
        "- SUSPECT: slow down or request an extra observation.",
        "- RECOVER: trigger recovery, replanning, or a safe controller.",
        "- HUMAN_REVIEW: transfer control or request human review.",
        "",
        "## Key Result",
        "",
        f"- Samples: {metrics['samples']}",
        f"- Score column: `{score_column}`",
        f"- NORMAL: {metrics['state_counts']['NORMAL']}",
        f"- SUSPECT: {metrics['state_counts']['SUSPECT']}",
        f"- RECOVER: {metrics['state_counts']['RECOVER']}",
        f"- HUMAN_REVIEW: {metrics['state_counts']['HUMAN_REVIEW']}",
        f"- Property violations: {metrics['property_checks']['violations']}",
        "",
        "## Interpretation",
        "",
        "This is not a formal proof of safety. It is a transparent runtime-assurance wrapper that turns learned reliability signals into actions that can be inspected, tuned, and later verified.",
        "",
        "## Limitations",
        "",
        "- Thresholds are quantile-based and should be calibrated on task-specific validation data.",
        "- The monitor currently uses score magnitude only; it does not yet check temporal logic properties.",
        "- Next step: add formal transition constraints and evaluate against downstream task failures.",
        "",
    ]
    report_path = os.path.join(output_dir, "runtime_monitor_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Run a runtime assurance monitor over reliability scores.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", default="outputs/runtime_monitor")
    parser.add_argument("--score-column", default="embedding_risk_score")
    parser.add_argument("--timestamp-column", default=None)
    parser.add_argument("--suspect-q", type=float, default=0.75)
    parser.add_argument("--recover-q", type=float, default=0.9)
    parser.add_argument("--review-q", type=float, default=0.97)
    parser.add_argument("--recover-patience", type=int, default=2)
    parser.add_argument("--review-patience", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    metrics = run_runtime_monitor(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        score_column=args.score_column,
        timestamp_column=args.timestamp_column,
        suspect_q=args.suspect_q,
        recover_q=args.recover_q,
        review_q=args.review_q,
        recover_patience=args.recover_patience,
        review_patience=args.review_patience,
    )
    print("Runtime monitor completed.")
    print(f"Samples: {metrics['samples']}")
    print(f"State counts: {metrics['state_counts']}")
    print(f"Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
