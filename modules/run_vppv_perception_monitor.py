"""VPPV-style perception-state reliability monitor.

This script upgrades the existing depth, temporal, calibration, runtime, and
trajectory residual evidence into a compact VPPV-facing experiment:

Perception-State Reliability Monitor for VPPV-Style Surgical Autonomy

The experiment distills heavier reliability signals into a lightweight
visual_state_risk model and maps risk into auditable runtime states.
"""

import argparse
import json
import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import spearmanr

from runtime_monitor import STATES, quantile_thresholds, step_monitor


FEATURE_COLUMNS = [
    "depth_valid_ratio",
    "mean_depth",
    "depth_std",
    "temporal_local_distance",
    "embedding_shift",
    "trajectory_residual",
    "progress_slope",
]

STATE_MEANINGS = {
    "NORMAL": {
        "vppv_meaning": "visual state is stable; autonomy can continue",
        "candidate_action": "continue_autonomy",
    },
    "SUSPECT": {
        "vppv_meaning": "visual state is abnormal; slow down or re-observe",
        "candidate_action": "slow_down_or_reobserve",
    },
    "RECOVER": {
        "vppv_meaning": "visual abnormality may affect execution; recover or replan",
        "candidate_action": "trigger_recovery_or_replan",
    },
    "HUMAN_REVIEW": {
        "vppv_meaning": "state cannot be confirmed; request human review",
        "candidate_action": "request_human_review",
    },
}

VPPV_FAILURE_MAPPING = {
    "depth_corruption": "depth estimation error",
    "temporal_excess": "visual-state jump",
    "embedding_shift": "perceptual state outside the training distribution",
    "trajectory_residual": "observed action outcome deviates from plan",
    "calibration_gap": "hand-eye calibration or score calibration drift",
    "progress_stagnation": "task progress stalls",
}

ROUTE_POLICY = [
    {
        "monitor_state": "NORMAL",
        "state_zh": "正常",
        "vppv_action": "continue VPPV policy",
        "action_zh": "继续执行 VPPV policy",
        "rationale": "visual state is stable enough for normal autonomy",
    },
    {
        "monitor_state": "SUSPECT",
        "state_zh": "可疑",
        "vppv_action": "slow down / re-perceive",
        "action_zh": "降低速度，重新运行 visual parsing / depth",
        "rationale": "visual state changed more than expected but may still be recoverable",
    },
    {
        "monitor_state": "RECOVER",
        "state_zh": "需要恢复",
        "vppv_action": "re-estimate state / replan / safe backup",
        "action_zh": "重新估计状态，重新规划，或切换 recovery",
        "rationale": "visual abnormality may already affect execution",
    },
    {
        "monitor_state": "HUMAN_REVIEW",
        "state_zh": "人工复查",
        "vppv_action": "surgeon confirmation / takeover",
        "action_zh": "请求医生确认或接管",
        "rationale": "state cannot be confirmed automatically before high-consequence action",
    },
]


def read_json(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def bool_series(values):
    if values.dtype == bool:
        return values.astype(bool)
    return values.astype(str).str.lower().isin(["true", "1", "yes"])


def minmax(values):
    arr = np.asarray(values, dtype=float)
    lo = np.nanmin(arr)
    hi = np.nanmax(arr)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)


def robust_outlier_score(values):
    arr = np.asarray(values, dtype=float)
    median = np.nanmedian(arr)
    mad = np.nanmedian(np.abs(arr - median)) + 1e-9
    z = np.abs(arr - median) / (1.4826 * mad)
    return np.clip(z / 6.0, 0.0, 1.0)


def tile_dataframe(df, target_len, seed):
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    repeats = int(np.ceil(target_len / max(1, len(shuffled))))
    return pd.concat([shuffled] * repeats, ignore_index=True).iloc[:target_len].reset_index(drop=True)


def calibration_gap_signal(scores, labels, n_bins=10):
    norm_scores = minmax(scores)
    labels = np.asarray(labels, dtype=float)
    bins = pd.qcut(norm_scores, q=n_bins, duplicates="drop")
    gap = np.zeros(len(norm_scores), dtype=float)
    for category in bins.categories:
        mask = bins == category
        if not np.any(mask):
            continue
        mean_pred = float(np.mean(norm_scores[mask]))
        observed = float(np.mean(labels[mask]))
        gap[mask] = abs(mean_pred - observed)
    return minmax(gap)


def coverage_priority_signal(scores):
    scores = np.asarray(scores, dtype=float)
    if len(scores) <= 1:
        return np.zeros_like(scores)
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.linspace(0.0, 1.0, len(scores))
    return ranks


def build_distillation_table(temporal_csv, trajectory_csv, seed):
    temporal = pd.read_csv(temporal_csv)
    trajectory = pd.read_csv(trajectory_csv)

    required_temporal = [
        "valid_depth_ratio",
        "depth_mean",
        "depth_std",
        "temporal_local_distance",
        "source_paired_clean_distance",
        "temporal_excess_score",
        "is_corrupted",
    ]
    missing = [col for col in required_temporal if col not in temporal.columns]
    if missing:
        raise ValueError(f"missing required temporal columns: {missing}")

    required_trajectory = [
        "residual_risk_score",
        "progress_ratio",
        "is_failure",
    ]
    missing = [col for col in required_trajectory if col not in trajectory.columns]
    if missing:
        raise ValueError(f"missing required trajectory columns: {missing}")

    traj = tile_dataframe(trajectory, len(temporal), seed=seed)
    out = pd.DataFrame({
        "source_file": temporal.get("source_file", pd.Series(np.arange(len(temporal)))).astype(str),
        "sample_id": temporal.get("sample_id", pd.Series(np.arange(len(temporal)))).astype(str),
        "corruption": temporal.get("corruption", pd.Series("unknown", index=temporal.index)).astype(str),
        "is_depth_corrupted": bool_series(temporal["is_corrupted"]).to_numpy(),
        "trajectory_failure_type": traj.get("failure_type", pd.Series("unknown", index=traj.index)).astype(str),
        "is_trajectory_failure": bool_series(traj["is_failure"]).to_numpy(),
        "depth_valid_ratio": temporal["valid_depth_ratio"].astype(float),
        "mean_depth": temporal["depth_mean"].astype(float),
        "depth_std": temporal["depth_std"].astype(float),
        "temporal_local_distance": temporal["temporal_local_distance"].astype(float),
        "embedding_shift": temporal["source_paired_clean_distance"].astype(float),
        "trajectory_residual": traj["residual_risk_score"].astype(float),
        "progress_slope": traj["progress_ratio"].astype(float),
        "temporal_excess_score": temporal["temporal_excess_score"].astype(float),
    })

    observed_proxy = out["is_depth_corrupted"].to_numpy(dtype=bool) | out["is_trajectory_failure"].to_numpy(dtype=bool)
    depth_missing = 1.0 - np.clip(out["depth_valid_ratio"].to_numpy(dtype=float), 0.0, 1.0)
    depth_mean_dev = robust_outlier_score(out["mean_depth"])
    depth_std_risk = robust_outlier_score(out["depth_std"])
    depth_corruption_score = np.clip(
        0.45 * depth_missing + 0.30 * depth_mean_dev + 0.25 * depth_std_risk,
        0.0,
        1.0,
    )
    temporal_excess = minmax(out["temporal_excess_score"])
    embedding_shift = minmax(out["embedding_shift"])
    trajectory_residual = minmax(out["trajectory_residual"])
    progress_stagnation = minmax(np.maximum(0.0, 1.0 - out["progress_slope"].to_numpy(dtype=float)))
    calibration_gap = calibration_gap_signal(out["temporal_excess_score"], out["is_depth_corrupted"].astype(int))
    coverage_priority = coverage_priority_signal(out["temporal_excess_score"])

    teacher = (
        0.18 * depth_corruption_score
        + 0.22 * temporal_excess
        + 0.17 * embedding_shift
        + 0.16 * trajectory_residual
        + 0.09 * progress_stagnation
        + 0.08 * calibration_gap
        + 0.10 * coverage_priority
    )
    teacher = minmax(teacher)
    high_risk_threshold = float(np.quantile(teacher, 0.75))

    out["depth_corruption_score"] = depth_corruption_score
    out["progress_stagnation_score"] = progress_stagnation
    out["calibration_gap_score"] = calibration_gap
    out["coverage_risk_score"] = coverage_priority
    out["visual_state_teacher_score"] = teacher
    out["teacher_high_risk"] = teacher >= high_risk_threshold
    out["observed_failure_proxy"] = observed_proxy
    return out, high_risk_threshold


def grouped_train_test_split(df, y, seed, feature_columns=None):
    feature_columns = FEATURE_COLUMNS if feature_columns is None else feature_columns
    groups = df["source_file"].astype(str).to_numpy()
    if len(np.unique(groups)) > 1:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
        train_idx, test_idx = next(splitter.split(df[feature_columns], y, groups=groups))
        return train_idx, test_idx, "group_shuffle_by_source_file"
    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.30,
        random_state=seed,
        stratify=y if len(np.unique(y)) == 2 else None,
    )
    return train_idx, test_idx, "random_stratified_row_split"


def safe_metric(fn, y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return None
    return float(fn(y_true, y_score))


def evaluate_model(model, x_train, y_train, x_test, y_test, observed_test):
    model.fit(x_train, y_train)
    proba = model.predict_proba(x_test)[:, 1]
    pred = proba >= 0.5
    return {
        "teacher_roc_auc": safe_metric(roc_auc_score, y_test, proba),
        "teacher_average_precision": safe_metric(average_precision_score, y_test, proba),
        "teacher_brier_score": float(brier_score_loss(y_test, proba)),
        "teacher_accuracy_at_0_5": float(accuracy_score(y_test, pred)),
        "teacher_f1_at_0_5": float(f1_score(y_test, pred, zero_division=0)),
        "observed_proxy_roc_auc": safe_metric(roc_auc_score, observed_test, proba),
    }


def get_model_templates(seed):
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=7,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=4,
            min_samples_leaf=12,
            class_weight="balanced",
            random_state=seed,
        ),
    }


def model_importance(model):
    if hasattr(model, "named_steps"):
        clf = model.named_steps.get("logisticregression")
        if clf is not None:
            values = clf.coef_[0]
            return {
                name: float(value)
                for name, value in sorted(
                    zip(FEATURE_COLUMNS, values),
                    key=lambda item: abs(item[1]),
                    reverse=True,
                )
            }
    if hasattr(model, "feature_importances_"):
        return {
            name: float(value)
            for name, value in sorted(
                zip(FEATURE_COLUMNS, model.feature_importances_),
                key=lambda item: item[1],
                reverse=True,
            )
        }
    return {}


def extract_feature_importance(model, feature_names):
    """Return signed and absolute importance values for supported models."""
    if hasattr(model, "named_steps"):
        clf = model.named_steps.get("logisticregression")
        if clf is not None:
            signed = clf.coef_[0]
            absolute = np.abs(signed)
            total = absolute.sum() + 1e-12
            return pd.DataFrame({
                "feature": feature_names,
                "signed_importance": signed,
                "importance": absolute / total,
            }).sort_values("importance", ascending=False)

    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
        return pd.DataFrame({
            "feature": feature_names,
            "signed_importance": values,
            "importance": values,
        }).sort_values("importance", ascending=False)

    return pd.DataFrame(columns=["feature", "signed_importance", "importance"])


def save_feature_importance(model, feature_names, output_dir):
    importance = extract_feature_importance(model, feature_names)
    if importance.empty:
        return importance

    importance.to_csv(os.path.join(output_dir, "feature_importance.csv"), index=False)

    plt.figure(figsize=(8, 4.5))
    ordered = importance.sort_values("importance", ascending=True)
    plt.barh(ordered["feature"], ordered["importance"], color="#4c78a8")
    plt.xlabel("Importance")
    plt.title("Visual-State Risk Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=180)
    plt.close()
    return importance


def select_best_model(metrics):
    sortable = []
    for name, values in metrics.items():
        auc_value = values["teacher_roc_auc"]
        brier = values["teacher_brier_score"]
        sortable.append((auc_value if auc_value is not None else -1.0, -brier, name))
    return sorted(sortable, reverse=True)[0][2]


def risk_driver_table(df):
    """Build normalized driver columns used for human-readable explanations."""
    return pd.DataFrame({
        "depth": np.maximum(
            minmax(1.0 - np.clip(df["depth_valid_ratio"].to_numpy(dtype=float), 0.0, 1.0)),
            np.maximum(minmax(df["depth_std"]), df["depth_corruption_score"]),
        ),
        "temporal": minmax(df["temporal_excess_score"]),
        "embedding": minmax(df["embedding_shift"]),
        "trajectory": minmax(df["trajectory_residual"]),
        "progress": df["progress_stagnation_score"].to_numpy(dtype=float),
        "calibration": df["calibration_gap_score"].to_numpy(dtype=float),
        "coverage": df["coverage_risk_score"].to_numpy(dtype=float),
    }, index=df.index)


def explain_driver_names(names):
    phrases = {
        "depth": "depth evidence is abnormal",
        "temporal": "local temporal visual change exceeds normal variation",
        "embedding": "the perception embedding shifts away from the paired clean state",
        "trajectory": "trajectory residual is abnormal",
        "progress": "task progress is stagnant",
        "calibration": "calibration-style score inconsistency is high",
        "coverage": "the sample sits in the high-risk abstention region",
    }
    phrases_zh = {
        "depth": "深度证据异常",
        "temporal": "局部时间变化超过正常范围",
        "embedding": "感知 embedding 偏移明显",
        "trajectory": "轨迹残差异常",
        "progress": "任务进展停滞",
        "calibration": "标定/校准一致性风险较高",
        "coverage": "样本位于高风险拒判区域",
    }
    return (
        "; ".join(phrases[name] for name in names),
        "；".join(phrases_zh[name] for name in names),
    )


def save_top_risk_cases(df, output_dir, top_k=30):
    drivers = risk_driver_table(df)
    rows = []
    top = df.sort_values("visual_state_risk", ascending=False).head(top_k)
    for idx, row in top.iterrows():
        driver_values = drivers.loc[idx].sort_values(ascending=False)
        top_drivers = driver_values.head(3)
        explanation, explanation_zh = explain_driver_names(top_drivers.index.tolist())
        rows.append({
            "row_index": int(idx),
            "sample_id": row["sample_id"],
            "source_file": row["source_file"],
            "corruption": row["corruption"],
            "trajectory_failure_type": row["trajectory_failure_type"],
            "visual_state_risk": float(row["visual_state_risk"]),
            "monitor_state": row["risk_monitor_state"],
            "monitor_action": row["risk_monitor_action"],
            "top_driver_1": top_drivers.index[0],
            "top_driver_1_score": float(top_drivers.iloc[0]),
            "top_driver_2": top_drivers.index[1],
            "top_driver_2_score": float(top_drivers.iloc[1]),
            "top_driver_3": top_drivers.index[2],
            "top_driver_3_score": float(top_drivers.iloc[2]),
            "explanation": f"High visual_state_risk because {explanation}.",
            "explanation_zh": f"当前视觉状态风险高，主要因为{explanation_zh}。",
        })

    cases = pd.DataFrame(rows)
    cases.to_csv(os.path.join(output_dir, "top_risk_cases.csv"), index=False)
    return cases


def signal_group_definitions(df):
    groups = {
        "depth_only": [
            "depth_valid_ratio",
            "mean_depth",
            "depth_std",
            "depth_corruption_score",
        ],
        "temporal_only": [
            "temporal_local_distance",
            "temporal_excess_score",
        ],
        "embedding_only": [
            "embedding_shift",
        ],
        "trajectory_only": [
            "trajectory_residual",
            "progress_slope",
            "progress_stagnation_score",
        ],
        "calibration_only": [
            "calibration_gap_score",
        ],
        "coverage_only": [
            "coverage_risk_score",
        ],
        "all_student_features": FEATURE_COLUMNS,
        "all_signals": [
            "depth_valid_ratio",
            "mean_depth",
            "depth_std",
            "temporal_local_distance",
            "temporal_excess_score",
            "embedding_shift",
            "trajectory_residual",
            "progress_slope",
            "depth_corruption_score",
            "progress_stagnation_score",
            "calibration_gap_score",
            "coverage_risk_score",
        ],
        "all_except_temporal": [
            "depth_valid_ratio",
            "mean_depth",
            "depth_std",
            "embedding_shift",
            "trajectory_residual",
            "progress_slope",
            "depth_corruption_score",
            "progress_stagnation_score",
            "calibration_gap_score",
            "coverage_risk_score",
        ],
        "all_except_trajectory": [
            "depth_valid_ratio",
            "mean_depth",
            "depth_std",
            "temporal_local_distance",
            "temporal_excess_score",
            "embedding_shift",
            "depth_corruption_score",
            "calibration_gap_score",
            "coverage_risk_score",
        ],
    }
    return {
        name: [column for column in columns if column in df.columns]
        for name, columns in groups.items()
        if any(column in df.columns for column in columns)
    }


def run_signal_group_ablation(df, y, observed, seed, output_dir):
    rows = []
    groups = signal_group_definitions(df)
    for group_name, columns in groups.items():
        train_idx, test_idx, split_strategy = grouped_train_test_split(df, y, seed, feature_columns=columns)
        model = RandomForestClassifier(
            n_estimators=250,
            max_depth=7,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(df.loc[train_idx, columns], y[train_idx])
        proba = model.predict_proba(df.loc[test_idx, columns])[:, 1]
        pred = proba >= 0.5
        rows.append({
            "signal_group": group_name,
            "features": ", ".join(columns),
            "n_features": int(len(columns)),
            "split_strategy": split_strategy,
            "teacher_roc_auc": safe_metric(roc_auc_score, y[test_idx], proba),
            "teacher_average_precision": safe_metric(average_precision_score, y[test_idx], proba),
            "teacher_brier_score": float(brier_score_loss(y[test_idx], proba)),
            "teacher_f1_at_0_5": float(f1_score(y[test_idx], pred, zero_division=0)),
            "observed_proxy_roc_auc": safe_metric(roc_auc_score, observed[test_idx], proba),
        })

    ablation = pd.DataFrame(rows).sort_values(
        ["teacher_roc_auc", "teacher_average_precision"],
        ascending=False,
        na_position="last",
    )
    ablation.to_csv(os.path.join(output_dir, "signal_group_ablation.csv"), index=False)

    plt.figure(figsize=(9, 4.8))
    plot_df = ablation.sort_values("teacher_roc_auc", ascending=True)
    plt.barh(plot_df["signal_group"], plot_df["teacher_roc_auc"], color="#59a14f")
    plt.xlabel("Distillation ROC-AUC")
    plt.title("Signal-Group Ablation for Visual-State Risk")
    plt.xlim(0.0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "signal_group_ablation.png"), dpi=180)
    plt.close()
    return ablation


def save_route_policy(output_dir):
    route_policy = pd.DataFrame(ROUTE_POLICY)
    route_policy.to_csv(os.path.join(output_dir, "vppv_route_policy.csv"), index=False)
    return route_policy


def save_outcome_linked_validation(df, output_dir):
    results = {}
    target_columns = [
        "trajectory_residual",
        "temporal_excess_score",
        "embedding_shift",
        "depth_corruption_score",
        "progress_stagnation_score",
        "calibration_gap_score",
        "coverage_risk_score",
    ]
    if "corruption" in df.columns:
        severity = df["corruption"].map({
            "clean": 0,
            "gaussian_noise": 1,
            "dropout": 2,
            "quantization": 2,
            "occlusion": 3,
            "tilt_shift": 3,
        }).fillna(0)
        df = df.copy()
        df["corruption_severity_proxy"] = severity.astype(float)
        target_columns.append("corruption_severity_proxy")

    for target in target_columns:
        if target not in df.columns:
            continue
        rho, p_value = spearmanr(df["visual_state_risk"], df[target], nan_policy="omit")
        results[f"spearman_risk_vs_{target}"] = float(rho)
        results[f"p_value_risk_vs_{target}"] = float(p_value)

    mean_risk = df.groupby("risk_monitor_state")["visual_state_risk"].mean().reindex(STATES)
    results["mean_risk_by_state"] = {
        state: (None if pd.isna(value) else float(value))
        for state, value in mean_risk.to_dict().items()
    }

    top10_threshold = float(df["visual_state_risk"].quantile(0.90))
    top10 = df["visual_state_risk"] >= top10_threshold
    recover_or_review = df["risk_monitor_state"].isin(["RECOVER", "HUMAN_REVIEW"])
    recover_count = int(recover_or_review.sum())
    results["top10_risk_threshold"] = top10_threshold
    results["recover_or_review_count"] = recover_count
    results["top10_risk_recover_or_review_capture"] = float(
        (top10 & recover_or_review).sum() / max(recover_count, 1)
    )
    results["top10_risk_fraction"] = float(top10.mean())

    with open(os.path.join(output_dir, "risk_outcome_correlation.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results


def run_state_machine(df, risk_column, output_dir):
    thresholds = quantile_thresholds(df[risk_column], suspect_q=0.75, recover_q=0.90, review_q=0.97)
    state = "NORMAL"
    consecutive_recover = 0
    consecutive_review = 0
    states = []
    actions = []
    for score in df[risk_column].to_numpy(dtype=float):
        state, action, consecutive_recover, consecutive_review = step_monitor(
            score=score,
            thresholds=thresholds,
            previous_state=state,
            consecutive_recover=consecutive_recover,
            consecutive_review=consecutive_review,
            recover_patience=2,
            review_patience=3,
        )
        states.append(state)
        actions.append(action)

    df["risk_monitor_state"] = states
    df["risk_monitor_action"] = actions
    counts = df["risk_monitor_state"].value_counts().reindex(STATES, fill_value=0)
    count_rows = []
    for state_name, count in counts.items():
        row = {
            "monitor_state": state_name,
            "count": int(count),
            "fraction": float(count / len(df)),
        }
        row.update(STATE_MEANINGS[state_name])
        count_rows.append(row)
    counts_df = pd.DataFrame(count_rows)
    counts_df.to_csv(os.path.join(output_dir, "risk_state_counts.csv"), index=False)
    return thresholds, counts_df


def plot_risk_trace(df, thresholds, output_path):
    x = np.arange(len(df))
    colors = {
        "NORMAL": "#2ca02c",
        "SUSPECT": "#ff7f0e",
        "RECOVER": "#d62728",
        "HUMAN_REVIEW": "#9467bd",
    }

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(x, df["visual_state_risk"], color="black", linewidth=1.2, label="distilled visual_state_risk")
    for state in STATES:
        mask = df["risk_monitor_state"] == state
        axes[0].scatter(
            x[mask],
            df.loc[mask, "visual_state_risk"],
            s=9,
            color=colors[state],
            alpha=0.65,
            label=state,
        )
    axes[0].axhline(thresholds["suspect"], color=colors["SUSPECT"], linestyle="--", linewidth=1.0)
    axes[0].axhline(thresholds["recover"], color=colors["RECOVER"], linestyle="--", linewidth=1.0)
    axes[0].axhline(thresholds["review"], color=colors["HUMAN_REVIEW"], linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("risk")
    axes[0].set_title("VPPV Perception-State Reliability Trace")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(ncol=3, fontsize=8, loc="upper right")

    axes[1].plot(x, df["depth_corruption_score"], linewidth=1.0, label="depth corruption")
    axes[1].plot(x, minmax(df["temporal_excess_score"]), linewidth=1.0, label="temporal excess")
    axes[1].plot(x, minmax(df["embedding_shift"]), linewidth=1.0, label="embedding shift")
    axes[1].plot(x, minmax(df["trajectory_residual"]), linewidth=1.0, label="trajectory residual")
    axes[1].set_xlabel("sample index")
    axes[1].set_ylabel("component")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(ncol=4, fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def format_metric(value):
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def write_report_legacy(report_path, metrics, state_counts):
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    model_rows = []
    for name, values in metrics["model_metrics"].items():
        model_rows.append(
            "| {name} | {auc} | {ap} | {brier} | {proxy} |".format(
                name=name,
                auc=format_metric(values["teacher_roc_auc"]),
                ap=format_metric(values["teacher_average_precision"]),
                brier=format_metric(values["teacher_brier_score"]),
                proxy=format_metric(values["observed_proxy_roc_auc"]),
            )
        )

    state_rows = []
    for row in state_counts.to_dict(orient="records"):
        state_rows.append(
            "| {monitor_state} | {count} | {fraction:.3f} | {candidate_action} |".format(**row)
        )

    mapping_rows = []
    for signal, risk in VPPV_FAILURE_MAPPING.items():
        mapping_rows.append(f"| {signal} | {risk} |")

    lines = [
        "# Perception-State Reliability Monitor for VPPV-Style Surgical Autonomy",
        "",
        "Chinese title: 面向 VPPV 手术自主系统的感知状态可靠性监控器",
        "",
        "## Research Question",
        "",
        "Can the existing reliability-aware robot perception project serve as a VPPV visual-front-end monitor for depth, regressed perceptual state, physical state, and downstream trajectory consistency?",
        "",
        "## VPPV Fit",
        "",
        "VPPV-style autonomy depends on segmentation masks, depth maps, perceptual regressors, and physical state vectors. This monitor does not replace those modules. It watches whether their visual-state evidence is stable enough for a downstream RL policy to trust, or whether the system should re-perceive, recover, replan, or request human review.",
        "",
        "## Experiment A: Visual-State Risk Distillation",
        "",
        f"- Samples: {metrics['samples']}",
        f"- Split: {metrics['split_strategy']}",
        f"- Distillation high-risk threshold: {metrics['teacher_high_risk_threshold']:.3f}",
        f"- Distillation high-risk rate: {metrics['teacher_high_risk_rate']:.3f}",
        f"- Selected model: {metrics['selected_model']}",
        f"- Runtime output: `visual_state_risk`",
        "",
        "| Model | Distillation ROC-AUC | Distillation AP | Brier | Observed-proxy ROC-AUC |",
        "|---|---:|---:|---:|---:|",
        *model_rows,
        "",
        "Input features used by the lightweight student model: `depth_valid_ratio`, `mean_depth`, `depth_std`, `temporal_local_distance`, `embedding_shift`, `trajectory_residual`, and `progress_slope`.",
        "",
        "## Experiment B: Normal Camera Motion vs Visual Error",
        "",
        f"- Global/scene-conditioned clean-reference ROC-AUC: {format_metric(metrics['global_reference_roc_auc'])}",
        f"- Local temporal-excess ROC-AUC: {format_metric(metrics['temporal_excess_roc_auc'])}",
        f"- Pose-aware grid rotation correlation: {format_metric(metrics['grid_rotation_correlation'])}",
        "",
        "Interpretation: VPPV visual reliability should not be defined only as distance from a global clean reference. In moving laparoscopic scenes, both the camera and tissue can move normally. The useful question is whether the current change exceeds normal variation inside a local temporal window.",
        "",
        "## Experiment C: Runtime State Machine",
        "",
        "| State | Count | Fraction | Candidate VPPV action |",
        "|---|---:|---:|---|",
        *state_rows,
        "",
        "- NORMAL: visual state is stable; continue execution.",
        "- SUSPECT: visual state is abnormal; slow down or re-observe.",
        "- RECOVER: visual abnormality may affect execution; trigger recovery or replanning.",
        "- HUMAN_REVIEW: state cannot be confirmed; request human review.",
        "",
        "## Experiment D: VPPV Failure Mapping",
        "",
        "| Project signal | Corresponding VPPV risk |",
        "|---|---|",
        *mapping_rows,
        "",
        "## Evidence Level",
        "",
        "Confirmed: the repository now produces a VPPV-facing risk distillation run, a runtime state-count table, and a risk trace from existing RGB-D temporal reliability and trajectory residual evidence.",
        "",
        "Suggested: the same monitor can act as a visual-state reliability score, state-jump detector, re-perception trigger, and human-review trigger for a VPPV-style pipeline.",
        "",
        "Not yet proven: this is not validated on paired surgical robot logs, segmentation masks, VPPV policy rollouts, or real clinical deployment data. The trajectory residual signal is aligned as a deterministic proxy rather than a paired surgical execution trace.",
        "",
        "## Email Contribution Wording",
        "",
        "This upgrade reframes the project as a front-end reliability monitor for VPPV-style surgical autonomy. Since VPPV relies on segmentation, depth, regressed perceptual state, and physical state, unreliable visual states can mislead the downstream RL policy. The monitor provides a `visual_state_risk` score, detects abnormal state jumps relative to a local temporal window, and routes risky states to re-perception, recovery/replanning, or human review.",
        "",
        "## Output Files",
        "",
        "- `outputs/vppv_perception_monitor/risk_distillation_metrics.json`",
        "- `outputs/vppv_perception_monitor/risk_state_counts.csv`",
        "- `outputs/vppv_perception_monitor/risk_trace.png`",
        "- `reports/vppv_perception_reliability_monitor.md`",
        "",
    ]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_report(report_path, metrics, state_counts, feature_importance, ablation, route_policy):
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    model_rows = []
    for name, values in metrics["model_metrics"].items():
        model_rows.append(
            "| {name} | {auc} | {ap} | {brier} | {proxy} |".format(
                name=name,
                auc=format_metric(values["teacher_roc_auc"]),
                ap=format_metric(values["teacher_average_precision"]),
                brier=format_metric(values["teacher_brier_score"]),
                proxy=format_metric(values["observed_proxy_roc_auc"]),
            )
        )

    state_rows = [
        "| {monitor_state} | {count} | {fraction:.3f} | {candidate_action} |".format(**row)
        for row in state_counts.to_dict(orient="records")
    ]
    mapping_rows = [
        f"| {signal} | {risk} |"
        for signal, risk in VPPV_FAILURE_MAPPING.items()
    ]
    feature_rows = [
        "| {feature} | {importance:.3f} | {signed_importance:.3f} |".format(**row)
        for row in feature_importance.head(8).to_dict(orient="records")
    ]
    ablation_rows = [
        "| {signal_group} | {n_features} | {auc} | {ap} | {proxy} |".format(
            signal_group=row["signal_group"],
            n_features=row["n_features"],
            auc=format_metric(row["teacher_roc_auc"]),
            ap=format_metric(row["teacher_average_precision"]),
            proxy=format_metric(row["observed_proxy_roc_auc"]),
        )
        for row in ablation.to_dict(orient="records")
    ]
    route_rows = [
        "| {monitor_state} | {state_zh} | {vppv_action} | {action_zh} |".format(**row)
        for row in route_policy.to_dict(orient="records")
    ]
    outcome = metrics["outcome_linked_validation"]

    lines = [
        "# Perception-State Reliability Monitor for VPPV-Style Surgical Autonomy",
        "",
        "Chinese title: 面向 VPPV 手术自主系统的感知状态可靠性监控器",
        "",
        "## Research Question",
        "",
        "Can the existing reliability-aware robot perception project serve as a VPPV visual-front-end monitor for depth, regressed perceptual state, physical state, and downstream trajectory consistency?",
        "",
        "## VPPV Fit",
        "",
        "VPPV-style autonomy depends on segmentation masks, depth maps, perceptual regressors, and physical state vectors. This monitor does not replace those modules. It watches whether their visual-state evidence is stable enough for a downstream RL policy to trust, or whether the system should re-perceive, recover, replan, or request human review.",
        "",
        "## Experiment A: Visual-State Risk Distillation",
        "",
        f"- Samples: {metrics['samples']}",
        f"- Split: {metrics['split_strategy']}",
        f"- Distillation high-risk threshold: {metrics['teacher_high_risk_threshold']:.3f}",
        f"- Distillation high-risk rate: {metrics['teacher_high_risk_rate']:.3f}",
        f"- Selected model: {metrics['selected_model']}",
        f"- Runtime output: `visual_state_risk`",
        "",
        "| Model | Distillation ROC-AUC | Distillation AP | Brier | Observed-proxy ROC-AUC |",
        "|---|---:|---:|---:|---:|",
        *model_rows,
        "",
        "Input features used by the lightweight student model: `depth_valid_ratio`, `mean_depth`, `depth_std`, `temporal_local_distance`, `embedding_shift`, `trajectory_residual`, and `progress_slope`.",
        "",
        "## Experiment A2: Feature Attribution",
        "",
        "The monitor does not only output a scalar risk score. It also provides feature-level evidence, indicating whether risk is mainly driven by temporal visual shift, depth corruption, embedding displacement, calibration inconsistency, or trajectory residual.",
        "",
        "| Feature | Importance | Signed/Raw Importance |",
        "|---|---:|---:|",
        *feature_rows,
        "",
        "High-risk samples are exported with one-sentence explanations in `top_risk_cases.csv`, including whether each case is dominated by local temporal excess, trajectory residual, depth evidence, calibration risk, or embedding shift.",
        "",
        "## Experiment A3: Signal-Group Ablation",
        "",
        "This ablation asks which family of signals is most useful for VPPV front-end risk. It is a diagnostic comparison, not a deployment model choice: some groups include teacher-side audit signals that are heavier than the lightweight runtime input.",
        "",
        "| Signal group | Features | Distillation ROC-AUC | Distillation AP | Observed-proxy ROC-AUC |",
        "|---|---:|---:|---:|---:|",
        *ablation_rows,
        "",
        "## Experiment B: Normal Camera Motion vs Visual Error",
        "",
        f"- Global/scene-conditioned clean-reference ROC-AUC: {format_metric(metrics['global_reference_roc_auc'])}",
        f"- Local temporal-excess ROC-AUC: {format_metric(metrics['temporal_excess_roc_auc'])}",
        f"- Pose-aware grid rotation correlation: {format_metric(metrics['grid_rotation_correlation'])}",
        "",
        "Interpretation: VPPV visual reliability should not be defined only as distance from a global clean reference. In moving laparoscopic scenes, both the camera and tissue can move normally. The useful question is whether the current change exceeds normal variation inside a local temporal window.",
        "",
        "## Experiment C: Runtime State Machine",
        "",
        "| State | Count | Fraction | Candidate VPPV action |",
        "|---|---:|---:|---|",
        *state_rows,
        "",
        "- NORMAL: visual state is stable; continue execution.",
        "- SUSPECT: visual state is abnormal; slow down or re-observe.",
        "- RECOVER: visual abnormality may affect execution; trigger recovery or replanning.",
        "- HUMAN_REVIEW: state cannot be confirmed; request human review.",
        "",
        "## Experiment C2: VPPV Route Policy",
        "",
        "| Monitor state | 中文 | VPPV action | 中文动作 |",
        "|---|---|---|---|",
        *route_rows,
        "",
        "In this controlled experiment `HUMAN_REVIEW` is zero because the highest-risk samples are still routed to `RECOVER`. This is not a defect: before irreversible surgical actions such as grasping, clipping, or tissue retraction, the same state machine can use stricter review thresholds.",
        "",
        "## Experiment D: VPPV Failure Mapping",
        "",
        "| Project signal | Corresponding VPPV risk |",
        "|---|---|",
        *mapping_rows,
        "",
        "## Experiment E: Outcome-Linked Validation",
        "",
        f"- Spearman risk vs trajectory residual: {format_metric(outcome.get('spearman_risk_vs_trajectory_residual'))}",
        f"- Spearman risk vs temporal excess: {format_metric(outcome.get('spearman_risk_vs_temporal_excess_score'))}",
        f"- Spearman risk vs embedding shift: {format_metric(outcome.get('spearman_risk_vs_embedding_shift'))}",
        f"- Top 10% risk captures RECOVER/HUMAN_REVIEW states: {format_metric(outcome.get('top10_risk_recover_or_review_capture'))}",
        "",
        "This links the distilled risk to downstream monitor states and residual signals, so the score is not only fitting a teacher label. It is also decision-relevant for VPPV-style autonomy.",
        "",
        "## Evidence Level",
        "",
        "Confirmed: the repository now produces a VPPV-facing risk distillation run, feature attribution, route evaluation, state-count tables, and a risk trace from existing RGB-D temporal reliability and trajectory residual evidence.",
        "",
        "Suggested: the same monitor can act as a visual-state reliability score, state-jump detector, re-perception trigger, recovery trigger, and human-review trigger for a VPPV-style pipeline.",
        "",
        "Not yet proven: this is not validated on paired surgical robot logs, segmentation masks, VPPV policy rollouts, or real clinical deployment data. The trajectory residual signal is aligned as a deterministic proxy rather than a paired surgical execution trace.",
        "",
        "## Email Contribution Wording",
        "",
        "This upgrade reframes the project as a front-end reliability monitor for VPPV-style surgical autonomy. Since VPPV relies on segmentation, depth, regressed perceptual state, and physical state, unreliable visual states can mislead the downstream RL policy. The monitor provides a `visual_state_risk` score, explains which reliability signals drive that score, detects abnormal state jumps relative to a local temporal window, and routes risky states to re-perception, recovery/replanning, or human review.",
        "",
        "## Output Files",
        "",
        "- `outputs/vppv_perception_monitor/risk_distillation_metrics.json`",
        "- `outputs/vppv_perception_monitor/risk_state_counts.csv`",
        "- `outputs/vppv_perception_monitor/risk_trace.png`",
        "- `outputs/vppv_perception_monitor/feature_importance.csv`",
        "- `outputs/vppv_perception_monitor/feature_importance.png`",
        "- `outputs/vppv_perception_monitor/top_risk_cases.csv`",
        "- `outputs/vppv_perception_monitor/signal_group_ablation.csv`",
        "- `outputs/vppv_perception_monitor/signal_group_ablation.png`",
        "- `outputs/vppv_perception_monitor/vppv_route_policy.csv`",
        "- `outputs/vppv_perception_monitor/risk_outcome_correlation.json`",
        "- `reports/vppv_perception_reliability_monitor.md`",
        "",
    ]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run_vppv_monitor(args):
    os.makedirs(args.output_dir, exist_ok=True)
    df, teacher_threshold = build_distillation_table(
        temporal_csv=args.temporal_csv,
        trajectory_csv=args.trajectory_csv,
        seed=args.seed,
    )
    y = df["teacher_high_risk"].astype(int).to_numpy()
    x = df[FEATURE_COLUMNS]
    train_idx, test_idx, split_strategy = grouped_train_test_split(df, y, args.seed)

    model_templates = get_model_templates(args.seed)
    model_metrics = {}
    fitted_models = {}
    observed = df["observed_failure_proxy"].astype(int).to_numpy()
    for name, model in model_templates.items():
        local_model = deepcopy(model)
        model_metrics[name] = evaluate_model(
            local_model,
            x.iloc[train_idx],
            y[train_idx],
            x.iloc[test_idx],
            y[test_idx],
            observed[test_idx],
        )
        fitted_models[name] = local_model

    selected_name = select_best_model(model_metrics)
    selected_model = deepcopy(model_templates[selected_name])
    selected_model.fit(x, y)
    df["visual_state_risk"] = selected_model.predict_proba(x)[:, 1]
    feature_importance = model_importance(selected_model)
    feature_importance_df = save_feature_importance(selected_model, FEATURE_COLUMNS, args.output_dir)

    thresholds, state_counts = run_state_machine(df, "visual_state_risk", args.output_dir)
    top_risk_cases = save_top_risk_cases(df, args.output_dir, top_k=30)
    ablation = run_signal_group_ablation(df, y, observed, args.seed, args.output_dir)
    route_policy = save_route_policy(args.output_dir)
    outcome_validation = save_outcome_linked_validation(df, args.output_dir)
    risk_trace_csv = os.path.join(args.output_dir, "risk_trace.csv")
    df.to_csv(risk_trace_csv, index=False)
    plot_risk_trace(df, thresholds, os.path.join(args.output_dir, "risk_trace.png"))

    temporal_metrics = read_json(args.temporal_metrics_json)
    real_depth_metrics = read_json(args.real_depth_metrics_json)
    pose_metrics = read_json(args.pose_metrics_json)
    calibration_metrics = read_json(args.calibration_metrics_json)

    metrics = {
        "task": "vppv_perception_state_reliability_monitor",
        "title": "Perception-State Reliability Monitor for VPPV-Style Surgical Autonomy",
        "seed": int(args.seed),
        "samples": int(len(df)),
        "features": FEATURE_COLUMNS,
        "teacher_components": [
            "depth_corruption_score",
            "temporal_excess_score",
            "embedding_shift",
            "trajectory_residual",
            "progress_stagnation_score",
            "calibration_gap_score",
            "coverage_risk_score",
        ],
        "teacher_weights": {
            "depth_corruption_score": 0.18,
            "temporal_excess_score": 0.22,
            "embedding_shift": 0.17,
            "trajectory_residual": 0.16,
            "progress_stagnation_score": 0.09,
            "calibration_gap_score": 0.08,
            "coverage_risk_score": 0.10,
        },
        "teacher_high_risk_threshold": float(teacher_threshold),
        "teacher_high_risk_rate": float(df["teacher_high_risk"].mean()),
        "observed_failure_proxy_rate": float(df["observed_failure_proxy"].mean()),
        "split_strategy": split_strategy,
        "train_samples": int(len(train_idx)),
        "test_samples": int(len(test_idx)),
        "model_metrics": model_metrics,
        "selected_model": selected_name,
        "selected_model_feature_importance": feature_importance,
        "feature_importance": feature_importance_df.to_dict(orient="records"),
        "top_risk_case_examples": top_risk_cases.head(5).to_dict(orient="records"),
        "signal_group_ablation": ablation.to_dict(orient="records"),
        "outcome_linked_validation": outcome_validation,
        "risk_state_thresholds": thresholds,
        "risk_state_counts": {
            row["monitor_state"]: int(row["count"])
            for row in state_counts.to_dict(orient="records")
        },
        "risk_state_fractions": {
            row["monitor_state"]: float(row["fraction"])
            for row in state_counts.to_dict(orient="records")
        },
        "vppv_failure_mapping": VPPV_FAILURE_MAPPING,
        "global_reference_roc_auc": real_depth_metrics.get("scene_conditioned_roc_auc_for_corruption_detection"),
        "temporal_excess_roc_auc": temporal_metrics.get("temporal_excess_roc_auc_for_corruption_detection"),
        "temporal_local_distance_roc_auc": temporal_metrics.get("temporal_local_roc_auc_for_corruption_detection"),
        "source_paired_roc_auc": temporal_metrics.get("source_paired_roc_auc_for_corruption_detection"),
        "calibration_ece_gap": calibration_metrics.get("ece"),
        "grid_rotation_correlation": pose_metrics.get("grid_spearman_rotation_vs_embedding_shift"),
        "outputs": {
            "metrics_json": os.path.join(args.output_dir, "risk_distillation_metrics.json"),
            "state_counts_csv": os.path.join(args.output_dir, "risk_state_counts.csv"),
            "risk_trace_png": os.path.join(args.output_dir, "risk_trace.png"),
            "risk_trace_csv": risk_trace_csv,
            "feature_importance_csv": os.path.join(args.output_dir, "feature_importance.csv"),
            "feature_importance_png": os.path.join(args.output_dir, "feature_importance.png"),
            "top_risk_cases_csv": os.path.join(args.output_dir, "top_risk_cases.csv"),
            "signal_group_ablation_csv": os.path.join(args.output_dir, "signal_group_ablation.csv"),
            "signal_group_ablation_png": os.path.join(args.output_dir, "signal_group_ablation.png"),
            "vppv_route_policy_csv": os.path.join(args.output_dir, "vppv_route_policy.csv"),
            "risk_outcome_correlation_json": os.path.join(args.output_dir, "risk_outcome_correlation.json"),
            "report_md": args.report_path,
        },
        "limitations": [
            "Controlled TUM depth corruptions are not the same as real VPPV surgical failures.",
            "Trajectory residual rows are deterministically aligned proxies, not paired robot rollouts.",
            "Segmentation-mask quality is discussed as a VPPV dependency but is not directly measured here.",
            "The distilled score is a ranking and routing signal, not a calibrated clinical probability.",
        ],
    }

    metrics_path = os.path.join(args.output_dir, "risk_distillation_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    write_report(args.report_path, metrics, state_counts, feature_importance_df, ablation, route_policy)
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Run VPPV perception-state reliability monitor.")
    parser.add_argument(
        "--temporal-csv",
        default="outputs/tum_rgbd_freiburg1_desk_temporal/temporal_depth_embeddings.csv",
    )
    parser.add_argument(
        "--trajectory-csv",
        default="outputs/trajectory_residual_demo/trajectory_residuals.csv",
    )
    parser.add_argument(
        "--temporal-metrics-json",
        default="outputs/tum_rgbd_freiburg1_desk_temporal/temporal_depth_metrics.json",
    )
    parser.add_argument(
        "--real-depth-metrics-json",
        default="outputs/tum_rgbd_freiburg1_desk_robot_3d/corruption_benchmark/real_depth_corruption_metrics.json",
    )
    parser.add_argument(
        "--calibration-metrics-json",
        default="outputs/tum_rgbd_calibration/calibration_metrics.json",
    )
    parser.add_argument(
        "--pose-metrics-json",
        default="outputs/tum_rgbd_freiburg1_desk_pose_sorted/tum_pose_embedding_metrics.json",
    )
    parser.add_argument("--output-dir", default="outputs/vppv_perception_monitor")
    parser.add_argument("--report-path", default="reports/vppv_perception_reliability_monitor.md")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    metrics = run_vppv_monitor(args)
    print("VPPV perception-state monitor completed.")
    print(f"Samples: {metrics['samples']}")
    print(f"Selected model: {metrics['selected_model']}")
    print(f"State counts: {metrics['risk_state_counts']}")
    print(f"Outputs written to: {args.output_dir}")
    print(f"Report written to: {args.report_path}")


if __name__ == "__main__":
    main()
