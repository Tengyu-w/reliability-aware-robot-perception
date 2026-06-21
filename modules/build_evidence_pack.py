"""Build a compact supervisor-facing evidence pack from experiment outputs."""

import argparse
import json
import os
from pathlib import Path


def read_json(path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt_metric(value):
    if isinstance(value, (int, float)):
        return f"{value:.3f}"
    return "n/a"


def find_first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return None


def build_evidence_pack(output_path, project_root):
    project_root = Path(project_root).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    synthetic_metrics = read_json(project_root / "outputs/robot_3d_multiseed/robot_3d_multiseed_metrics.json")
    real_depth_metrics = read_json(
        find_first_existing([
            project_root / "outputs/tum_rgbd_freiburg1_desk_robot_3d/corruption_benchmark/real_depth_corruption_metrics.json",
            project_root / "outputs/robot_3d_pipeline_smoke/corruption_benchmark/real_depth_corruption_metrics.json",
            project_root / "outputs/real_depth_corruption_smoke/real_depth_corruption_metrics.json",
        ]) or Path("__missing__")
    )
    video_metrics = read_json(project_root / "outputs/baseline/metrics.json")
    temporal_metrics = read_json(
        project_root / "outputs/tum_rgbd_freiburg1_desk_temporal/temporal_depth_metrics.json"
    )
    pose_metrics = read_json(
        find_first_existing([
            project_root / "outputs/tum_rgbd_freiburg1_desk_pose_sorted/tum_pose_embedding_metrics.json",
            project_root / "outputs/tum_rgbd_freiburg1_desk_pose/tum_pose_embedding_metrics.json",
        ]) or Path("__missing__")
    )
    pca_descriptor_metrics = read_json(
        project_root / "outputs/tum_rgbd_freiburg1_desk_pca_descriptor/tum_pca_depth_descriptor_metrics.json"
    )
    runtime_monitor_metrics = read_json(
        project_root / "outputs/tum_rgbd_runtime_monitor/runtime_monitor_metrics.json"
    )
    calibration_metrics = read_json(
        project_root / "outputs/tum_rgbd_calibration/calibration_metrics.json"
    )
    trajectory_metrics = read_json(
        project_root / "outputs/trajectory_residual_demo/trajectory_residual_metrics.json"
    )
    trajectory_monitor_metrics = read_json(
        project_root / "outputs/trajectory_runtime_monitor/runtime_monitor_metrics.json"
    )

    lines = [
        "# Reliability-Aware Sequential and 3D Perception Evidence Pack",
        "",
        "## Positioning",
        "",
        "One-page summary: `docs/supervisor_one_pager.md`.",
        "",
        "This project extends an ECG-style embedding reliability framework into video action understanding and robot 3D perception. The central question is whether embedding-space evidence, calibration-style uncertainty, and selective prediction can identify unreliable model or perception inputs before downstream decisions.",
        "",
        "## Evidence Layers",
        "",
        "| Layer | What it demonstrates | Current status |",
        "|---|---|---|",
        "| ECG uncertainty foundation | Prior experience with embedding/reliability analysis | External project, referenced in applications |",
        "| Video action recognition | Sequential perception embeddings and action-class reliability | Implemented in `modules/main.py` and `modules/embedding_analysis.py` |",
        "| Synthetic 3D perception | Depth-to-point-cloud embedding reliability under controlled corruptions | Implemented and runnable |",
        "| Real depth workflow | Public depth-map preparation, profiling, and corruption benchmark | Implemented; TUM RGB-D sample run completed in current outputs |",
        "",
        "## Current 3D Reliability Results",
        "",
    ]

    if synthetic_metrics:
        lines.extend([
            "### Multi-Seed Synthetic Depth Benchmark",
            "",
            f"- Seeds: {synthetic_metrics.get('seeds')}",
            f"- Samples per scene: {synthetic_metrics.get('samples_per_scene')}",
            f"- Embedding-risk ROC-AUC: {fmt_metric(synthetic_metrics.get('embedding_risk_roc_auc_mean'))} +/- {fmt_metric(synthetic_metrics.get('embedding_risk_roc_auc_std'))}",
            f"- Coverage-risk AUC: {fmt_metric(synthetic_metrics.get('coverage_risk_auc_mean'))} +/- {fmt_metric(synthetic_metrics.get('coverage_risk_auc_std'))}",
            "",
        ])
    else:
        lines.extend([
            "### Multi-Seed Synthetic Depth Benchmark",
            "",
            "- Not run yet in this checkout.",
            "",
        ])

    if real_depth_metrics:
        lines.extend([
            "### Real-Depth Corruption Benchmark",
            "",
            f"- Source files: {real_depth_metrics.get('source_files')}",
            f"- Samples: {real_depth_metrics.get('samples')}",
            f"- Embedding-risk ROC-AUC: {fmt_metric(real_depth_metrics.get('embedding_risk_roc_auc_for_corruption_detection'))}",
            f"- Scene-conditioned ROC-AUC: {fmt_metric(real_depth_metrics.get('scene_conditioned_roc_auc_for_corruption_detection'))}",
            f"- Coverage-risk AUC: {fmt_metric(real_depth_metrics.get('coverage_risk_auc'))}",
            "",
        ])
    else:
        lines.extend([
            "### Real-Depth Corruption Benchmark",
            "",
            "- Ready to run once a public RGB-D/depth subset is prepared.",
            "",
        ])

    if temporal_metrics:
        lines.extend([
            "### TUM RGB-D Temporal Reliability Benchmark",
            "",
            f"- Source files: {temporal_metrics.get('source_files')}",
            f"- Samples: {temporal_metrics.get('samples')}",
            f"- Temporal window: +/- {temporal_metrics.get('window')} frames",
            f"- Temporal excess ROC-AUC: {fmt_metric(temporal_metrics.get('temporal_excess_roc_auc_for_corruption_detection'))}",
            f"- Temporal local-distance ROC-AUC: {fmt_metric(temporal_metrics.get('temporal_local_roc_auc_for_corruption_detection'))}",
            f"- Source-paired ROC-AUC: {fmt_metric(temporal_metrics.get('source_paired_roc_auc_for_corruption_detection'))}",
            "",
        ])
    else:
        lines.extend([
            "### TUM RGB-D Temporal Reliability Benchmark",
            "",
            "- Not run yet.",
            "",
        ])

    if pose_metrics:
        global_trans = pose_metrics.get(
            "global_spearman_translation_vs_embedding_shift",
            pose_metrics.get("spearman_translation_vs_embedding_shift"),
        )
        global_rot = pose_metrics.get(
            "global_spearman_rotation_vs_embedding_shift",
            pose_metrics.get("spearman_rotation_vs_embedding_shift"),
        )
        grid_trans = pose_metrics.get("grid_spearman_translation_vs_embedding_shift")
        grid_rot = pose_metrics.get("grid_spearman_rotation_vs_embedding_shift")
        lines.extend([
            "### TUM RGB-D Pose-Aware Embedding Shift Analysis",
            "",
            f"- Adjacent frame pairs: {pose_metrics.get('pairs')}",
            f"- Global Spearman translation vs embedding shift: {fmt_metric(global_trans)}",
            f"- Global Spearman rotation vs embedding shift: {fmt_metric(global_rot)}",
            f"- Grid Spearman translation vs embedding shift: {fmt_metric(grid_trans)}",
            f"- Grid Spearman rotation vs embedding shift: {fmt_metric(grid_rot)}",
            "- Interpretation: local grid descriptors improve rotation sensitivity over global statistics, but neither descriptor is sufficient as a full SLAM-aware representation.",
            "",
        ])
    else:
        lines.extend([
            "### TUM RGB-D Pose-Aware Embedding Shift Analysis",
            "",
            "- Not run yet.",
            "",
        ])

    if pca_descriptor_metrics:
        lines.extend([
            "### TUM RGB-D Learned PCA Depth Descriptor Baseline",
            "",
            f"- Adjacent frame pairs: {pca_descriptor_metrics.get('pairs')}",
            f"- PCA components: {pca_descriptor_metrics.get('components')}",
            f"- Explained variance ratio: {fmt_metric(pca_descriptor_metrics.get('explained_variance_ratio_sum'))}",
            f"- Spearman translation vs embedding shift: {fmt_metric(pca_descriptor_metrics.get('spearman_translation_vs_embedding_shift'))}",
            f"- Spearman rotation vs embedding shift: {fmt_metric(pca_descriptor_metrics.get('spearman_rotation_vs_embedding_shift'))}",
            "- Interpretation: a lightweight learned depth descriptor tracks camera rotation better than hand-crafted global/grid descriptors.",
            "",
        ])
    else:
        lines.extend([
            "### TUM RGB-D Learned PCA Depth Descriptor Baseline",
            "",
            "- Not run yet.",
            "",
        ])

    if runtime_monitor_metrics:
        state_counts = runtime_monitor_metrics.get("state_counts", {})
        property_checks = runtime_monitor_metrics.get("property_checks", {})
        lines.extend([
            "### Runtime Assurance Monitor",
            "",
            f"- Samples: {runtime_monitor_metrics.get('samples')}",
            f"- NORMAL: {state_counts.get('NORMAL')}",
            f"- SUSPECT: {state_counts.get('SUSPECT')}",
            f"- RECOVER: {state_counts.get('RECOVER')}",
            f"- HUMAN_REVIEW: {state_counts.get('HUMAN_REVIEW')}",
            f"- Safety-property violations: {property_checks.get('violations')}",
            "- Interpretation: reliability scores are converted into auditable autonomy states.",
            "",
        ])
    else:
        lines.extend([
            "### Runtime Assurance Monitor",
            "",
            "- Not run yet.",
            "",
        ])

    if calibration_metrics:
        lines.extend([
            "### Reliability Score Calibration",
            "",
            f"- Samples: {calibration_metrics.get('samples')}",
            f"- ROC-AUC: {fmt_metric(calibration_metrics.get('roc_auc'))}",
            f"- Average precision: {fmt_metric(calibration_metrics.get('average_precision'))}",
            f"- ECE-style gap: {fmt_metric(calibration_metrics.get('ece'))}",
            f"- Risk at full coverage: {fmt_metric(calibration_metrics.get('risk_at_full_coverage'))}",
            f"- Risk at 80% coverage: {fmt_metric(calibration_metrics.get('risk_at_80pct_coverage'))}",
            f"- Risk at 50% coverage: {fmt_metric(calibration_metrics.get('risk_at_50pct_coverage'))}",
            "- Interpretation: the score ranks corruptions well, but raw normalized scores are not probability-calibrated.",
            "",
        ])
    else:
        lines.extend([
            "### Reliability Score Calibration",
            "",
            "- Not run yet.",
            "",
        ])

    if trajectory_metrics:
        lines.extend([
            "### Trajectory Residual Reliability Demo",
            "",
            f"- Samples: {trajectory_metrics.get('samples')}",
            f"- Failure types: {', '.join(trajectory_metrics.get('failure_types', []))}",
            f"- ROC-AUC: {fmt_metric(trajectory_metrics.get('roc_auc'))}",
            f"- Average precision: {fmt_metric(trajectory_metrics.get('average_precision'))}",
            "- Interpretation: planned-vs-observed trajectory residuals provide a control-facing action-outcome reliability score.",
            "",
        ])
    else:
        lines.extend([
            "### Trajectory Residual Reliability Demo",
            "",
            "- Not run yet.",
            "",
        ])

    if trajectory_monitor_metrics:
        state_counts = trajectory_monitor_metrics.get("state_counts", {})
        lines.extend([
            "### Trajectory Runtime Monitor",
            "",
            f"- Samples: {trajectory_monitor_metrics.get('samples')}",
            f"- NORMAL: {state_counts.get('NORMAL')}",
            f"- SUSPECT: {state_counts.get('SUSPECT')}",
            f"- RECOVER: {state_counts.get('RECOVER')}",
            f"- HUMAN_REVIEW: {state_counts.get('HUMAN_REVIEW')}",
            "",
        ])

    if video_metrics:
        lines.extend([
            "### Video Sequential Perception Benchmark",
            "",
            f"- Best validation accuracy: {fmt_metric(video_metrics.get('best_val_acc_percent'))}%",
            "- Validation embedding tables, PCA plots, and class-wise reports are produced by the training pipeline.",
            "",
        ])
    else:
        lines.extend([
            "### Video Sequential Perception Benchmark",
            "",
            "- Code is implemented; run with real video paths to produce metrics.",
            "",
        ])

    lines.extend([
        "## Supervisor Fit",
        "",
        "Detailed supervisor-by-supervisor notes are in `docs/hk_supervisor_alignment.md`.",
        "",
        "| Supervisor direction | Matching evidence |",
        "|---|---|",
        "| Trustworthy ML / calibration / robustness | Embedding risk, uncertainty scores, selective prediction, multi-seed reporting |",
        "| Human-robot collaboration / industrial AI | Action recognition reliability and 3D perception screening around tools/workcells |",
        "| Reliable 3D scene understanding | Depth-to-point-cloud embeddings and corruption detection |",
        "| Embodied navigation / autonomous systems | Perception risk as a trigger for abstention, fallback, or human review |",
        "| Medical or clinical trustworthy AI | ECG reliability project plus transferable reliability methodology |",
        "",
        "## What Is Shown",
        "",
        "- The project has working code for video embeddings and 3D geometry embeddings.",
        "- The 3D module can detect controlled perception corruptions using embedding-distance risk.",
        "- TUM RGB-D results show that naive global/local distance can fail under camera motion, motivating temporal excess scoring.",
        "- Pose-aware analysis shows a limitation: hand-crafted descriptors only weakly track frame-to-frame camera motion, though local grids improve rotation sensitivity.",
        "- A lightweight learned PCA depth descriptor improves pose-motion sensitivity, especially for rotation.",
        "- Runtime monitoring converts reliability scores into NORMAL/SUSPECT/RECOVER/HUMAN_REVIEW decisions with a simple property check.",
        "- Calibration analysis separates risk ranking from probability calibration.",
        "- Trajectory residual analysis extends the project from perception reliability to action-outcome reliability.",
        "- The workflow records data preparation and supports mean/std reporting across seeds.",
        "",
        "## What Remains Unproven",
        "",
        "- These results are not closed-loop robot validation.",
        "- Synthetic or controlled corruptions do not replace dataset-native failure labels.",
        "- Public-data results should be added before making strong claims about real-world robot perception.",
        "",
        "## Next Experiment",
        "",
        "Replace synthetic trajectory residuals with robot logs, surgical-tool tracking, or simulator rollouts; then evaluate monitor decisions against downstream task failures.",
        "",
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Build a supervisor-facing evidence pack.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output", default="docs/application_evidence_pack.md")
    return parser.parse_args()


def main():
    args = parse_args()
    path = build_evidence_pack(args.output, args.project_root)
    print(f"Evidence pack written to: {path}")


if __name__ == "__main__":
    main()
