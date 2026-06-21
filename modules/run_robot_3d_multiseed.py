"""Run multi-seed synthetic 3D perception reliability benchmark."""

import argparse

from robot_3d_reliability import run_multi_seed_synthetic_benchmark, write_robot_3d_report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run multi-seed synthetic robot 3D reliability diagnostics."
    )
    parser.add_argument("--output-dir", default="outputs/robot_3d_multiseed")
    parser.add_argument("--samples-per-scene", type=int, default=24)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    return parser.parse_args()


def main():
    args = parse_args()
    metrics = run_multi_seed_synthetic_benchmark(
        output_dir=args.output_dir,
        seeds=args.seeds,
        samples_per_scene=args.samples_per_scene,
    )
    report_path = write_robot_3d_report(args.output_dir, metrics)
    print("Multi-seed robot 3D reliability benchmark completed.")
    print(f"Outputs written to: {args.output_dir}")
    print(f"Report written to: {report_path}")
    print(
        "Embedding-risk ROC-AUC: "
        f"{metrics['embedding_risk_roc_auc_mean']:.3f} "
        f"+/- {metrics['embedding_risk_roc_auc_std']:.3f}"
    )


if __name__ == "__main__":
    main()
