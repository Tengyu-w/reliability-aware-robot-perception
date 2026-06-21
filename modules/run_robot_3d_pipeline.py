"""Run a prepared depth dataset through the 3D reliability workflow."""

import argparse
import os

from robot_3d_reliability import (
    analyze_real_depth_directory,
    run_real_depth_corruption_benchmark,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run real-depth profiling and corruption reliability benchmark."
    )
    parser.add_argument("--depth-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/robot_3d_pipeline")
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-files", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    profile_dir = os.path.join(args.output_dir, "profile")
    benchmark_dir = os.path.join(args.output_dir, "corruption_benchmark")

    profile_metrics = analyze_real_depth_directory(
        depth_dir=args.depth_dir,
        output_dir=profile_dir,
        depth_scale=args.depth_scale,
    )
    benchmark_metrics = run_real_depth_corruption_benchmark(
        depth_dir=args.depth_dir,
        output_dir=benchmark_dir,
        depth_scale=args.depth_scale,
        seed=args.seed,
        max_files=args.max_files,
    )

    print("Robot 3D reliability pipeline completed.")
    print(f"Depth samples profiled: {profile_metrics['samples']}")
    print(
        "Corruption benchmark ROC-AUC: "
        f"{benchmark_metrics['embedding_risk_roc_auc_for_corruption_detection']:.3f}"
    )
    print(f"Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
