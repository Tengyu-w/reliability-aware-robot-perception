"""Run local temporal-reference reliability benchmark on ordered depth frames."""

import argparse

from robot_3d_reliability import run_temporal_depth_reliability_benchmark


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark depth reliability using local temporal clean references."
    )
    parser.add_argument("--depth-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/temporal_depth_reliability")
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--window", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    metrics = run_temporal_depth_reliability_benchmark(
        depth_dir=args.depth_dir,
        output_dir=args.output_dir,
        depth_scale=args.depth_scale,
        seed=args.seed,
        max_files=args.max_files,
        window=args.window,
    )
    print("Temporal depth reliability benchmark completed.")
    print(f"Source depth files: {metrics['source_files']}")
    print(f"Window: {metrics['window']}")
    print(f"Outputs written to: {args.output_dir}")
    print(
        "Temporal excess ROC-AUC: "
        f"{metrics['temporal_excess_roc_auc_for_corruption_detection']:.3f}"
    )


if __name__ == "__main__":
    main()
