"""Run reliability benchmark on real depth maps with controlled corruptions."""

import argparse

from robot_3d_reliability import run_real_depth_corruption_benchmark


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark real depth-map reliability with controlled corruptions."
    )
    parser.add_argument("--depth-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/real_depth_corruption")
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-files", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    metrics = run_real_depth_corruption_benchmark(
        depth_dir=args.depth_dir,
        output_dir=args.output_dir,
        depth_scale=args.depth_scale,
        seed=args.seed,
        max_files=args.max_files,
    )
    print("Real depth corruption reliability benchmark completed.")
    print(f"Source depth files: {metrics['source_files']}")
    print(f"Outputs written to: {args.output_dir}")
    print(
        "Embedding-risk ROC-AUC: "
        f"{metrics['embedding_risk_roc_auc_for_corruption_detection']:.3f}"
    )


if __name__ == "__main__":
    main()
