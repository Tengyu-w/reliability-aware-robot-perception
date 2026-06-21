"""Run the synthetic 3D perception reliability evidence demo."""

import argparse

from robot_3d_reliability import run_synthetic_3d_reliability_demo


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run synthetic robot 3D perception reliability diagnostics."
    )
    parser.add_argument("--output-dir", default="outputs/robot_3d_demo")
    parser.add_argument("--samples-per-scene", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    metrics = run_synthetic_3d_reliability_demo(
        output_dir=args.output_dir,
        samples_per_scene=args.samples_per_scene,
        seed=args.seed,
    )
    print("Synthetic robot 3D reliability demo completed.")
    print(f"Outputs written to: {args.output_dir}")
    print(
        "Embedding-risk ROC-AUC for corruption detection: "
        f"{metrics['embedding_risk_roc_auc_for_corruption_detection']:.3f}"
    )


if __name__ == "__main__":
    main()
