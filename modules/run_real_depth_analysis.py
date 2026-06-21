"""Analyze real depth maps with geometry embeddings."""

import argparse

from robot_3d_reliability import analyze_real_depth_directory


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze a directory of depth maps.")
    parser.add_argument("--depth-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/real_depth_analysis")
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=1000.0,
        help="Scale factor for image depth values, e.g. 1000 for millimeters.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    metrics = analyze_real_depth_directory(
        depth_dir=args.depth_dir,
        output_dir=args.output_dir,
        depth_scale=args.depth_scale,
    )
    print("Real depth embedding analysis completed.")
    print(f"Samples: {metrics['samples']}")
    print(f"Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
