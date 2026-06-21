"""Download and prepare a small TUM RGB-D sample sequence.

This is an optional helper. It downloads one official TUM RGB-D archive, extracts
it under a raw-data directory, then prepares the depth folder for the existing
3D reliability pipeline.
"""

import argparse
import os
import shutil
import ssl
import tarfile
import urllib.request
from pathlib import Path

from prepare_depth_dataset import prepare_depth_dataset


TUM_SEQUENCES = {
    "freiburg1_desk": {
        "url": "https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz",
        "archive": "rgbd_dataset_freiburg1_desk.tgz",
        "folder": "rgbd_dataset_freiburg1_desk",
        "depth_scale": 5000.0,
    },
    "freiburg1_room": {
        "url": "https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_room.tgz",
        "archive": "rgbd_dataset_freiburg1_room.tgz",
        "folder": "rgbd_dataset_freiburg1_room",
        "depth_scale": 5000.0,
    },
    "freiburg2_xyz": {
        "url": "https://vision.in.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_xyz.tgz",
        "archive": "rgbd_dataset_freiburg2_xyz.tgz",
        "folder": "rgbd_dataset_freiburg2_xyz",
        "depth_scale": 5000.0,
    },
}


def safe_extract_tar(tar, path):
    """Extract a tar archive while preventing path traversal."""
    target_root = Path(path).resolve()
    for member in tar.getmembers():
        member_path = (target_root / member.name).resolve()
        if not str(member_path).startswith(str(target_root)):
            raise ValueError(f"unsafe archive member: {member.name}")
    tar.extractall(target_root)


def download_file(url, destination, allow_insecure=False):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination
    print(f"Downloading {url}")
    try:
        if allow_insecure:
            context = ssl._create_unverified_context()
            with urllib.request.urlopen(url, context=context) as response:
                with open(destination, "wb") as f:
                    shutil.copyfileobj(response, f)
        else:
            urllib.request.urlretrieve(url, destination)
    except Exception:
        if destination.exists():
            destination.unlink()
        raise
    return destination


def download_and_prepare_tum(sequence, raw_dir, prepared_dir, max_files=None, seed=42, allow_insecure=False, order="random"):
    if sequence not in TUM_SEQUENCES:
        choices = ", ".join(sorted(TUM_SEQUENCES))
        raise ValueError(f"unknown sequence '{sequence}'. Choices: {choices}")

    spec = TUM_SEQUENCES[sequence]
    raw_dir = Path(raw_dir).resolve()
    prepared_dir = Path(prepared_dir).resolve()
    archive_path = raw_dir / spec["archive"]
    extracted_dir = raw_dir / spec["folder"]

    download_file(spec["url"], archive_path, allow_insecure=allow_insecure)
    if not extracted_dir.exists():
        with tarfile.open(archive_path, "r:gz") as tar:
            safe_extract_tar(tar, raw_dir)

    depth_dir = extracted_dir / "depth"
    if not depth_dir.exists():
        raise FileNotFoundError(f"depth folder not found after extraction: {depth_dir}")

    summary = prepare_depth_dataset(
        input_dir=depth_dir,
        output_dir=prepared_dir,
        max_files=max_files,
        seed=seed,
        copy_files=True,
        order=order,
    )
    summary["sequence"] = sequence
    summary["depth_scale"] = spec["depth_scale"]
    summary["official_url"] = spec["url"]
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Download and prepare a TUM RGB-D sample.")
    parser.add_argument("--sequence", default="freiburg1_desk", choices=sorted(TUM_SEQUENCES))
    parser.add_argument("--raw-dir", default="data/raw/tum_rgbd")
    parser.add_argument("--prepared-dir", default="data/prepared_depth/tum_rgbd_freiburg1_desk")
    parser.add_argument("--max-files", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--order", choices=["random", "sorted"], default="random")
    parser.add_argument(
        "--allow-insecure",
        action="store_true",
        help="Skip TLS certificate verification if the local CA store is broken.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    summary = download_and_prepare_tum(
        sequence=args.sequence,
        raw_dir=args.raw_dir,
        prepared_dir=args.prepared_dir,
        max_files=args.max_files,
        seed=args.seed,
        allow_insecure=args.allow_insecure,
        order=args.order,
    )
    print("TUM RGB-D sample prepared.")
    print(f"Sequence: {summary['sequence']}")
    print(f"Samples: {summary['samples']}")
    print(f"Depth scale: {summary['depth_scale']}")
    print(f"Prepared dir: {summary['output_dir']}")
    print(f"Manifest: {summary['manifest_path']}")


if __name__ == "__main__":
    main()
