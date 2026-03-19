#!/usr/bin/env python3
"""
Download public multi-view datasets for benchmarking.

Supported datasets:
  - south-building  : COLMAP example (128 images, ~40 MB)
  - gerrard-hall    : COLMAP example (100 images, ~30 MB)
  - fountain        : ETH3D low-res (11 images, ~8 MB)

Usage:
    python scripts/download_dataset.py --dataset south-building --output data/
    python scripts/download_dataset.py --list
"""

import argparse
import os
import sys
import shutil
import tarfile
import zipfile
import urllib.request
import urllib.error

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils.io_utils import ensure_dir
from src.utils.logger import get_logger

_logger = get_logger("dense3d.download")


# ═════════════════════════════════════════════════════════════════════════════
# Dataset Registry
# ═════════════════════════════════════════════════════════════════════════════

DATASETS = {
    "south-building": {
        "url": "https://demuc.de/colmap/datasets/south-building.zip",
        "description": "COLMAP South Building — 128 outdoor images (~40 MB)",
        "images_subdir": "south-building/images",
        "format": "zip",
    },
    "gerrard-hall": {
        "url": "https://demuc.de/colmap/datasets/gerrard-hall.zip",
        "description": "COLMAP Gerrard Hall — 100 outdoor images (~30 MB)",
        "images_subdir": "gerrard-hall/images",
        "format": "zip",
    },
    "fountain": {
        "url": "https://www.eth3d.net/data/fountain_undistorted.7z",
        "description": "ETH3D Fountain — 11 high-quality images (~8 MB)",
        "images_subdir": "fountain_undistorted/images",
        "format": "7z",
    },
    "person-hall": {
        "url": "https://demuc.de/colmap/datasets/person-hall.zip",
        "description": "COLMAP Person Hall — 330 indoor images (~70 MB)",
        "images_subdir": "person-hall/images",
        "format": "zip",
    },
}


# ═════════════════════════════════════════════════════════════════════════════
# Download & Extract
# ═════════════════════════════════════════════════════════════════════════════

def _download_file(url: str, dest: str) -> None:
    """Download a file with progress reporting."""
    _logger.info("Downloading %s …", url)

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Dense3D-Pipeline/1.0"})
        with urllib.request.urlopen(req, timeout=120) as response:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 256  # 256 KB

            with open(dest, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded / total * 100
                        mb = downloaded / (1024 * 1024)
                        print(f"\r  Progress: {pct:.1f}% ({mb:.1f} MB)", end="", flush=True)

            print()  # newline
            _logger.info("Downloaded → %s (%.1f MB)", dest, downloaded / (1024 * 1024))

    except urllib.error.URLError as exc:
        _logger.error("Download failed: %s", exc)
        raise


def _extract_archive(archive_path: str, dest_dir: str, fmt: str) -> None:
    """Extract a downloaded archive."""
    _logger.info("Extracting %s …", archive_path)

    if fmt == "zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(dest_dir)
    elif fmt == "tar" or fmt == "tar.gz":
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(dest_dir)
    elif fmt == "7z":
        _logger.warning(
            "7z extraction requires the `py7zr` package. "
            "Install with: pip install py7zr"
        )
        try:
            import py7zr
            with py7zr.SevenZipFile(archive_path, mode="r") as z:
                z.extractall(path=dest_dir)
        except ImportError:
            _logger.error("py7zr not installed — cannot extract .7z files")
            raise
    else:
        _logger.error("Unknown archive format: %s", fmt)
        raise ValueError(f"Unsupported format: {fmt}")

    _logger.info("Extracted to %s", dest_dir)


def download_dataset(name: str, output_dir: str) -> str:
    """Download and extract a dataset.

    Returns:
        Path to the extracted images directory.
    """
    if name not in DATASETS:
        available = ", ".join(DATASETS.keys())
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}")

    ds = DATASETS[name]
    ensure_dir(output_dir)

    # Download
    ext = os.path.splitext(ds["url"])[1]
    archive_name = f"{name}{ext}"
    archive_path = os.path.join(output_dir, archive_name)

    if not os.path.isfile(archive_path):
        _download_file(ds["url"], archive_path)
    else:
        _logger.info("Archive already exists: %s", archive_path)

    # Extract
    _extract_archive(archive_path, output_dir, ds["format"])

    # Locate images
    images_dir = os.path.join(output_dir, ds["images_subdir"])
    if os.path.isdir(images_dir):
        n_images = len([
            f for f in os.listdir(images_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        _logger.info("✓ Dataset '%s': %d images in %s", name, n_images, images_dir)
    else:
        _logger.warning("Expected images dir not found: %s", images_dir)
        images_dir = output_dir

    return images_dir


def list_datasets() -> None:
    """Print available datasets."""
    print("\n  Available Datasets")
    print("  " + "=" * 60)
    for name, ds in DATASETS.items():
        print(f"  {name:<18s}  {ds['description']}")
    print("  " + "=" * 60)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Download public multi-view datasets"
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Dataset name to download",
    )
    parser.add_argument(
        "--output", type=str, default="data",
        help="Output directory (default: data/)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available datasets",
    )
    args = parser.parse_args()

    if args.list or args.dataset is None:
        list_datasets()
        if args.dataset is None:
            print("  Use --dataset <name> to download\n")
        return

    download_dataset(args.dataset, args.output)


if __name__ == "__main__":
    main()
