#!/usr/bin/env python3
"""
Benchmark script: compare SIFT vs ORB vs SuperPoint on feature quality
and processing time.

Usage:
    python scripts/benchmark.py --images data/images --output benchmarks/
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.features.factory import create_extractor
from src.matching.factory import create_matcher
from src.utils.io_utils import load_images_from_dir, ensure_dir
from src.utils.logger import get_logger
from src.utils.timer import Timer

_logger = get_logger("dense3d.benchmark")


def benchmark_extractor(
    name: str,
    images: list,
) -> Dict[str, Any]:
    """Benchmark a single feature extractor."""
    cfg: Dict[str, Any] = {"type": name}
    extractor = create_extractor(cfg)

    t0 = time.perf_counter()
    features = []
    for img_name, img in images:
        feat = extractor.extract(img)
        feat.image_name = img_name
        features.append(feat)
    elapsed = time.perf_counter() - t0

    kps = [f.num_keypoints for f in features]

    return {
        "extractor": name,
        "total_keypoints": sum(kps),
        "avg_keypoints": float(np.mean(kps)),
        "extraction_time_s": elapsed,
        "time_per_image_s": elapsed / max(len(images), 1),
        "features": features,
    }


def benchmark_matching(
    matcher_type: str,
    feature_type: str,
    features: list,
) -> Dict[str, Any]:
    """Benchmark matching across all image pairs."""
    cfg: Dict[str, Any] = {"type": matcher_type, "feature_type": feature_type}
    matcher = create_matcher(cfg)

    t0 = time.perf_counter()
    inlier_ratios = []
    match_counts = []

    n = len(features)
    for i in range(min(n, 5)):  # Limit pairs for speed
        for j in range(i + 1, min(n, 5)):
            result = matcher.match(
                features[i].keypoints,
                features[i].descriptors,
                features[j].keypoints,
                features[j].descriptors,
            )
            inlier_ratios.append(result.inlier_ratio)
            match_counts.append(result.num_matches)

    elapsed = time.perf_counter() - t0

    return {
        "matcher": matcher_type,
        "feature": feature_type,
        "num_pairs": len(inlier_ratios),
        "avg_inlier_ratio": float(np.mean(inlier_ratios)) if inlier_ratios else 0.0,
        "avg_matches": float(np.mean(match_counts)) if match_counts else 0.0,
        "matching_time_s": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature Benchmark")
    parser.add_argument("--images", type=str, required=True, help="Image directory")
    parser.add_argument("--output", type=str, default="benchmarks", help="Output directory")
    parser.add_argument("--max-images", type=int, default=10, help="Max images to use")
    args = parser.parse_args()

    ensure_dir(args.output)

    _logger.info("Loading images from %s …", args.images)
    images = load_images_from_dir(args.images)
    if not images:
        _logger.error("No images found")
        sys.exit(1)
    images = images[: args.max_images]
    _logger.info("Using %d images for benchmarking", len(images))

    results: Dict[str, Any] = {"num_images": len(images), "extractors": [], "matchers": []}

    # ── Benchmark feature extractors ─────────────────────────────────────
    extractors_to_test = ["sift", "orb", "superpoint"]
    feature_cache: Dict[str, list] = {}

    for ext_name in extractors_to_test:
        _logger.info("Benchmarking extractor: %s", ext_name)
        try:
            r = benchmark_extractor(ext_name, images)
            feature_cache[ext_name] = r.pop("features")
            results["extractors"].append(r)
            _logger.info(
                "  %s: %d kp, %.3f s total",
                ext_name,
                r["total_keypoints"],
                r["extraction_time_s"],
            )
        except Exception as exc:
            _logger.warning("  %s failed: %s", ext_name, exc)

    # ── Benchmark matchers ───────────────────────────────────────────────
    matcher_configs = [
        ("bf", "sift"),
        ("bf", "orb"),
        ("flann", "sift"),
        ("superglue", "superpoint"),
    ]

    for matcher_type, feat_type in matcher_configs:
        if feat_type not in feature_cache:
            continue
        _logger.info("Benchmarking matcher: %s + %s", matcher_type, feat_type)
        try:
            r = benchmark_matching(matcher_type, feat_type, feature_cache[feat_type])
            results["matchers"].append(r)
            _logger.info(
                "  %s+%s: avg inlier=%.2f%%, %.3f s",
                matcher_type,
                feat_type,
                r["avg_inlier_ratio"] * 100,
                r["matching_time_s"],
            )
        except Exception as exc:
            _logger.warning("  %s+%s failed: %s", matcher_type, feat_type, exc)

    # ── Save results ─────────────────────────────────────────────────────
    out_path = os.path.join(args.output, "benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    _logger.info("Benchmark results saved → %s", out_path)

    # ── Print comparison table ───────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  FEATURE EXTRACTOR COMPARISON")
    print("=" * 72)
    print(f"  {'Extractor':<14} {'Keypoints':>10} {'Time (s)':>10} {'KP/image':>10}")
    print("-" * 72)
    for r in results["extractors"]:
        print(
            f"  {r['extractor']:<14} {r['total_keypoints']:>10} "
            f"{r['extraction_time_s']:>10.3f} {r['avg_keypoints']:>10.0f}"
        )

    print("\n" + "=" * 72)
    print("  MATCHER COMPARISON")
    print("=" * 72)
    print(f"  {'Matcher':<12} {'Feature':<12} {'Inlier %':>10} {'Avg Matches':>12} {'Time (s)':>10}")
    print("-" * 72)
    for r in results["matchers"]:
        print(
            f"  {r['matcher']:<12} {r['feature']:<12} "
            f"{r['avg_inlier_ratio']*100:>9.1f}% {r['avg_matches']:>12.0f} "
            f"{r['matching_time_s']:>10.3f}"
        )

    print("=" * 72)


if __name__ == "__main__":
    main()
