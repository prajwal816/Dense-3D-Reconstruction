#!/usr/bin/env python3
"""
CLI entry point for the Dense 3D Reconstruction Pipeline.

Usage:
    python scripts/run_pipeline.py --config configs/default.yaml --images data/images
    python scripts/run_pipeline.py --help
"""

import argparse
import json
import os
import sys

# Allow running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pipeline import ReconstructionPipeline
from src.utils.logger import get_logger

_logger = get_logger("dense3d.cli")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dense Multi-View 3D Reconstruction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline with default config
  python scripts/run_pipeline.py --config configs/default.yaml --images data/images

  # Use ORB features and skip MVS
  python scripts/run_pipeline.py --config configs/default.yaml \\
      --images data/images --feature-type orb --skip-mvs

  # SuperPoint + SuperGlue deep matching
  python scripts/run_pipeline.py --config configs/default.yaml \\
      --images data/images --feature-type superpoint --matcher-type superglue

  # Custom output directory, no visualisation
  python scripts/run_pipeline.py --config configs/default.yaml \\
      --images data/images --output outputs/run_01 --no-viz
""",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--images",
        type=str,
        default=None,
        help="Path to image directory (overrides config)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (overrides config)",
    )

    # Feature options
    feat_group = parser.add_argument_group("Feature Extraction")
    feat_group.add_argument(
        "--feature-type",
        type=str,
        choices=["sift", "orb", "superpoint"],
        default=None,
        help="Feature extractor type (overrides config)",
    )
    feat_group.add_argument(
        "--max-keypoints",
        type=int,
        default=None,
        help="Maximum keypoints per image",
    )

    # Matching options
    match_group = parser.add_argument_group("Feature Matching")
    match_group.add_argument(
        "--matcher-type",
        type=str,
        choices=["bf", "flann", "superglue"],
        default=None,
        help="Matcher type (overrides config)",
    )
    match_group.add_argument(
        "--ratio-test",
        type=float,
        default=None,
        help="Lowe's ratio test threshold",
    )

    # Pipeline control
    ctrl_group = parser.add_argument_group("Pipeline Control")
    ctrl_group.add_argument(
        "--skip-mvs",
        action="store_true",
        help="Skip Multi-View Stereo stage",
    )
    ctrl_group.add_argument(
        "--skip-mesh",
        action="store_true",
        help="Skip mesh generation",
    )
    ctrl_group.add_argument(
        "--no-viz",
        action="store_true",
        help="Disable visualisation outputs",
    )

    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Logging verbosity",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Build config overrides from CLI args
    overrides = {}
    if args.images:
        overrides.setdefault("paths", {})["images"] = args.images
    if args.output:
        overrides.setdefault("paths", {})["output"] = args.output
    if args.feature_type:
        overrides.setdefault("features", {})["type"] = args.feature_type
    if args.max_keypoints:
        overrides.setdefault("features", {})["max_keypoints"] = args.max_keypoints
    if args.matcher_type:
        overrides.setdefault("matching", {})["type"] = args.matcher_type
    if args.ratio_test:
        overrides.setdefault("matching", {})["ratio_test"] = args.ratio_test
    if args.log_level:
        overrides.setdefault("logging", {})["level"] = args.log_level

    # Create and run pipeline
    try:
        pipeline = ReconstructionPipeline(args.config, overrides=overrides or None)
        metrics = pipeline.run(
            image_dir=args.images,
            skip_mvs=args.skip_mvs,
            skip_mesh=args.skip_mesh,
            visualize=not args.no_viz,
        )

        # Save metrics
        metrics_path = os.path.join(
            pipeline.output_dir, "metrics.json"
        )
        # Convert numpy types for JSON serialisation
        def _convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        import numpy as np

        serialisable = json.loads(json.dumps(metrics, default=_convert))
        with open(metrics_path, "w") as f:
            json.dump(serialisable, f, indent=2)
        _logger.info("Metrics saved → %s", metrics_path)

    except FileNotFoundError as exc:
        _logger.error("File not found: %s", exc)
        sys.exit(1)
    except RuntimeError as exc:
        _logger.error("Pipeline error: %s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        _logger.info("Pipeline interrupted by user")
        sys.exit(130)


if __name__ == "__main__":
    main()
