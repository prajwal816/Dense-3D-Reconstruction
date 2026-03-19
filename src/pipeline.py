"""
End-to-end Dense 3D Reconstruction Pipeline orchestrator.

images → features → matches → SfM → MVS → mesh
"""

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .utils.logger import get_logger
from .utils.config import Config, load_config
from .utils.timer import Timer
from .utils.io_utils import ensure_dir, load_images_from_dir

from .features.factory import create_extractor
from .features.base import FeatureData
from .matching.factory import create_matcher
from .matching.base import MatchResult
from .sfm.colmap_sfm import COLMAPSfM
from .sfm.sparse_reconstruction import SparseReconstruction
from .mvs.colmap_mvs import COLMAPMVS

_logger = get_logger("dense3d.pipeline")


class ReconstructionPipeline:
    """Full SfM + MVS reconstruction pipeline.

    Config-driven: all parameters come from a :class:`Config` object.

    Example::

        from src.pipeline import ReconstructionPipeline
        pipe = ReconstructionPipeline("configs/default.yaml")
        pipe.run()
    """

    def __init__(self, config_path: str, overrides: Optional[Dict[str, Any]] = None) -> None:
        self.cfg = load_config(config_path, overrides)
        self.timer = Timer()
        self.metrics: Dict[str, Any] = {}

        # Set up output directories
        self.output_dir = os.path.abspath(self.cfg.get("paths", {}).get("output", "outputs"))
        ensure_dir(self.output_dir)

        # Initialise logger with config
        log_cfg = self.cfg.get("logging", {})
        self.logger = get_logger(
            "dense3d",
            level=log_cfg.get("level", "INFO"),
            log_file=log_cfg.get("log_file"),
            colored=log_cfg.get("colored", True),
        )

        self.logger.info("Pipeline initialised (output → %s)", self.output_dir)

    def run(
        self,
        image_dir: Optional[str] = None,
        skip_mvs: bool = False,
        skip_mesh: bool = False,
        visualize: bool = True,
    ) -> Dict[str, Any]:
        """Execute the full pipeline.

        Args:
            image_dir: Override the image directory from config.
            skip_mvs: Skip the dense MVS stage.
            skip_mesh: Skip mesh generation.
            visualize: Run visualisation at the end.

        Returns:
            Dictionary of collected metrics.
        """
        img_dir = image_dir or self.cfg.paths.get("images", "data/images")
        img_dir = os.path.abspath(img_dir)

        self.logger.info("=" * 60)
        self.logger.info("  Dense 3D Reconstruction Pipeline")
        self.logger.info("=" * 60)
        self.logger.info("  Images  : %s", img_dir)
        self.logger.info("  Output  : %s", self.output_dir)
        self.logger.info("  Features: %s", self.cfg.features.get("type", "sift"))
        self.logger.info("  Matcher : %s", self.cfg.matching.get("type", "bf"))
        self.logger.info("=" * 60)

        # ── 1. Load images ───────────────────────────────────────────────
        with self.timer("load_images"):
            images = load_images_from_dir(img_dir)
            if not images:
                raise RuntimeError(f"No images found in {img_dir}")
            self.logger.info("Loaded %d images", len(images))

        # ── 2. Feature extraction ────────────────────────────────────────
        with self.timer("feature_extraction"):
            extractor = create_extractor(dict(self.cfg.features))
            features: List[Tuple[str, FeatureData]] = []
            for name, img in images:
                feat = extractor.extract(img)
                feat.image_name = name
                features.append((name, feat))
                self.logger.debug("  %s: %d keypoints", name, feat.num_keypoints)

            total_kp = sum(f.num_keypoints for _, f in features)
            self.metrics["total_keypoints"] = total_kp
            self.logger.info("Extracted %d keypoints across %d images", total_kp, len(features))

        # ── 3. Feature matching ──────────────────────────────────────────
        with self.timer("feature_matching"):
            matcher_cfg = dict(self.cfg.matching)
            matcher_cfg["feature_type"] = self.cfg.features.get("type", "sift")
            matcher = create_matcher(matcher_cfg)

            match_results: List[Tuple[str, str, MatchResult]] = []
            n_images = len(features)
            total_inlier_ratio = 0.0
            n_pairs = 0

            for i in range(n_images):
                for j in range(i + 1, n_images):
                    name_i, feat_i = features[i]
                    name_j, feat_j = features[j]

                    result = matcher.match(
                        feat_i.keypoints,
                        feat_i.descriptors,
                        feat_j.keypoints,
                        feat_j.descriptors,
                    )
                    match_results.append((name_i, name_j, result))
                    total_inlier_ratio += result.inlier_ratio
                    n_pairs += 1

            avg_inlier = total_inlier_ratio / max(n_pairs, 1)
            self.metrics["num_pairs"] = n_pairs
            self.metrics["avg_inlier_ratio"] = avg_inlier
            self.logger.info(
                "Matched %d pairs — avg inlier ratio: %.2f%%",
                n_pairs,
                avg_inlier * 100,
            )

        # ── 4. Visualise matches (first pair) ───────────────────────────
        if visualize and match_results:
            try:
                from .visualization.match_visualizer import draw_matches

                name_i, name_j, mr = match_results[0]
                img_i = dict(images)[name_i]
                img_j = dict(images)[name_j]
                match_vis_path = os.path.join(self.output_dir, "matches_preview.png")
                draw_matches(
                    img_i,
                    features[0][1].keypoints,
                    img_j,
                    features[1][1].keypoints,
                    mr.matches_idx,
                    mr.inlier_mask,
                    output_path=match_vis_path,
                )
            except Exception as exc:
                self.logger.warning("Match visualisation failed: %s", exc)

        # ── 5. Structure from Motion ─────────────────────────────────────
        sparse_recon = SparseReconstruction()
        with self.timer("sfm"):
            sfm_cfg = dict(self.cfg.get("sfm", {}))
            if sfm_cfg.get("use_colmap", True):
                sfm = COLMAPSfM(
                    colmap_binary=self.cfg.paths.get("colmap_binary", "colmap"),
                    workspace=os.path.join(self.output_dir, "colmap_ws"),
                    camera_model=sfm_cfg.get("camera_model", "SIMPLE_RADIAL"),
                    single_camera=sfm_cfg.get("single_camera", True),
                    cfg=sfm_cfg,
                )
                try:
                    sparse_recon = sfm.run(img_dir)
                    self.logger.info(sparse_recon.summary())
                    self.metrics["sfm_num_points"] = sparse_recon.num_points
                    self.metrics["sfm_num_poses"] = sparse_recon.num_poses
                except RuntimeError as exc:
                    self.logger.error("SfM failed: %s", exc)
                    self.logger.info(
                        "SfM requires COLMAP installed and in PATH. "
                        "Skipping SfM/MVS stages."
                    )
                    skip_mvs = True

        # ── 6. Multi-View Stereo ─────────────────────────────────────────
        dense_ply_path = ""
        if not skip_mvs and sparse_recon.num_points > 0:
            with self.timer("mvs"):
                mvs_cfg = dict(self.cfg.get("mvs", {}))
                if mvs_cfg.get("use_colmap", True):
                    mvs = COLMAPMVS(
                        colmap_binary=self.cfg.paths.get("colmap_binary", "colmap"),
                        workspace=os.path.join(self.output_dir, "colmap_ws"),
                        cfg=mvs_cfg,
                    )
                    try:
                        model_dir = sparse_recon.model_dir or os.path.join(
                            self.output_dir, "colmap_ws", "sparse", "0"
                        )
                        dense_ply_path = mvs.run(img_dir, model_dir)
                        self.metrics["dense_ply"] = dense_ply_path
                    except RuntimeError as exc:
                        self.logger.error("MVS failed: %s", exc)
        else:
            self.logger.info("Skipping MVS stage")

        # ── 7. Mesh generation ───────────────────────────────────────────
        if not skip_mesh and dense_ply_path and os.path.isfile(dense_ply_path):
            with self.timer("mesh_generation"):
                try:
                    from .reconstruction.point_cloud_processor import PointCloudProcessor
                    from .reconstruction.mesh_generator import MeshGenerator

                    recon_cfg = dict(self.cfg.get("reconstruction", {}))
                    processor = PointCloudProcessor(recon_cfg)
                    pcd = processor.load(dense_ply_path)
                    pcd = processor.process(pcd)

                    mesh_gen = MeshGenerator(
                        method=recon_cfg.get("mesh_method", "poisson"),
                        cfg=recon_cfg,
                    )
                    mesh_path = os.path.join(self.output_dir, "mesh.ply")
                    mesh_gen.generate(pcd, output_path=mesh_path)
                    self.metrics["mesh_path"] = mesh_path
                except Exception as exc:
                    self.logger.error("Mesh generation failed: %s", exc)
        else:
            self.logger.info("Skipping mesh generation")

        # ── 8. Visualise camera trajectory ───────────────────────────────
        if visualize and sparse_recon.num_poses > 0:
            try:
                from .visualization.camera_trajectory import plot_camera_trajectory

                traj_path = os.path.join(self.output_dir, "camera_trajectory.png")
                plot_camera_trajectory(
                    sparse_recon.poses,
                    sparse_recon.points3d if sparse_recon.num_points > 0 else None,
                    output_path=traj_path,
                )
            except Exception as exc:
                self.logger.warning("Trajectory plot failed: %s", exc)

        # ── Summary ──────────────────────────────────────────────────────
        self.metrics["timing"] = self.timer.records
        self.metrics["total_time"] = self.timer.total
        self.logger.info(self.timer.summary())

        # Reconstruction completeness metric
        if sparse_recon.num_poses > 0:
            completeness = sparse_recon.num_poses / len(images)
            self.metrics["reconstruction_completeness"] = completeness
            self.logger.info("Reconstruction completeness: %.1f%%", completeness * 100)

        self.logger.info("Pipeline complete ✓")
        return self.metrics
