"""
COLMAP Structure-from-Motion wrapper.

Provides a Python interface to drive COLMAP's feature extraction,
matching, and incremental mapper through its CLI.
"""

import os
import shutil
import subprocess
from typing import Any, Dict, List, Optional

import numpy as np

from .camera import CameraModel, CameraPose
from .sparse_reconstruction import SparseReconstruction
from ..utils.logger import get_logger
from ..utils.io_utils import ensure_dir

_logger = get_logger("dense3d.sfm.colmap")


class COLMAPSfM:
    """Wrapper around COLMAP's SfM pipeline.

    Args:
        colmap_binary: Path (or name) of the ``colmap`` executable.
        workspace: Root workspace directory for COLMAP artefacts.
        camera_model: COLMAP camera model name.
        single_camera: Assume all images come from the same camera.
        cfg: Additional SfM configuration from the YAML config.
    """

    def __init__(
        self,
        colmap_binary: str = "colmap",
        workspace: str = "outputs/colmap_ws",
        camera_model: str = "SIMPLE_RADIAL",
        single_camera: bool = True,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.colmap = colmap_binary
        self.workspace = os.path.abspath(workspace)
        self.camera_model = camera_model
        self.single_camera = single_camera
        self.cfg = cfg or {}

        self.database_path = os.path.join(self.workspace, "database.db")
        self.sparse_dir = os.path.join(self.workspace, "sparse")
        self.image_dir = ""  # set by run()

        _logger.info(
            "COLMAP SfM wrapper (binary=%s, cam=%s, single=%s)",
            colmap_binary,
            camera_model,
            single_camera,
        )

    # ── Convenience: check COLMAP is available ───────────────────────────
    def check_colmap(self) -> bool:
        """Return True if the COLMAP binary is reachable."""
        try:
            result = subprocess.run(
                [self.colmap, "help"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    # ── Full SfM run ─────────────────────────────────────────────────────
    def run(self, image_dir: str) -> SparseReconstruction:
        """Execute the complete COLMAP SfM pipeline.

        1. Feature extraction
        2. Exhaustive matching
        3. Incremental mapping

        Args:
            image_dir: Directory containing input images.

        Returns:
            :class:`SparseReconstruction` with cameras, poses, and 3D points.
        """
        self.image_dir = os.path.abspath(image_dir)
        ensure_dir(self.workspace)
        ensure_dir(self.sparse_dir)

        if not self.check_colmap():
            _logger.error(
                "COLMAP binary not found at '%s'. "
                "Please install COLMAP and add it to PATH.",
                self.colmap,
            )
            raise RuntimeError(f"COLMAP not found: {self.colmap}")

        self._extract_features()
        self._match_features()
        self._run_mapper()

        return self._load_reconstruction()

    # ── Step 1: Feature extraction ───────────────────────────────────────
    def _extract_features(self) -> None:
        _logger.info("COLMAP: extracting features …")
        cmd = [
            self.colmap, "feature_extractor",
            "--database_path", self.database_path,
            "--image_path", self.image_dir,
            "--ImageReader.camera_model", self.camera_model,
        ]
        if self.single_camera:
            cmd += ["--ImageReader.single_camera", "1"]

        self._run_cmd(cmd, "feature_extractor")

    # ── Step 2: Feature matching ─────────────────────────────────────────
    def _match_features(self) -> None:
        _logger.info("COLMAP: exhaustive matching …")
        cmd = [
            self.colmap, "exhaustive_matcher",
            "--database_path", self.database_path,
        ]
        self._run_cmd(cmd, "exhaustive_matcher")

    # ── Step 3: Incremental mapper ───────────────────────────────────────
    def _run_mapper(self) -> None:
        _logger.info("COLMAP: running incremental mapper …")
        cmd = [
            self.colmap, "mapper",
            "--database_path", self.database_path,
            "--image_path", self.image_dir,
            "--output_path", self.sparse_dir,
        ]

        mapper_cfg = self.cfg.get("mapper", {})
        if mapper_cfg.get("init_min_tri_angle"):
            cmd += [
                "--Mapper.init_min_tri_angle",
                str(mapper_cfg["init_min_tri_angle"]),
            ]
        if mapper_cfg.get("ba_refine_focal_length") is not None:
            cmd += [
                "--Mapper.ba_refine_focal_length",
                "1" if mapper_cfg["ba_refine_focal_length"] else "0",
            ]

        self._run_cmd(cmd, "mapper")

    # ── Load the resulting reconstruction ────────────────────────────────
    def _load_reconstruction(self) -> SparseReconstruction:
        """Parse COLMAP's output into our data structures."""
        # COLMAP writes models into sparse/0/, sparse/1/, etc.
        model_dirs = sorted(
            d
            for d in os.listdir(self.sparse_dir)
            if os.path.isdir(os.path.join(self.sparse_dir, d))
        )

        if not model_dirs:
            _logger.warning("No reconstruction models found — returning empty result")
            return SparseReconstruction()

        model_dir = os.path.join(self.sparse_dir, model_dirs[0])
        _logger.info("Loading reconstruction from %s", model_dir)

        recon = SparseReconstruction()
        recon.model_dir = model_dir

        # Convert binary model to txt for easy parsing
        txt_dir = os.path.join(self.workspace, "sparse_txt")
        ensure_dir(txt_dir)
        try:
            self._run_cmd(
                [
                    self.colmap, "model_converter",
                    "--input_path", model_dir,
                    "--output_path", txt_dir,
                    "--output_type", "TXT",
                ],
                "model_converter",
            )
            recon.load_from_txt(txt_dir)
        except Exception as exc:
            _logger.warning("Could not convert model to TXT: %s", exc)

        return recon

    # ── Subprocess runner ────────────────────────────────────────────────
    @staticmethod
    def _run_cmd(cmd: List[str], step_name: str) -> None:
        _logger.debug("Running: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            _logger.error("COLMAP %s failed:\n%s", step_name, result.stderr)
            raise RuntimeError(f"COLMAP {step_name} failed (exit {result.returncode})")
        _logger.debug("COLMAP %s stdout:\n%s", step_name, result.stdout[-500:] if result.stdout else "")
