"""
COLMAP Multi-View Stereo wrapper.

Drives COLMAP's image undistortion, patch-match stereo, and stereo fusion
through CLI subprocess calls.
"""

import os
import subprocess
from typing import Any, Dict, List, Optional

from ..utils.logger import get_logger
from ..utils.io_utils import ensure_dir

_logger = get_logger("dense3d.mvs.colmap")


class COLMAPMVS:
    """Wrapper around COLMAP's dense MVS pipeline.

    Pipeline:  image_undistorter → patch_match_stereo → stereo_fusion

    Args:
        colmap_binary: Path to the ``colmap`` executable.
        workspace: Root workspace directory.
        cfg: MVS section of the YAML config.
    """

    def __init__(
        self,
        colmap_binary: str = "colmap",
        workspace: str = "outputs/colmap_ws",
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.colmap = colmap_binary
        self.workspace = os.path.abspath(workspace)
        self.cfg = cfg or {}

        self.dense_dir = os.path.join(self.workspace, "dense")
        self.fused_ply = os.path.join(self.dense_dir, "fused.ply")

        _logger.info("COLMAP MVS wrapper (binary=%s)", colmap_binary)

    def run(
        self,
        image_dir: str,
        sparse_dir: str,
    ) -> str:
        """Execute the full COLMAP MVS pipeline.

        Args:
            image_dir: Original images directory.
            sparse_dir: Path to sparse model (e.g. ``sparse/0``).

        Returns:
            Path to the fused dense point cloud PLY file.
        """
        ensure_dir(self.dense_dir)

        self._undistort_images(image_dir, sparse_dir)
        self._patch_match_stereo()
        self._stereo_fusion()

        if os.path.isfile(self.fused_ply):
            _logger.info("Dense point cloud written to %s", self.fused_ply)
        else:
            _logger.warning("Expected fused PLY not found at %s", self.fused_ply)

        return self.fused_ply

    # ── Step 1: Image undistortion ───────────────────────────────────────
    def _undistort_images(self, image_dir: str, sparse_dir: str) -> None:
        _logger.info("COLMAP MVS: undistorting images …")
        cmd = [
            self.colmap, "image_undistorter",
            "--image_path", os.path.abspath(image_dir),
            "--input_path", os.path.abspath(sparse_dir),
            "--output_path", self.dense_dir,
            "--output_type", "COLMAP",
        ]
        max_size = self.cfg.get("patch_match", {}).get("max_image_size")
        if max_size:
            cmd += ["--max_image_size", str(max_size)]

        self._run_cmd(cmd, "image_undistorter")

    # ── Step 2: Patch-match stereo ───────────────────────────────────────
    def _patch_match_stereo(self) -> None:
        _logger.info("COLMAP MVS: running patch-match stereo …")
        pm_cfg = self.cfg.get("patch_match", {})
        cmd = [
            self.colmap, "patch_match_stereo",
            "--workspace_path", self.dense_dir,
        ]
        if pm_cfg.get("window_radius"):
            cmd += ["--PatchMatchStereo.window_radius", str(pm_cfg["window_radius"])]
        if pm_cfg.get("num_iterations"):
            cmd += ["--PatchMatchStereo.num_iterations", str(pm_cfg["num_iterations"])]
        if pm_cfg.get("geom_consistency") is not None:
            cmd += [
                "--PatchMatchStereo.geom_consistency",
                "true" if pm_cfg["geom_consistency"] else "false",
            ]

        self._run_cmd(cmd, "patch_match_stereo")

    # ── Step 3: Stereo fusion ────────────────────────────────────────────
    def _stereo_fusion(self) -> None:
        _logger.info("COLMAP MVS: fusing depth maps …")
        fuse_cfg = self.cfg.get("fusion", {})
        cmd = [
            self.colmap, "stereo_fusion",
            "--workspace_path", self.dense_dir,
            "--output_path", self.fused_ply,
        ]
        if fuse_cfg.get("min_num_pixels"):
            cmd += ["--StereoFusion.min_num_pixels", str(fuse_cfg["min_num_pixels"])]
        if fuse_cfg.get("max_reproj_error"):
            cmd += ["--StereoFusion.max_reproj_error", str(fuse_cfg["max_reproj_error"])]

        self._run_cmd(cmd, "stereo_fusion")

    # ── Subprocess runner ────────────────────────────────────────────────
    @staticmethod
    def _run_cmd(cmd: List[str], step_name: str) -> None:
        _logger.debug("Running: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            _logger.error("COLMAP %s failed:\n%s", step_name, result.stderr)
            raise RuntimeError(f"COLMAP {step_name} failed (exit {result.returncode})")
        _logger.debug("COLMAP %s complete", step_name)
