"""
Sparse reconstruction data structure and COLMAP TXT parser.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .camera import CameraModel, CameraPose
from ..utils.logger import get_logger

_logger = get_logger("dense3d.sfm.sparse")


@dataclass
class SparseReconstruction:
    """Container for a sparse SfM reconstruction.

    Attributes:
        cameras: Camera intrinsic models keyed by camera_id.
        poses: List of estimated camera poses.
        points3d: (N, 3) array of triangulated 3D points.
        point_colors: Optional (N, 3) uint8 RGB colours.
        point_errors: Optional (N,) reprojection errors.
        model_dir: Path to the COLMAP model directory.
    """

    cameras: Dict[int, CameraModel] = field(default_factory=dict)
    poses: List[CameraPose] = field(default_factory=list)
    points3d: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    point_colors: Optional[np.ndarray] = None
    point_errors: Optional[np.ndarray] = None
    model_dir: str = ""

    @property
    def num_cameras(self) -> int:
        return len(self.cameras)

    @property
    def num_poses(self) -> int:
        return len(self.poses)

    @property
    def num_points(self) -> int:
        return self.points3d.shape[0]

    # ── COLMAP TXT parsers ───────────────────────────────────────────────
    def load_from_txt(self, txt_dir: str) -> None:
        """Parse COLMAP TXT model files (cameras.txt, images.txt, points3D.txt)."""
        self._parse_cameras_txt(os.path.join(txt_dir, "cameras.txt"))
        self._parse_images_txt(os.path.join(txt_dir, "images.txt"))
        self._parse_points3d_txt(os.path.join(txt_dir, "points3D.txt"))
        _logger.info(
            "Loaded sparse model: %d cameras, %d images, %d points",
            self.num_cameras,
            self.num_poses,
            self.num_points,
        )

    def _parse_cameras_txt(self, path: str) -> None:
        if not os.path.isfile(path):
            _logger.warning("cameras.txt not found at %s", path)
            return
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                cam = CameraModel(
                    camera_id=int(parts[0]),
                    model=parts[1],
                    width=int(parts[2]),
                    height=int(parts[3]),
                    params=np.array([float(x) for x in parts[4:]]),
                )
                self.cameras[cam.camera_id] = cam

    def _parse_images_txt(self, path: str) -> None:
        if not os.path.isfile(path):
            _logger.warning("images.txt not found at %s", path)
            return
        with open(path, "r") as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]

        # images.txt has pairs of lines: pose line, then keypoint line
        for i in range(0, len(lines), 2):
            parts = lines[i].split()
            if len(parts) < 10:
                continue
            image_id = int(parts[0])
            qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
            camera_id = int(parts[8])
            image_name = parts[9]

            # Quaternion (w, x, y, z) → rotation matrix
            R = self._quat_to_rot(qw, qx, qy, qz)

            pose = CameraPose(
                image_id=image_id,
                image_name=image_name,
                camera_id=camera_id,
                rotation=R,
                translation=np.array([tx, ty, tz]),
            )
            self.poses.append(pose)

    def _parse_points3d_txt(self, path: str) -> None:
        if not os.path.isfile(path):
            _logger.warning("points3D.txt not found at %s", path)
            return
        points, colors, errors = [], [], []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                points.append([float(parts[1]), float(parts[2]), float(parts[3])])
                colors.append([int(parts[4]), int(parts[5]), int(parts[6])])
                errors.append(float(parts[7]))

        if points:
            self.points3d = np.array(points, dtype=np.float64)
            self.point_colors = np.array(colors, dtype=np.uint8)
            self.point_errors = np.array(errors, dtype=np.float64)

    @staticmethod
    def _quat_to_rot(w: float, x: float, y: float, z: float) -> np.ndarray:
        """Convert quaternion (w, x, y, z) to 3×3 rotation matrix."""
        R = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )
        return R

    def summary(self) -> str:
        """Return a formatted summary string."""
        lines = [
            f"Sparse Reconstruction Summary",
            f"  Cameras : {self.num_cameras}",
            f"  Images  : {self.num_poses}",
            f"  Points  : {self.num_points}",
        ]
        if self.point_errors is not None and self.point_errors.size > 0:
            lines.append(
                f"  Reproj. : mean={self.point_errors.mean():.4f}, "
                f"median={np.median(self.point_errors):.4f}"
            )
        return "\n".join(lines)
