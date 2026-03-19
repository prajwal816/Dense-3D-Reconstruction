"""
Point cloud processing: filtering, downsampling, normal estimation,
outlier removal.
"""

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..utils.logger import get_logger
from ..utils.io_utils import ensure_dir

_logger = get_logger("dense3d.reconstruction.pcd_proc")

try:
    import open3d as o3d

    _HAS_OPEN3D = True
except ImportError:
    _HAS_OPEN3D = False


class PointCloudProcessor:
    """Point cloud cleaning and pre-processing pipeline.

    Args:
        cfg: ``reconstruction`` section of the YAML config.
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        if not _HAS_OPEN3D:
            raise ImportError("Open3D is required for PointCloudProcessor")
        self.cfg = cfg or {}
        _logger.info("PointCloudProcessor initialized")

    def load(self, path: str) -> "o3d.geometry.PointCloud":
        """Load a point cloud from disk."""
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Point cloud not found: {path}")
        pcd = o3d.io.read_point_cloud(path)
        _logger.info("Loaded point cloud from %s (%d points)", path, len(pcd.points))
        return pcd

    def save(self, pcd: "o3d.geometry.PointCloud", path: str) -> None:
        """Write a point cloud to disk."""
        ensure_dir(os.path.dirname(path))
        o3d.io.write_point_cloud(path, pcd)
        _logger.info("Saved point cloud → %s (%d points)", path, len(pcd.points))

    def process(
        self,
        pcd: "o3d.geometry.PointCloud",
        voxel_downsample: bool = True,
        remove_outliers: bool = True,
        estimate_normals: bool = True,
    ) -> "o3d.geometry.PointCloud":
        """Run the full cleaning pipeline.

        Steps:
        1. Voxel downsampling
        2. Statistical outlier removal
        3. Normal estimation

        Returns:
            Cleaned point cloud.
        """
        n_before = len(pcd.points)

        if voxel_downsample:
            voxel_size = self.cfg.get("point_cloud", {}).get("voxel_size", 0.005)
            pcd = pcd.voxel_down_sample(voxel_size)
            _logger.info(
                "Voxel downsampled: %d → %d (voxel=%.4f)",
                n_before,
                len(pcd.points),
                voxel_size,
            )

        if remove_outliers:
            pcd = self._remove_outliers(pcd)

        if estimate_normals:
            pcd = self._estimate_normals(pcd)

        return pcd

    def _remove_outliers(
        self, pcd: "o3d.geometry.PointCloud"
    ) -> "o3d.geometry.PointCloud":
        pc_cfg = self.cfg.get("point_cloud", {})
        nb_neighbors = pc_cfg.get("nb_neighbors", 20)
        std_ratio = pc_cfg.get("std_ratio", 2.0)

        n_before = len(pcd.points)
        pcd, inlier_idx = pcd.remove_statistical_outlier(nb_neighbors, std_ratio)
        _logger.info(
            "Outlier removal: %d → %d (nb=%d, std=%.1f)",
            n_before,
            len(pcd.points),
            nb_neighbors,
            std_ratio,
        )
        return pcd

    def _estimate_normals(
        self, pcd: "o3d.geometry.PointCloud"
    ) -> "o3d.geometry.PointCloud":
        ne_cfg = self.cfg.get("normal_estimation", {})
        radius = ne_cfg.get("radius", 0.1)
        max_nn = ne_cfg.get("max_nn", 30)

        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius, max_nn=max_nn
            )
        )
        pcd.orient_normals_consistent_tangent_plane(k=15)
        _logger.info("Normals estimated (radius=%.3f, max_nn=%d)", radius, max_nn)
        return pcd

    @staticmethod
    def crop_bounding_box(
        pcd: "o3d.geometry.PointCloud",
        min_bound: np.ndarray,
        max_bound: np.ndarray,
    ) -> "o3d.geometry.PointCloud":
        """Crop a point cloud to an axis-aligned bounding box."""
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=min_bound, max_bound=max_bound
        )
        return pcd.crop(bbox)

    @staticmethod
    def compute_statistics(pcd: "o3d.geometry.PointCloud") -> Dict[str, Any]:
        """Compute basic statistics of a point cloud."""
        pts = np.asarray(pcd.points)
        stats: Dict[str, Any] = {
            "num_points": len(pts),
            "centroid": pts.mean(axis=0).tolist(),
            "min_bound": pts.min(axis=0).tolist(),
            "max_bound": pts.max(axis=0).tolist(),
            "has_colors": pcd.has_colors(),
            "has_normals": pcd.has_normals(),
        }
        return stats
