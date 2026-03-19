"""
Surface reconstruction from point clouds — Poisson and Ball-Pivoting.
"""

import os
from typing import Any, Dict, List, Optional

import numpy as np

from ..utils.logger import get_logger
from ..utils.io_utils import ensure_dir

_logger = get_logger("dense3d.reconstruction.mesh")

try:
    import open3d as o3d

    _HAS_OPEN3D = True
except ImportError:
    _HAS_OPEN3D = False


class MeshGenerator:
    """Generate triangle meshes from point clouds.

    Supports Poisson surface reconstruction and Ball-Pivoting algorithm.

    Args:
        method: ``"poisson"`` or ``"ball_pivoting"``.
        cfg: Optional reconstruction config section.
    """

    def __init__(
        self,
        method: str = "poisson",
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not _HAS_OPEN3D:
            raise ImportError("Open3D is required for MeshGenerator")

        self.method = method.lower()
        self.cfg = cfg or {}
        _logger.info("MeshGenerator initialized (method=%s)", self.method)

    def generate(
        self,
        pcd: "o3d.geometry.PointCloud",
        output_path: Optional[str] = None,
    ) -> "o3d.geometry.TriangleMesh":
        """Generate a mesh from a point cloud.

        Args:
            pcd: Input Open3D point cloud (must have normals).
            output_path: If provided, save the mesh to this path.

        Returns:
            Open3D TriangleMesh.
        """
        # Ensure normals exist
        if not pcd.has_normals():
            _logger.info("Estimating normals for mesh generation …")
            ne_cfg = self.cfg.get("normal_estimation", {})
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=ne_cfg.get("radius", 0.1),
                    max_nn=ne_cfg.get("max_nn", 30),
                )
            )
            pcd.orient_normals_consistent_tangent_plane(k=15)

        if self.method == "poisson":
            mesh = self._poisson(pcd)
        elif self.method == "ball_pivoting":
            mesh = self._ball_pivoting(pcd)
        else:
            raise ValueError(f"Unknown mesh method: {self.method}")

        mesh.compute_vertex_normals()

        if output_path:
            ensure_dir(os.path.dirname(output_path))
            o3d.io.write_triangle_mesh(output_path, mesh)
            _logger.info("Mesh saved → %s", output_path)

        _logger.info(
            "Mesh: %d vertices, %d triangles",
            len(mesh.vertices),
            len(mesh.triangles),
        )
        return mesh

    # ── Poisson ──────────────────────────────────────────────────────────
    def _poisson(self, pcd: "o3d.geometry.PointCloud") -> "o3d.geometry.TriangleMesh":
        pcfg = self.cfg.get("poisson", {})
        depth = pcfg.get("depth", 9)
        width = pcfg.get("width", 0)
        scale = pcfg.get("scale", 1.1)
        linear_fit = pcfg.get("linear_fit", False)

        _logger.info("Running Poisson reconstruction (depth=%d, scale=%.2f) …", depth, scale)
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth, width=width, scale=scale, linear_fit=linear_fit
        )

        # Remove low-density vertices (clean up)
        densities = np.asarray(densities)
        density_threshold = np.quantile(densities, 0.01)
        vertices_to_remove = densities < density_threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)

        return mesh

    # ── Ball-Pivoting ────────────────────────────────────────────────────
    def _ball_pivoting(
        self, pcd: "o3d.geometry.PointCloud"
    ) -> "o3d.geometry.TriangleMesh":
        bpa_cfg = self.cfg.get("ball_pivoting", {})
        radii = bpa_cfg.get("radii", [0.005, 0.01, 0.02, 0.04])

        _logger.info("Running Ball-Pivoting reconstruction (radii=%s) …", radii)
        radii_vec = o3d.utility.DoubleVector(radii)
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, radii_vec
        )
        return mesh
