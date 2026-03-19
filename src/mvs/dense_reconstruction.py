"""
Open3D-based dense reconstruction utilities.

Provides TSDF volume integration and depth-map fusion as an alternative
to COLMAP's built-in MVS when more control is needed.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..utils.logger import get_logger
from ..utils.io_utils import ensure_dir

_logger = get_logger("dense3d.mvs.dense")

try:
    import open3d as o3d

    _HAS_OPEN3D = True
except ImportError:
    _HAS_OPEN3D = False


class DenseReconstructor:
    """Open3D TSDF-based dense reconstruction from depth maps.

    Args:
        voxel_length: TSDF voxel size in world units.
        sdf_trunc: SDF truncation distance.
        depth_scale: Scale factor to convert raw depth to metres.
        depth_trunc: Maximum depth in metres.
    """

    def __init__(
        self,
        voxel_length: float = 0.005,
        sdf_trunc: float = 0.04,
        depth_scale: float = 1000.0,
        depth_trunc: float = 3.0,
    ) -> None:
        if not _HAS_OPEN3D:
            raise ImportError("Open3D is required for DenseReconstructor")

        self.voxel_length = voxel_length
        self.sdf_trunc = sdf_trunc
        self.depth_scale = depth_scale
        self.depth_trunc = depth_trunc

        _logger.info(
            "DenseReconstructor (voxel=%.4f, trunc=%.3f, depth_scale=%.0f)",
            voxel_length,
            sdf_trunc,
            depth_scale,
        )

    def integrate(
        self,
        color_images: List[np.ndarray],
        depth_images: List[np.ndarray],
        intrinsic: np.ndarray,
        extrinsics: List[np.ndarray],
        width: int,
        height: int,
    ) -> "o3d.geometry.TriangleMesh":
        """Integrate RGBD frames into a TSDF volume and extract a mesh.

        Args:
            color_images: List of HxWx3 uint8 BGR images.
            depth_images: List of HxW uint16/float32 depth maps.
            intrinsic: 3×3 camera intrinsic matrix.
            extrinsics: List of 4×4 world-to-camera matrices.
            width: Image width.
            height: Image height.

        Returns:
            Open3D TriangleMesh extracted from the TSDF volume.
        """
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=self.voxel_length,
            sdf_trunc=self.sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

        cam_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width,
            height,
            intrinsic[0, 0],
            intrinsic[1, 1],
            intrinsic[0, 2],
            intrinsic[1, 2],
        )

        for i, (color, depth, extr) in enumerate(
            zip(color_images, depth_images, extrinsics)
        ):
            import cv2

            color_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(color_rgb),
                o3d.geometry.Image(depth.astype(np.float32)),
                depth_scale=self.depth_scale,
                depth_trunc=self.depth_trunc,
                convert_rgb_to_intensity=False,
            )
            volume.integrate(rgbd, cam_intrinsic, np.linalg.inv(extr))
            _logger.debug("Integrated frame %d / %d", i + 1, len(color_images))

        mesh = volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        _logger.info(
            "TSDF mesh: %d vertices, %d triangles",
            len(mesh.vertices),
            len(mesh.triangles),
        )
        return mesh

    def depth_to_pointcloud(
        self,
        color: np.ndarray,
        depth: np.ndarray,
        intrinsic: np.ndarray,
        extrinsic: np.ndarray,
        width: int,
        height: int,
    ) -> "o3d.geometry.PointCloud":
        """Convert a single RGBD frame to a point cloud."""
        import cv2

        color_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color_rgb),
            o3d.geometry.Image(depth.astype(np.float32)),
            depth_scale=self.depth_scale,
            depth_trunc=self.depth_trunc,
            convert_rgb_to_intensity=False,
        )
        cam = o3d.camera.PinholeCameraIntrinsic(
            width,
            height,
            intrinsic[0, 0],
            intrinsic[1, 1],
            intrinsic[0, 2],
            intrinsic[1, 2],
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd, cam, np.linalg.inv(extrinsic)
        )
        return pcd
