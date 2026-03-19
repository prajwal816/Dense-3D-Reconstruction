"""
Interactive 3D viewer for point clouds and meshes using Open3D.
"""

import os
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ..utils.logger import get_logger

_logger = get_logger("dense3d.viz.viewer")

try:
    import open3d as o3d

    _HAS_OPEN3D = True
except ImportError:
    _HAS_OPEN3D = False


class Viewer3D:
    """Open3D-based interactive 3D viewer.

    Args:
        window_name: Title of the viewer window.
        width: Window width in pixels.
        height: Window height in pixels.
        background_color: RGB background (0–1 each).
        point_size: Render point size.
    """

    def __init__(
        self,
        window_name: str = "Dense3D Viewer",
        width: int = 1920,
        height: int = 1080,
        background_color: Optional[List[float]] = None,
        point_size: float = 2.0,
    ) -> None:
        if not _HAS_OPEN3D:
            raise ImportError("Open3D is required for Viewer3D")

        self.window_name = window_name
        self.width = width
        self.height = height
        self.bg_color = background_color or [0.1, 0.1, 0.1]
        self.point_size = point_size
        self._geometries: List[Any] = []
        _logger.info("Viewer3D initialized (%dx%d)", width, height)

    def add_point_cloud(
        self,
        pcd: "o3d.geometry.PointCloud",
        color: Optional[List[float]] = None,
    ) -> None:
        """Add a point cloud to the scene."""
        if color is not None and not pcd.has_colors():
            pcd.paint_uniform_color(color)
        self._geometries.append(pcd)
        _logger.debug("Added point cloud (%d points)", len(pcd.points))

    def add_mesh(self, mesh: "o3d.geometry.TriangleMesh") -> None:
        """Add a triangle mesh to the scene."""
        self._geometries.append(mesh)
        _logger.debug(
            "Added mesh (%d verts, %d tris)",
            len(mesh.vertices),
            len(mesh.triangles),
        )

    def add_camera_frustums(
        self,
        poses: list,
        intrinsic: Optional[np.ndarray] = None,
        scale: float = 0.3,
        color: Optional[List[float]] = None,
    ) -> None:
        """Draw camera frustums as wireframe line sets.

        Args:
            poses: List of CameraPose objects.
            intrinsic: 3×3 camera matrix (uses a default if not given).
            scale: Frustum size scale.
            color: Frustum color [R, G, B].
        """
        c = color or [1.0, 0.2, 0.2]
        for pose in poses:
            frustum = self._make_frustum(
                pose.rotation, pose.translation, intrinsic, scale
            )
            frustum.paint_uniform_color(c)
            self._geometries.append(frustum)

    def add_coordinate_frame(self, size: float = 1.0, origin: Optional[List[float]] = None) -> None:
        """Add an XYZ coordinate frame."""
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=size, origin=origin or [0, 0, 0]
        )
        self._geometries.append(frame)

    def show(self) -> None:
        """Launch the interactive viewer."""
        if not self._geometries:
            _logger.warning("Nothing to display")
            return

        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name=self.window_name,
            width=self.width,
            height=self.height,
        )

        for geom in self._geometries:
            vis.add_geometry(geom)

        opt = vis.get_render_option()
        opt.background_color = np.array(self.bg_color)
        opt.point_size = self.point_size

        _logger.info("Launching viewer — close window to continue")
        vis.run()
        vis.destroy_window()

    def save_screenshot(self, path: str) -> None:
        """Render to an off-screen image and save."""
        from ..utils.io_utils import ensure_dir

        ensure_dir(os.path.dirname(path))
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name=self.window_name,
            width=self.width,
            height=self.height,
            visible=False,
        )
        for geom in self._geometries:
            vis.add_geometry(geom)
        opt = vis.get_render_option()
        opt.background_color = np.array(self.bg_color)
        opt.point_size = self.point_size
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(path, do_render=True)
        vis.destroy_window()
        _logger.info("Screenshot saved → %s", path)

    # ── Helpers ──────────────────────────────────────────────────────────
    @staticmethod
    def _make_frustum(
        R: np.ndarray,
        t: np.ndarray,
        K: Optional[np.ndarray],
        scale: float,
    ) -> "o3d.geometry.LineSet":
        """Create a simple camera frustum line-set."""
        if K is None:
            K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)

        w, h = int(K[0, 2] * 2), int(K[1, 2] * 2)
        corners_img = np.array(
            [[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64
        )
        # Back-project to z=scale
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        corners_cam = np.zeros((4, 3))
        for i, (u, v) in enumerate(corners_img):
            corners_cam[i] = [(u - cx) / fx * scale, (v - cy) / fy * scale, scale]

        # Camera centre in world
        C = -R.T @ t
        corners_world = (R.T @ corners_cam.T).T + C

        points = np.vstack([C.reshape(1, 3), corners_world])
        lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]

        ls = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines),
        )
        return ls
