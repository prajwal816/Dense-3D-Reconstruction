"""
Camera trajectory visualization (Matplotlib 3D).
"""

import os
from typing import List, Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from ..utils.logger import get_logger
from ..utils.io_utils import ensure_dir

_logger = get_logger("dense3d.viz.trajectory")


def plot_camera_trajectory(
    poses: list,
    points3d: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
    title: str = "Camera Trajectory",
    show: bool = False,
    max_points: int = 5000,
) -> None:
    """Plot camera centres and (optionally) sparse 3D points in 3D.

    Args:
        poses: List of :class:`CameraPose` objects.
        points3d: Optional (N, 3) sparse point array.
        output_path: Save figure to this path if given.
        title: Plot title.
        show: Whether to display interactively.
        max_points: Cap on scatter points for readability.
    """
    if not poses:
        _logger.warning("No poses to plot")
        return

    centres = np.array([p.camera_center for p in poses])

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Camera path
    ax.plot(
        centres[:, 0],
        centres[:, 1],
        centres[:, 2],
        "r-o",
        markersize=4,
        linewidth=1.5,
        label="Camera path",
    )

    # Camera look directions
    for pose in poses:
        C = pose.camera_center
        forward = pose.rotation.T @ np.array([0, 0, 1])
        ax.quiver(
            C[0], C[1], C[2],
            forward[0], forward[1], forward[2],
            length=0.15,
            color="blue",
            alpha=0.4,
        )

    # Sparse points
    if points3d is not None and points3d.shape[0] > 0:
        pts = points3d
        if pts.shape[0] > max_points:
            idx = np.random.choice(pts.shape[0], max_points, replace=False)
            pts = pts[idx]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            c="gray",
            s=0.5,
            alpha=0.3,
            label=f"3D points ({points3d.shape[0]})",
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()

    if output_path:
        ensure_dir(os.path.dirname(output_path))
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        _logger.info("Camera trajectory plot saved → %s", output_path)

    if show:
        plt.show()

    plt.close(fig)
