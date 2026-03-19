"""
File I/O utilities: image loading, PLY read/write, directory helpers.
"""

import os
import glob
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .logger import get_logger

_logger = get_logger("dense3d.io")

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def ensure_dir(path: str) -> str:
    """Create directory (and parents) if it does not exist. Returns the path."""
    os.makedirs(path, exist_ok=True)
    return path


def load_image(
    path: str,
    grayscale: bool = False,
    max_size: Optional[int] = None,
) -> np.ndarray:
    """Load an image from disk.

    Args:
        path: Image file path.
        grayscale: If ``True`` convert to single-channel.
        max_size: If set, resize so the longest edge equals *max_size*.

    Returns:
        Image as a NumPy array (BGR or grayscale).

    Raises:
        FileNotFoundError: If *path* does not exist.
        RuntimeError: If OpenCV cannot decode the file.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path}")

    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(path, flag)
    if img is None:
        raise RuntimeError(f"Failed to decode image: {path}")

    if max_size is not None:
        h, w = img.shape[:2]
        scale = max_size / max(h, w)
        if scale < 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return img


def load_images_from_dir(
    directory: str,
    grayscale: bool = False,
    max_size: Optional[int] = None,
) -> List[Tuple[str, np.ndarray]]:
    """Load all images from a directory, sorted by filename.

    Returns:
        List of *(filename, image_array)* tuples.
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Image directory not found: {directory}")

    paths: List[str] = sorted(
        p
        for p in glob.glob(os.path.join(directory, "*"))
        if os.path.splitext(p)[1].lower() in _IMAGE_EXTENSIONS
    )

    if not paths:
        _logger.warning("No images found in %s", directory)
        return []

    images: List[Tuple[str, np.ndarray]] = []
    for p in paths:
        try:
            img = load_image(p, grayscale=grayscale, max_size=max_size)
            images.append((os.path.basename(p), img))
        except Exception as exc:
            _logger.warning("Skipping %s: %s", p, exc)

    _logger.info("Loaded %d images from %s", len(images), directory)
    return images


def write_ply(path: str, points: np.ndarray, colors: Optional[np.ndarray] = None) -> None:
    """Write a point cloud to a PLY file.

    Args:
        path: Output file path.
        points: (N, 3) float array of XYZ coordinates.
        colors: Optional (N, 3) uint8 array of RGB colors.
    """
    ensure_dir(os.path.dirname(path))
    n = points.shape[0]
    has_color = colors is not None and colors.shape[0] == n

    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if has_color:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            line = f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f}"
            if has_color:
                line += f" {int(colors[i, 0])} {int(colors[i, 1])} {int(colors[i, 2])}"
            f.write(line + "\n")

    _logger.info("Wrote PLY (%d points) → %s", n, path)


def read_ply(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Read a PLY file (ASCII format).

    Returns:
        Tuple of (points, colors) where colors may be ``None``.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"PLY file not found: {path}")

    try:
        import open3d as o3d

        pcd = o3d.io.read_point_cloud(path)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        if colors.size == 0:
            colors = None
        else:
            colors = (colors * 255).astype(np.uint8)
        return points, colors
    except ImportError:
        _logger.warning("Open3D not available; falling back to manual PLY reader")

    points_list: List[List[float]] = []
    colors_list: List[List[int]] = []
    header_ended = False
    has_color = False

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not header_ended:
                if "property" in line and "red" in line:
                    has_color = True
                if line == "end_header":
                    header_ended = True
                continue
            parts = line.split()
            points_list.append([float(parts[0]), float(parts[1]), float(parts[2])])
            if has_color and len(parts) >= 6:
                colors_list.append([int(parts[3]), int(parts[4]), int(parts[5])])

    pts = np.array(points_list, dtype=np.float64)
    clr = np.array(colors_list, dtype=np.uint8) if colors_list else None
    return pts, clr
