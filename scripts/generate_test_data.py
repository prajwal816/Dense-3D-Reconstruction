#!/usr/bin/env python3
"""
Generate synthetic multi-view test images for pipeline validation.

Creates a 3D scene with textured objects and renders it from multiple
camera viewpoints using OpenCV projection, producing a set of images
that the full reconstruction pipeline can consume.

Usage:
    python scripts/generate_test_data.py --output data/images --num-views 8
"""

import argparse
import os
import sys
import math

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils.io_utils import ensure_dir
from src.utils.logger import get_logger

_logger = get_logger("dense3d.testdata")


# ═════════════════════════════════════════════════════════════════════════════
# 3D Scene Definition
# ═════════════════════════════════════════════════════════════════════════════

def _make_textured_cube(center, size, density=50):
    """Generate coloured 3D points on the surface of a cube."""
    pts, colors = [], []
    half = size / 2.0
    cx, cy, cz = center

    for face_axis in range(3):
        for sign in [-1, 1]:
            for _ in range(density):
                p = [0.0, 0.0, 0.0]
                p[face_axis] = sign * half
                other = [i for i in range(3) if i != face_axis]
                p[other[0]] = np.random.uniform(-half, half)
                p[other[1]] = np.random.uniform(-half, half)
                pts.append([p[0] + cx, p[1] + cy, p[2] + cz])
                # Colour based on face
                r = int(128 + 127 * sign * (face_axis == 0))
                g = int(128 + 127 * sign * (face_axis == 1))
                b = int(128 + 127 * sign * (face_axis == 2))
                colors.append([r, g, b])

    return np.array(pts), np.array(colors, dtype=np.uint8)


def _make_ground_plane(size=4.0, y=-1.0, density=400):
    """Generate a textured ground plane with a checkerboard pattern."""
    pts, colors = [], []
    half = size / 2.0

    for _ in range(density):
        x = np.random.uniform(-half, half)
        z = np.random.uniform(-half, half)
        pts.append([x, y, z])

        # Checkerboard
        cx = int(x * 2) % 2
        cz = int(z * 2) % 2
        if (cx + cz) % 2 == 0:
            colors.append([200, 200, 200])
        else:
            colors.append([80, 80, 80])

    return np.array(pts), np.array(colors, dtype=np.uint8)


def _make_sphere(center, radius, density=200):
    """Generate coloured points on a sphere surface."""
    pts, colors = [], []
    cx, cy, cz = center

    for _ in range(density):
        theta = np.random.uniform(0, 2 * math.pi)
        phi = np.random.uniform(0, math.pi)
        x = cx + radius * math.sin(phi) * math.cos(theta)
        y = cy + radius * math.sin(phi) * math.sin(theta)
        z = cz + radius * math.cos(phi)
        pts.append([x, y, z])
        r = int(128 + 127 * math.sin(theta))
        g = int(128 + 127 * math.cos(phi))
        b = 180
        colors.append([max(0, min(255, r)), max(0, min(255, g)), b])

    return np.array(pts), np.array(colors, dtype=np.uint8)


def build_scene():
    """Assemble 3D scene: cubes + ground + sphere."""
    all_pts, all_colors = [], []

    # Central cube
    p, c = _make_textured_cube([0, 0, 0], 1.0, density=80)
    all_pts.append(p); all_colors.append(c)

    # Smaller offset cube
    p, c = _make_textured_cube([1.5, -0.5, 0.8], 0.6, density=60)
    all_pts.append(p); all_colors.append(c)

    # Sphere
    p, c = _make_sphere([-1.2, 0.3, -0.5], 0.5, density=150)
    all_pts.append(p); all_colors.append(c)

    # Ground plane
    p, c = _make_ground_plane(size=5.0, y=-1.0, density=500)
    all_pts.append(p); all_colors.append(c)

    # Random feature points (extra texture)
    n_random = 200
    rp = np.random.uniform(-2, 2, (n_random, 3))
    rc = np.random.randint(50, 250, (n_random, 3), dtype=np.uint8)
    all_pts.append(rp); all_colors.append(rc)

    points = np.vstack(all_pts)
    colors = np.vstack(all_colors)
    return points, colors


# ═════════════════════════════════════════════════════════════════════════════
# Camera & Rendering
# ═════════════════════════════════════════════════════════════════════════════

def _camera_orbit(num_views, radius=3.5, height=1.0, look_at=None):
    """Generate camera extrinsics orbiting around a point."""
    if look_at is None:
        look_at = np.array([0.0, 0.0, 0.0])

    poses = []
    for i in range(num_views):
        angle = 2 * math.pi * i / num_views
        cx = radius * math.cos(angle)
        cz = radius * math.sin(angle)
        cy = height + 0.3 * math.sin(angle * 2)  # Slight vertical variation

        cam_pos = np.array([cx, cy, cz])

        # Look-at matrix
        forward = look_at - cam_pos
        forward /= np.linalg.norm(forward)
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)

        R = np.stack([right, -up, forward], axis=0)  # world-to-camera
        t = -R @ cam_pos

        poses.append((R, t, cam_pos))

    return poses


def render_image(points3d, colors, R, t, K, width, height):
    """Project 3D points and render a synthetic image."""
    # Project: p_cam = R @ p_world + t
    p_cam = (R @ points3d.T).T + t
    # Filter points behind camera
    mask = p_cam[:, 2] > 0.1
    p_cam = p_cam[mask]
    c = colors[mask]

    # Project to pixel
    px = (K[0, 0] * p_cam[:, 0] / p_cam[:, 2]) + K[0, 2]
    py = (K[1, 1] * p_cam[:, 1] / p_cam[:, 2]) + K[1, 2]

    img = np.zeros((height, width, 3), dtype=np.uint8)
    # Soft background gradient
    for row in range(height):
        ratio = row / height
        img[row, :] = [int(30 + 40 * ratio), int(30 + 30 * ratio), int(50 + 50 * ratio)]

    # Draw points as small circles with depth ordering
    depths = p_cam[:, 2]
    order = np.argsort(-depths)  # back to front

    for idx in order:
        x, y = int(round(px[idx])), int(round(py[idx]))
        if 0 <= x < width and 0 <= y < height:
            radius = max(2, int(8 / depths[idx]))
            color = (int(c[idx, 0]), int(c[idx, 1]), int(c[idx, 2]))
            cv2.circle(img, (x, y), radius, color, -1, cv2.LINE_AA)

    # Add Gaussian noise for realism
    noise = np.random.normal(0, 3, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Slight Gaussian blur
    img = cv2.GaussianBlur(img, (3, 3), 0.5)

    return img


def generate_test_images(output_dir, num_views=8, width=640, height=480):
    """Generate a complete set of multi-view test images."""
    ensure_dir(output_dir)

    _logger.info("Building 3D scene …")
    points, colors = build_scene()
    _logger.info("Scene: %d points", points.shape[0])

    # Camera intrinsics
    fx = fy = 525.0
    cx, cy = width / 2.0, height / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    # Generate orbiting cameras
    poses = _camera_orbit(num_views, radius=3.5, height=1.0)

    _logger.info("Rendering %d views (%dx%d) …", num_views, width, height)
    for i, (R, t, cam_pos) in enumerate(poses):
        img = render_image(points, colors, R, t, K, width, height)

        filename = f"view_{i:03d}.png"
        path = os.path.join(output_dir, filename)
        cv2.imwrite(path, img)
        _logger.info("  [%d/%d] %s  (cam at [%.1f, %.1f, %.1f])",
                      i + 1, num_views, filename, *cam_pos)

    # Save ground-truth camera info for reference
    gt_path = os.path.join(output_dir, "ground_truth_cameras.txt")
    with open(gt_path, "w") as f:
        f.write("# view_id  fx  fy  cx  cy  R(9 vals)  t(3 vals)\n")
        for i, (R, t, _) in enumerate(poses):
            r_flat = " ".join(f"{v:.6f}" for v in R.flatten())
            t_flat = " ".join(f"{v:.6f}" for v in t)
            f.write(f"{i}  {fx}  {fy}  {cx}  {cy}  {r_flat}  {t_flat}\n")

    _logger.info("Ground-truth cameras → %s", gt_path)
    _logger.info("✓ Generated %d test images in %s", num_views, output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic multi-view test images"
    )
    parser.add_argument(
        "--output", type=str, default="data/images",
        help="Output directory for images (default: data/images)"
    )
    parser.add_argument(
        "--num-views", type=int, default=8,
        help="Number of camera viewpoints (default: 8)"
    )
    parser.add_argument(
        "--width", type=int, default=640,
        help="Image width (default: 640)"
    )
    parser.add_argument(
        "--height", type=int, default=480,
        help="Image height (default: 480)"
    )
    args = parser.parse_args()
    generate_test_images(args.output, args.num_views, args.width, args.height)


if __name__ == "__main__":
    main()
