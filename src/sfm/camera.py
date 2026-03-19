"""
Camera intrinsics and extrinsics data classes.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class CameraModel:
    """Camera intrinsic parameters.

    Attributes:
        camera_id: Unique camera identifier.
        model: COLMAP camera model name.
        width: Image width in pixels.
        height: Image height in pixels.
        params: Intrinsic parameters (focal, cx, cy, distortion…).
    """

    camera_id: int = 0
    model: str = "SIMPLE_RADIAL"
    width: int = 0
    height: int = 0
    params: np.ndarray = field(default_factory=lambda: np.zeros(4))

    @property
    def focal_length(self) -> float:
        """Return the first focal-length parameter."""
        return float(self.params[0]) if self.params.size > 0 else 0.0

    @property
    def principal_point(self) -> Tuple[float, float]:
        """Return (cx, cy)."""
        if self.params.size >= 3:
            return float(self.params[1]), float(self.params[2])
        return self.width / 2.0, self.height / 2.0

    def to_K(self) -> np.ndarray:
        """Build 3×3 intrinsic matrix K."""
        f = self.focal_length
        cx, cy = self.principal_point
        return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)

    def to_dict(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "model": self.model,
            "width": self.width,
            "height": self.height,
            "params": self.params.tolist(),
        }


@dataclass
class CameraPose:
    """Camera extrinsic parameters (world-to-camera).

    Attributes:
        image_id: Unique image identifier.
        image_name: Filename.
        camera_id: Associated camera model ID.
        rotation: (3, 3) rotation matrix R (world→camera).
        translation: (3,) translation vector t.
    """

    image_id: int = 0
    image_name: str = ""
    camera_id: int = 0
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @property
    def projection_matrix(self) -> np.ndarray:
        """3×4 projection matrix [R | t]."""
        return np.hstack([self.rotation, self.translation.reshape(3, 1)])

    @property
    def camera_center(self) -> np.ndarray:
        """3D camera center in world coordinates: C = -R^T t."""
        return -self.rotation.T @ self.translation

    def to_dict(self) -> dict:
        return {
            "image_id": self.image_id,
            "image_name": self.image_name,
            "camera_id": self.camera_id,
            "rotation": self.rotation.tolist(),
            "translation": self.translation.tolist(),
        }


def save_cameras(
    cameras: Dict[int, CameraModel],
    poses: List[CameraPose],
    path: str,
) -> None:
    """Serialize cameras and poses to JSON."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = {
        "cameras": {str(k): v.to_dict() for k, v in cameras.items()},
        "poses": [p.to_dict() for p in poses],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_cameras(path: str) -> Tuple[Dict[int, CameraModel], List[CameraPose]]:
    """Deserialize cameras and poses from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cameras: Dict[int, CameraModel] = {}
    for k, v in data.get("cameras", {}).items():
        cam = CameraModel(
            camera_id=v["camera_id"],
            model=v["model"],
            width=v["width"],
            height=v["height"],
            params=np.array(v["params"]),
        )
        cameras[int(k)] = cam

    poses: List[CameraPose] = []
    for p in data.get("poses", []):
        pose = CameraPose(
            image_id=p["image_id"],
            image_name=p["image_name"],
            camera_id=p["camera_id"],
            rotation=np.array(p["rotation"]),
            translation=np.array(p["translation"]),
        )
        poses.append(pose)

    return cameras, poses
