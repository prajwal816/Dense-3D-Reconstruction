"""
Abstract base class for feature extractors.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class FeatureData:
    """Container for extracted keypoints + descriptors.

    Attributes:
        keypoints: (N, 2) float array of (x, y) pixel coordinates.
        descriptors: (N, D) descriptor matrix.
        scores: Optional (N,) confidence scores.
        image_name: Source image filename.
    """

    keypoints: np.ndarray
    descriptors: np.ndarray
    scores: Optional[np.ndarray] = None
    image_name: str = ""

    @property
    def num_keypoints(self) -> int:
        return self.keypoints.shape[0]


class FeatureExtractor(ABC):
    """Interface every feature extractor must implement."""

    name: str = "base"

    @abstractmethod
    def extract(self, image: np.ndarray) -> FeatureData:
        """Extract keypoints and descriptors from a single image.

        Args:
            image: HxW (grayscale) or HxWx3 (BGR) uint8 array.

        Returns:
            :class:`FeatureData` holding keypoints and descriptors.
        """
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} ({self.name})>"
