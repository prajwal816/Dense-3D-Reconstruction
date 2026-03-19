"""
ORB feature extractor (OpenCV).
"""

import cv2
import numpy as np

from .base import FeatureExtractor, FeatureData
from ..utils.logger import get_logger

_logger = get_logger("dense3d.features.orb")


class ORBExtractor(FeatureExtractor):
    """Oriented FAST and Rotated BRIEF feature extractor.

    Args:
        n_features: Maximum number of keypoints.
        scale_factor: Pyramid decimation ratio.
        n_levels: Number of pyramid levels.
        edge_threshold: Border size to ignore.
        patch_size: Size of the neighbourhood patch.
    """

    name = "orb"

    def __init__(
        self,
        n_features: int = 5000,
        scale_factor: float = 1.2,
        n_levels: int = 8,
        edge_threshold: int = 31,
        patch_size: int = 31,
    ) -> None:
        self.detector = cv2.ORB_create(
            nfeatures=n_features,
            scaleFactor=scale_factor,
            nlevels=n_levels,
            edgeThreshold=edge_threshold,
            patchSize=patch_size,
        )
        self._n_features = n_features
        _logger.info(
            "Initialized ORB extractor (max_kp=%d, levels=%d)",
            n_features,
            n_levels,
        )

    def extract(self, image: np.ndarray) -> FeatureData:
        """Detect and compute ORB features.

        Args:
            image: Input image (BGR or grayscale).

        Returns:
            :class:`FeatureData` with uint8 binary descriptors of dim 32.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        kps, descs = self.detector.detectAndCompute(gray, None)

        if kps is None or len(kps) == 0:
            _logger.warning("No ORB keypoints detected")
            return FeatureData(
                keypoints=np.empty((0, 2), dtype=np.float32),
                descriptors=np.empty((0, 32), dtype=np.uint8),
            )

        coords = np.array([kp.pt for kp in kps], dtype=np.float32)
        scores = np.array([kp.response for kp in kps], dtype=np.float32)

        _logger.debug("Extracted %d ORB keypoints", len(kps))
        return FeatureData(
            keypoints=coords,
            descriptors=descs,
            scores=scores,
        )
