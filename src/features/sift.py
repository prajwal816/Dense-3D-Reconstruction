"""
SIFT feature extractor (OpenCV).
"""

import cv2
import numpy as np

from .base import FeatureExtractor, FeatureData
from ..utils.logger import get_logger

_logger = get_logger("dense3d.features.sift")


class SIFTExtractor(FeatureExtractor):
    """Scale-Invariant Feature Transform extractor.

    Args:
        n_features: Maximum number of keypoints to retain.
        n_octave_layers: Number of layers in each octave.
        contrast_threshold: Filter low-contrast keypoints.
        edge_threshold: Filter edge-like keypoints.
        sigma: Gaussian sigma for the first octave.
    """

    name = "sift"

    def __init__(
        self,
        n_features: int = 8192,
        n_octave_layers: int = 3,
        contrast_threshold: float = 0.04,
        edge_threshold: float = 10.0,
        sigma: float = 1.6,
    ) -> None:
        self.detector = cv2.SIFT_create(
            nfeatures=n_features,
            nOctaveLayers=n_octave_layers,
            contrastThreshold=contrast_threshold,
            edgeThreshold=edge_threshold,
            sigma=sigma,
        )
        self._n_features = n_features
        _logger.info(
            "Initialized SIFT extractor (max_kp=%d, sigma=%.2f)",
            n_features,
            sigma,
        )

    def extract(self, image: np.ndarray) -> FeatureData:
        """Detect and compute SIFT features.

        Args:
            image: Input image (BGR or grayscale).

        Returns:
            :class:`FeatureData` with float32 descriptors of dim 128.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        kps, descs = self.detector.detectAndCompute(gray, None)

        if kps is None or len(kps) == 0:
            _logger.warning("No SIFT keypoints detected")
            return FeatureData(
                keypoints=np.empty((0, 2), dtype=np.float32),
                descriptors=np.empty((0, 128), dtype=np.float32),
            )

        coords = np.array([kp.pt for kp in kps], dtype=np.float32)
        scores = np.array([kp.response for kp in kps], dtype=np.float32)

        _logger.debug("Extracted %d SIFT keypoints", len(kps))
        return FeatureData(
            keypoints=coords,
            descriptors=descs.astype(np.float32),
            scores=scores,
        )
