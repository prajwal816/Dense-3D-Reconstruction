"""
Abstract base class for feature matchers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass
class MatchResult:
    """Container for pairwise match output.

    Attributes:
        matches_idx: (M, 2) int array — indices into kp1 and kp2.
        matched_kp1: (M, 2) float array — matched coordinates in image 1.
        matched_kp2: (M, 2) float array — matched coordinates in image 2.
        inlier_mask: Optional (M,) bool mask after geometric verification.
        num_inliers: Number of inliers after RANSAC.
        inlier_ratio: Fraction of matches that are inliers.
        homography: Optional 3×3 homography matrix.
        fundamental: Optional 3×3 fundamental matrix.
    """

    matches_idx: np.ndarray
    matched_kp1: np.ndarray
    matched_kp2: np.ndarray
    inlier_mask: Optional[np.ndarray] = None
    num_inliers: int = 0
    inlier_ratio: float = 0.0
    homography: Optional[np.ndarray] = None
    fundamental: Optional[np.ndarray] = None

    @property
    def num_matches(self) -> int:
        return self.matches_idx.shape[0]


class FeatureMatcher(ABC):
    """Interface every feature matcher must implement."""

    name: str = "base"

    @abstractmethod
    def match(
        self,
        kp1: np.ndarray,
        desc1: np.ndarray,
        kp2: np.ndarray,
        desc2: np.ndarray,
    ) -> MatchResult:
        """Match descriptors between two images.

        Args:
            kp1: (N, 2) keypoints from image 1.
            desc1: (N, D) descriptors from image 1.
            kp2: (M, 2) keypoints from image 2.
            desc2: (M, D) descriptors from image 2.

        Returns:
            :class:`MatchResult`.
        """
        ...

    @staticmethod
    def geometric_verification(
        pts1: np.ndarray,
        pts2: np.ndarray,
        method: str = "fundamental",
        ransac_threshold: float = 3.0,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """RANSAC-based geometric verification.

        Args:
            pts1: (N, 2) matched points in image 1.
            pts2: (N, 2) matched points in image 2.
            method: ``"fundamental"`` or ``"homography"``.
            ransac_threshold: Pixel reprojection threshold.

        Returns:
            Tuple of (inlier_mask, matrix) where matrix is F or H.
        """
        import cv2

        if pts1.shape[0] < 8:
            return np.zeros(pts1.shape[0], dtype=bool), None

        if method == "homography":
            mat, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransac_threshold)
        else:
            mat, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, ransac_threshold)

        if mask is None:
            return np.zeros(pts1.shape[0], dtype=bool), None

        return mask.ravel().astype(bool), mat

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} ({self.name})>"
