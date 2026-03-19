"""
Brute-force feature matcher with ratio test.
"""

import cv2
import numpy as np

from .base import FeatureMatcher, MatchResult
from ..utils.logger import get_logger

_logger = get_logger("dense3d.matching.bf")


class BFMatcher(FeatureMatcher):
    """OpenCV Brute-Force matcher with Lowe's ratio test.

    Args:
        ratio_test: Ratio threshold for the second-best match filter.
        cross_check: Whether to enable cross-check (disables ratio test).
        norm_type: ``"l2"`` for float descriptors (SIFT), ``"hamming"`` for
            binary descriptors (ORB).
        verify_geom: Whether to apply RANSAC geometric verification.
    """

    name = "bf"

    def __init__(
        self,
        ratio_test: float = 0.75,
        cross_check: bool = False,
        norm_type: str = "l2",
        verify_geom: bool = True,
    ) -> None:
        self.ratio_test = ratio_test
        self.cross_check = cross_check
        self.verify_geom = verify_geom

        norm = cv2.NORM_HAMMING if norm_type == "hamming" else cv2.NORM_L2
        self._matcher = cv2.BFMatcher(norm, crossCheck=cross_check)

        _logger.info(
            "Initialized BF matcher (ratio=%.2f, cross_check=%s, norm=%s)",
            ratio_test,
            cross_check,
            norm_type,
        )

    def match(
        self,
        kp1: np.ndarray,
        desc1: np.ndarray,
        kp2: np.ndarray,
        desc2: np.ndarray,
    ) -> MatchResult:
        if desc1.shape[0] == 0 or desc2.shape[0] == 0:
            return MatchResult(
                matches_idx=np.empty((0, 2), dtype=np.int32),
                matched_kp1=np.empty((0, 2), dtype=np.float32),
                matched_kp2=np.empty((0, 2), dtype=np.float32),
            )

        if self.cross_check:
            raw = self._matcher.match(desc1, desc2)
            idx_pairs = np.array([[m.queryIdx, m.trainIdx] for m in raw], dtype=np.int32)
        else:
            raw = self._matcher.knnMatch(desc1, desc2, k=2)
            good = []
            for pair in raw:
                if len(pair) == 2:
                    m, n = pair
                    if m.distance < self.ratio_test * n.distance:
                        good.append((m.queryIdx, m.trainIdx))
            if not good:
                return MatchResult(
                    matches_idx=np.empty((0, 2), dtype=np.int32),
                    matched_kp1=np.empty((0, 2), dtype=np.float32),
                    matched_kp2=np.empty((0, 2), dtype=np.float32),
                )
            idx_pairs = np.array(good, dtype=np.int32)

        pts1 = kp1[idx_pairs[:, 0]]
        pts2 = kp2[idx_pairs[:, 1]]

        result = MatchResult(
            matches_idx=idx_pairs,
            matched_kp1=pts1,
            matched_kp2=pts2,
        )

        if self.verify_geom and idx_pairs.shape[0] >= 8:
            mask, F = self.geometric_verification(pts1, pts2, method="fundamental")
            result.inlier_mask = mask
            result.num_inliers = int(mask.sum())
            result.inlier_ratio = result.num_inliers / max(idx_pairs.shape[0], 1)
            result.fundamental = F
        else:
            result.num_inliers = idx_pairs.shape[0]
            result.inlier_ratio = 1.0

        _logger.info(
            "BF match: %d raw → %d inliers (ratio=%.2f%%)",
            idx_pairs.shape[0],
            result.num_inliers,
            result.inlier_ratio * 100,
        )
        return result
