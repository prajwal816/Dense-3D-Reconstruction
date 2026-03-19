"""
FLANN-based feature matcher (fast approximate nearest-neighbour).
"""

import cv2
import numpy as np

from .base import FeatureMatcher, MatchResult
from ..utils.logger import get_logger

_logger = get_logger("dense3d.matching.flann")


class FLANNMatcher(FeatureMatcher):
    """FLANN (Fast Library for Approximate Nearest Neighbors) matcher.

    Best suited for float descriptors (SIFT / SuperPoint).

    Args:
        trees: Number of KD-trees for the index.
        checks: Number of recursive traversals at search time.
        ratio_test: Lowe's ratio test threshold.
        verify_geom: Run RANSAC geometric verification.
    """

    name = "flann"

    def __init__(
        self,
        trees: int = 5,
        checks: int = 50,
        ratio_test: float = 0.75,
        verify_geom: bool = True,
    ) -> None:
        self.ratio_test = ratio_test
        self.verify_geom = verify_geom

        index_params = dict(algorithm=1, trees=trees)  # FLANN_INDEX_KDTREE
        search_params = dict(checks=checks)
        self._matcher = cv2.FlannBasedMatcher(index_params, search_params)

        _logger.info(
            "Initialized FLANN matcher (trees=%d, checks=%d, ratio=%.2f)",
            trees,
            checks,
            ratio_test,
        )

    def match(
        self,
        kp1: np.ndarray,
        desc1: np.ndarray,
        kp2: np.ndarray,
        desc2: np.ndarray,
    ) -> MatchResult:
        if desc1.shape[0] < 2 or desc2.shape[0] < 2:
            return MatchResult(
                matches_idx=np.empty((0, 2), dtype=np.int32),
                matched_kp1=np.empty((0, 2), dtype=np.float32),
                matched_kp2=np.empty((0, 2), dtype=np.float32),
            )

        # FLANN requires float32
        d1 = desc1.astype(np.float32)
        d2 = desc2.astype(np.float32)

        raw = self._matcher.knnMatch(d1, d2, k=2)
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
            "FLANN match: %d raw → %d inliers (ratio=%.2f%%)",
            idx_pairs.shape[0],
            result.num_inliers,
            result.inlier_ratio * 100,
        )
        return result
