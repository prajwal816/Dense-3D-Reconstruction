"""
SuperGlue deep feature matcher wrapper.

Provides a simulated / stub implementation that mirrors the real SuperGlue
interface.  When pretrained weights are available the forward pass runs a
real model; otherwise it falls back to brute-force matching on the
descriptors so the pipeline stays functional.
"""

import numpy as np
import cv2

from .base import FeatureMatcher, MatchResult
from ..utils.logger import get_logger

_logger = get_logger("dense3d.matching.superglue")

try:
    import torch
    import torch.nn as nn

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class SuperGlueMatcher(FeatureMatcher):
    """SuperGlue graph-neural-network matcher.

    Falls back to brute-force L2 matching when weights are not available.

    Args:
        weights_path: Path to ``superglue_outdoor.pth`` or ``superglue_indoor.pth``.
        match_threshold: Confidence threshold for accepting a match.
        sinkhorn_iterations: Number of Sinkhorn iterations.
    """

    name = "superglue"

    def __init__(
        self,
        weights_path: str = "",
        match_threshold: float = 0.2,
        sinkhorn_iterations: int = 20,
    ) -> None:
        self.match_threshold = match_threshold
        self.sinkhorn_iterations = sinkhorn_iterations
        self._using_fallback = True

        if _HAS_TORCH and weights_path:
            import os

            if os.path.isfile(weights_path):
                try:
                    self._device = "cuda" if torch.cuda.is_available() else "cpu"
                    self._weights = torch.load(weights_path, map_location=self._device)
                    self._using_fallback = False
                    _logger.info("Loaded SuperGlue weights from %s", weights_path)
                except Exception as exc:
                    _logger.warning("Failed to load SuperGlue weights: %s", exc)
            else:
                _logger.warning(
                    "SuperGlue weights not found at %s — using BF fallback",
                    weights_path,
                )

        if self._using_fallback:
            self._bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            _logger.info("SuperGlue running in BF-fallback mode")

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

        if self._using_fallback:
            return self._match_fallback(kp1, desc1, kp2, desc2)
        return self._match_deep(kp1, desc1, kp2, desc2)

    # ── Deep path (placeholder — needs full GNN implementation) ──────────
    def _match_deep(
        self,
        kp1: np.ndarray,
        desc1: np.ndarray,
        kp2: np.ndarray,
        desc2: np.ndarray,
    ) -> MatchResult:
        """Forward through the SuperGlue network.

        NOTE: A full implementation requires the GNN architecture with
        self/cross-attention layers and the optimal transport solver.
        This stub delegates to the fallback for now.
        """
        _logger.info("SuperGlue deep matching — delegating to BF (stub)")
        return self._match_fallback(kp1, desc1, kp2, desc2)

    # ── Brute-force fallback ─────────────────────────────────────────────
    def _match_fallback(
        self,
        kp1: np.ndarray,
        desc1: np.ndarray,
        kp2: np.ndarray,
        desc2: np.ndarray,
    ) -> MatchResult:
        d1 = desc1.astype(np.float32)
        d2 = desc2.astype(np.float32)

        raw = self._bf.knnMatch(d1, d2, k=2)
        good = []
        for pair in raw:
            if len(pair) == 2:
                m, n = pair
                if m.distance < 0.75 * n.distance:
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

        # Geometric verification
        mask, F = self.geometric_verification(pts1, pts2, method="fundamental")
        result = MatchResult(
            matches_idx=idx_pairs,
            matched_kp1=pts1,
            matched_kp2=pts2,
            inlier_mask=mask,
            num_inliers=int(mask.sum()) if mask is not None else idx_pairs.shape[0],
            inlier_ratio=(
                float(mask.sum()) / max(idx_pairs.shape[0], 1)
                if mask is not None
                else 1.0
            ),
            fundamental=F,
        )

        _logger.info(
            "SuperGlue (fallback) match: %d raw → %d inliers (%.2f%%)",
            idx_pairs.shape[0],
            result.num_inliers,
            result.inlier_ratio * 100,
        )
        return result
