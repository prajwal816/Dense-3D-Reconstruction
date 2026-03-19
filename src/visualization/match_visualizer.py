"""
Feature match visualization — side-by-side match drawing.
"""

import os
from typing import Optional

import cv2
import numpy as np

from ..utils.logger import get_logger
from ..utils.io_utils import ensure_dir

_logger = get_logger("dense3d.viz.matches")


def draw_matches(
    img1: np.ndarray,
    kp1: np.ndarray,
    img2: np.ndarray,
    kp2: np.ndarray,
    matches_idx: np.ndarray,
    inlier_mask: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
    max_draw: int = 200,
    show: bool = False,
) -> np.ndarray:
    """Draw feature matches side-by-side.

    Inliers are drawn in green, outliers in red.

    Args:
        img1: First image (BGR).
        kp1: (N, 2) keypoint coordinates.
        img2: Second image (BGR).
        kp2: (M, 2) keypoint coordinates.
        matches_idx: (K, 2) matched index pairs.
        inlier_mask: Optional (K,) boolean mask.
        output_path: Save visualisation to this path.
        max_draw: Limit on number of drawn matches.
        show: Display with cv2.imshow.

    Returns:
        Concatenated visualisation image.
    """
    if len(img1.shape) == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if len(img2.shape) == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h = max(h1, h2)
    canvas = np.zeros((h, w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1:] = img2

    n_draw = min(matches_idx.shape[0], max_draw)
    idx = np.random.choice(matches_idx.shape[0], n_draw, replace=False) if n_draw < matches_idx.shape[0] else np.arange(n_draw)

    for i in idx:
        i1, i2 = matches_idx[i]
        pt1 = tuple(kp1[i1].astype(int))
        pt2 = (int(kp2[i2][0]) + w1, int(kp2[i2][1]))

        if inlier_mask is not None:
            color = (0, 255, 0) if inlier_mask[i] else (0, 0, 255)
        else:
            color = (0, 255, 0)

        cv2.circle(canvas, pt1, 3, color, -1)
        cv2.circle(canvas, pt2, 3, color, -1)
        cv2.line(canvas, pt1, pt2, color, 1, cv2.LINE_AA)

    # Stats text
    n_matches = matches_idx.shape[0]
    n_inliers = int(inlier_mask.sum()) if inlier_mask is not None else n_matches
    text = f"Matches: {n_matches}  |  Inliers: {n_inliers}  |  Ratio: {n_inliers / max(n_matches, 1):.1%}"
    cv2.putText(canvas, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    _logger.info("Match visualisation: %s", text)

    if output_path:
        ensure_dir(os.path.dirname(output_path))
        cv2.imwrite(output_path, canvas)
        _logger.info("Match image saved → %s", output_path)

    if show:
        cv2.imshow("Feature Matches", canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return canvas
