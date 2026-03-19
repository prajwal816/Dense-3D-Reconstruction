"""
SuperPoint deep feature extractor wrapper.

This provides a simulated / stub implementation that mirrors the real
SuperPoint architecture interface.  When the actual pretrained weights
are available the ``_forward`` method will run a real PyTorch model;
otherwise it gracefully falls back to a high-quality SIFT proxy so the
rest of the pipeline can be exercised end-to-end.
"""

import numpy as np
import cv2

from .base import FeatureExtractor, FeatureData
from ..utils.logger import get_logger

_logger = get_logger("dense3d.features.superpoint")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# ═════════════════════════════════════════════════════════════════════════════
# Lightweight SuperPoint network definition (for reference / weight loading)
# ═════════════════════════════════════════════════════════════════════════════
if _HAS_TORCH:

    class _SuperPointNet(nn.Module):
        """Minimal SuperPoint-style encoder → detector + descriptor heads."""

        def __init__(self) -> None:
            super().__init__()
            # Shared encoder
            self.relu = nn.ReLU(inplace=True)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

            c1, c2, c3, c4 = 64, 64, 128, 128
            self.conv1a = nn.Conv2d(1, c1, 3, padding=1)
            self.conv1b = nn.Conv2d(c1, c1, 3, padding=1)
            self.conv2a = nn.Conv2d(c1, c2, 3, padding=1)
            self.conv2b = nn.Conv2d(c2, c2, 3, padding=1)
            self.conv3a = nn.Conv2d(c2, c3, 3, padding=1)
            self.conv3b = nn.Conv2d(c3, c3, 3, padding=1)
            self.conv4a = nn.Conv2d(c3, c4, 3, padding=1)
            self.conv4b = nn.Conv2d(c4, c4, 3, padding=1)

            # Detector head
            self.convPa = nn.Conv2d(c4, 256, 3, padding=1)
            self.convPb = nn.Conv2d(256, 65, 1)

            # Descriptor head
            self.convDa = nn.Conv2d(c4, 256, 3, padding=1)
            self.convDb = nn.Conv2d(256, 256, 1)

        def forward(self, x: "torch.Tensor"):
            # Shared encoder
            x = self.relu(self.conv1a(x))
            x = self.relu(self.conv1b(x))
            x = self.pool(x)
            x = self.relu(self.conv2a(x))
            x = self.relu(self.conv2b(x))
            x = self.pool(x)
            x = self.relu(self.conv3a(x))
            x = self.relu(self.conv3b(x))
            x = self.pool(x)
            x = self.relu(self.conv4a(x))
            x = self.relu(self.conv4b(x))

            # Detector
            cP = self.relu(self.convPa(x))
            semi = self.convPb(cP)  # (B, 65, H/8, W/8)

            # Descriptor
            cD = self.relu(self.convDa(x))
            desc = self.convDb(cD)  # (B, 256, H/8, W/8)
            desc = F.normalize(desc, p=2, dim=1)

            return semi, desc


class SuperPointExtractor(FeatureExtractor):
    """SuperPoint deep feature extractor.

    If PyTorch weights are available they are loaded; otherwise the
    extractor falls back to a SIFT-based proxy that keeps the rest of
    the pipeline functional.

    Args:
        weights_path: Path to ``superpoint_v1.pth``.
        nms_radius: Non-maximum-suppression radius in pixels.
        keypoint_threshold: Detection confidence threshold.
        max_keypoints: Cap on the number of returned keypoints.
    """

    name = "superpoint"

    def __init__(
        self,
        weights_path: str = "",
        nms_radius: int = 4,
        keypoint_threshold: float = 0.005,
        max_keypoints: int = 4096,
    ) -> None:
        self.nms_radius = nms_radius
        self.keypoint_threshold = keypoint_threshold
        self.max_keypoints = max_keypoints
        self._model = None
        self._device = "cpu"
        self._using_fallback = True

        if _HAS_TORCH and weights_path:
            try:
                import os
                if os.path.isfile(weights_path):
                    self._device = "cuda" if torch.cuda.is_available() else "cpu"
                    self._model = _SuperPointNet().to(self._device)
                    self._model.load_state_dict(torch.load(weights_path, map_location=self._device))
                    self._model.eval()
                    self._using_fallback = False
                    _logger.info(
                        "Loaded SuperPoint weights from %s (device=%s)",
                        weights_path,
                        self._device,
                    )
                else:
                    _logger.warning(
                        "SuperPoint weights not found at %s — using SIFT fallback",
                        weights_path,
                    )
            except Exception as exc:
                _logger.warning("Failed to load SuperPoint model: %s — using SIFT fallback", exc)

        if self._using_fallback:
            self._sift = cv2.SIFT_create(nfeatures=max_keypoints)
            _logger.info(
                "SuperPoint running in SIFT-fallback mode (max_kp=%d)",
                max_keypoints,
            )

    # ── Public API ───────────────────────────────────────────────────────
    def extract(self, image: np.ndarray) -> FeatureData:
        if self._using_fallback:
            return self._extract_fallback(image)
        return self._extract_deep(image)

    # ── Deep path ────────────────────────────────────────────────────────
    def _extract_deep(self, image: np.ndarray) -> FeatureData:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        h, w = gray.shape
        inp = torch.from_numpy(gray.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
        inp = inp.to(self._device)

        with torch.no_grad():
            semi, coarse_desc = self._model(inp)

        # Decode keypoints from semi-dense heatmap
        dense = torch.exp(semi)
        dense = dense[:, :-1, :, :]  # Remove dustbin
        heatmap = dense.squeeze(0)
        hc, wc = heatmap.shape[1], heatmap.shape[2]
        heatmap = heatmap.permute(1, 2, 0).reshape(hc, wc, 8, 8)
        heatmap = heatmap.permute(0, 2, 1, 3).reshape(hc * 8, wc * 8)
        heatmap = heatmap[:h, :w]

        # Threshold + NMS
        coords = (heatmap > self.keypoint_threshold).nonzero(as_tuple=False)
        scores_t = heatmap[coords[:, 0], coords[:, 1]]

        if coords.shape[0] == 0:
            return FeatureData(
                keypoints=np.empty((0, 2), dtype=np.float32),
                descriptors=np.empty((0, 256), dtype=np.float32),
            )

        # Simple grid NMS
        kps_np = coords.cpu().numpy().astype(np.float32)[:, ::-1]  # (x, y)
        scores_np = scores_t.cpu().numpy()

        if kps_np.shape[0] > self.max_keypoints:
            top_idx = np.argsort(-scores_np)[: self.max_keypoints]
            kps_np = kps_np[top_idx]
            scores_np = scores_np[top_idx]

        # Sample descriptors via bilinear interpolation
        desc_map = coarse_desc.squeeze(0)  # (256, H/8, W/8)
        pts_norm = torch.from_numpy(kps_np.copy()).float().to(self._device)
        pts_norm[:, 0] = (pts_norm[:, 0] / (w / 2.0)) - 1.0
        pts_norm[:, 1] = (pts_norm[:, 1] / (h / 2.0)) - 1.0
        grid = pts_norm.unsqueeze(0).unsqueeze(0)
        descs = F.grid_sample(desc_map.unsqueeze(0), grid, align_corners=True)
        descs = descs.squeeze().T
        descs = F.normalize(descs, p=2, dim=1).cpu().numpy()

        return FeatureData(
            keypoints=kps_np,
            descriptors=descs.astype(np.float32),
            scores=scores_np,
        )

    # ── SIFT fallback ────────────────────────────────────────────────────
    def _extract_fallback(self, image: np.ndarray) -> FeatureData:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        kps, descs = self._sift.detectAndCompute(gray, None)

        if kps is None or len(kps) == 0:
            return FeatureData(
                keypoints=np.empty((0, 2), dtype=np.float32),
                descriptors=np.empty((0, 128), dtype=np.float32),
            )

        coords = np.array([kp.pt for kp in kps], dtype=np.float32)
        scores = np.array([kp.response for kp in kps], dtype=np.float32)

        # Normalize descriptors to unit length (mimic SuperPoint)
        if descs is not None:
            norms = np.linalg.norm(descs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            descs = (descs / norms).astype(np.float32)

        _logger.debug("SuperPoint fallback: extracted %d SIFT keypoints", len(kps))
        return FeatureData(keypoints=coords, descriptors=descs, scores=scores)
