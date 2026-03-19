"""
Factory for feature extractors — instantiate from config dict.
"""

from typing import Dict, Any

from .base import FeatureExtractor
from .sift import SIFTExtractor
from .orb import ORBExtractor
from .superpoint import SuperPointExtractor
from ..utils.logger import get_logger

_logger = get_logger("dense3d.features.factory")

_REGISTRY: Dict[str, type] = {
    "sift": SIFTExtractor,
    "orb": ORBExtractor,
    "superpoint": SuperPointExtractor,
}


def create_extractor(cfg: Dict[str, Any]) -> FeatureExtractor:
    """Create a feature extractor from a configuration dict.

    The dict must contain a ``"type"`` key (``sift`` | ``orb`` |
    ``superpoint``).  Remaining keys are forwarded to the constructor.

    Args:
        cfg: Configuration dictionary.

    Returns:
        An initialised :class:`FeatureExtractor`.

    Raises:
        ValueError: If the requested type is unknown.
    """
    cfg = dict(cfg)  # shallow copy
    feat_type = cfg.pop("type", "sift").lower()

    if feat_type not in _REGISTRY:
        raise ValueError(
            f"Unknown feature type '{feat_type}'. "
            f"Available: {list(_REGISTRY.keys())}"
        )

    # Merge type-specific sub-config if present
    sub = cfg.pop(feat_type, {})
    params = {**sub, **{k: v for k, v in cfg.items() if k != "max_keypoints"}}

    # Forward max_keypoints under the right constructor arg name
    max_kp = cfg.get("max_keypoints")
    if max_kp is not None:
        if feat_type == "sift":
            params.setdefault("n_features", max_kp)
        elif feat_type == "orb":
            params.setdefault("n_features", max_kp)
        elif feat_type == "superpoint":
            params.setdefault("max_keypoints", max_kp)

    extractor = _REGISTRY[feat_type](**params)
    _logger.info("Created extractor: %s", extractor)
    return extractor
