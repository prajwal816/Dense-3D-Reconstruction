"""
Factory for feature matchers — instantiate from config dict.
"""

from typing import Any, Dict

from .base import FeatureMatcher
from .bf_matcher import BFMatcher
from .flann_matcher import FLANNMatcher
from .superglue import SuperGlueMatcher
from ..utils.logger import get_logger

_logger = get_logger("dense3d.matching.factory")

_REGISTRY: Dict[str, type] = {
    "bf": BFMatcher,
    "flann": FLANNMatcher,
    "superglue": SuperGlueMatcher,
}


def create_matcher(cfg: Dict[str, Any]) -> FeatureMatcher:
    """Create a feature matcher from a configuration dict.

    The dict must contain a ``"type"`` key (``bf`` | ``flann`` |
    ``superglue``).

    An optional ``"feature_type"`` key (e.g. ``"orb"``) is used to
    select the correct norm for BF matching.

    Args:
        cfg: Configuration dictionary.

    Returns:
        An initialised :class:`FeatureMatcher`.
    """
    cfg = dict(cfg)
    match_type = cfg.pop("type", "bf").lower()

    if match_type not in _REGISTRY:
        raise ValueError(
            f"Unknown matcher type '{match_type}'. Available: {list(_REGISTRY.keys())}"
        )

    feature_type = cfg.pop("feature_type", "sift").lower()

    # Merge type-specific sub-config
    sub = cfg.pop(match_type, {})
    params: Dict[str, Any] = {**sub}

    # Forward common keys
    for key in ("ratio_test", "cross_check"):
        if key in cfg:
            params.setdefault(key, cfg[key])

    # Auto-select norm for BF matcher
    if match_type == "bf":
        if feature_type == "orb":
            params.setdefault("norm_type", "hamming")
        else:
            params.setdefault("norm_type", "l2")

    matcher = _REGISTRY[match_type](**params)
    _logger.info("Created matcher: %s (feature_type=%s)", matcher, feature_type)
    return matcher
