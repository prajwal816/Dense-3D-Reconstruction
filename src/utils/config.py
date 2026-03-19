"""
YAML configuration loader with attribute-style access.
"""

import os
import copy
from typing import Any, Dict, Optional

import yaml


class Config(dict):
    """Dictionary subclass that supports attribute-style access.

    Example::

        cfg = Config({"features": {"type": "sift"}})
        assert cfg.features.type == "sift"
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict) and not isinstance(value, Config):
                self[key] = Config(value)

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")

    def __repr__(self) -> str:
        return f"Config({super().__repr__()})"

    def to_dict(self) -> Dict[str, Any]:
        """Recursively convert back to plain dict."""
        out: Dict[str, Any] = {}
        for k, v in self.items():
            out[k] = v.to_dict() if isinstance(v, Config) else v
        return out


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into a copy of *base*."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(
    path: str,
    overrides: Optional[Dict[str, Any]] = None,
) -> Config:
    """Load a YAML config file and optionally merge overrides.

    Args:
        path: Path to the YAML file.
        overrides: Optional dict of values to merge on top.

    Returns:
        A :class:`Config` instance.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    if overrides:
        data = _deep_merge(data, overrides)

    return Config(data)
