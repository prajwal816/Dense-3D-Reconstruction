"""
Context-manager timer for benchmarking pipeline stages.
"""

import time
from typing import Dict, Optional

from .logger import get_logger

_logger = get_logger("dense3d.timer")


class Timer:
    """High-resolution wall-clock timer with named stages.

    Usage::

        timer = Timer()
        with timer("feature_extraction"):
            extract_features(...)
        with timer("matching"):
            match_features(...)
        print(timer.summary())
    """

    def __init__(self) -> None:
        self._records: Dict[str, float] = {}
        self._current_name: Optional[str] = None
        self._start: float = 0.0

    # ── Context manager interface ────────────────────────────────────────
    def __call__(self, name: str) -> "Timer":
        self._current_name = name
        return self

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        _logger.debug("⏱  Started  [%s]", self._current_name)
        return self

    def __exit__(self, *exc) -> None:  # type: ignore[no-untyped-def]
        elapsed = time.perf_counter() - self._start
        name = self._current_name or "unnamed"
        self._records[name] = elapsed
        _logger.info("⏱  Finished [%s] in %.3f s", name, elapsed)
        self._current_name = None

    # ── Query helpers ────────────────────────────────────────────────────
    @property
    def records(self) -> Dict[str, float]:
        """Return ``{stage_name: elapsed_seconds}``."""
        return dict(self._records)

    @property
    def total(self) -> float:
        """Total elapsed time across all recorded stages."""
        return sum(self._records.values())

    def summary(self) -> str:
        """Return a formatted summary table."""
        lines = ["", "╔══════════════════════════════════════════════════╗"]
        lines.append("║           Pipeline Timing Summary               ║")
        lines.append("╠══════════════════════════════════════════════════╣")
        for name, elapsed in self._records.items():
            lines.append(f"║  {name:<32s} {elapsed:>10.3f} s ║")
        lines.append("╠══════════════════════════════════════════════════╣")
        lines.append(f"║  {'TOTAL':<32s} {self.total:>10.3f} s ║")
        lines.append("╚══════════════════════════════════════════════════╝")
        return "\n".join(lines)
