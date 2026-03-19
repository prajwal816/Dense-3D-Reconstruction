"""
Colored logger with file + console handlers.
"""

import logging
import os
import sys
from typing import Optional

try:
    import colorlog

    _HAS_COLORLOG = True
except ImportError:
    _HAS_COLORLOG = False


def get_logger(
    name: str = "dense3d",
    level: str = "INFO",
    log_file: Optional[str] = None,
    colored: bool = True,
) -> logging.Logger:
    """Create or retrieve a named logger with console and optional file output.

    Args:
        name: Logger name.
        level: Logging level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to a log file.
        colored: Whether to use colored console output.

    Returns:
        Configured ``logging.Logger`` instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"

    # ── Console handler ──────────────────────────────────────────────────
    if colored and _HAS_COLORLOG:
        console_fmt = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt=date_fmt,
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )
    else:
        console_fmt = logging.Formatter(fmt, datefmt=date_fmt)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    # ── File handler ─────────────────────────────────────────────────────
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))
        logger.addHandler(file_handler)

    return logger
