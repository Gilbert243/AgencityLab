"""
Logging helpers for AgencityLab.
"""

from __future__ import annotations

import logging
from typing import Optional


def get_logger(name: str = "agencitylab") -> logging.Logger:
    """Return a configured logger instance."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.propagate = False
    return logger


def set_log_level(level: str = "INFO", name: str = "agencitylab") -> logging.Logger:
    """Set the log level of the AgencityLab logger."""
    logger = get_logger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger
