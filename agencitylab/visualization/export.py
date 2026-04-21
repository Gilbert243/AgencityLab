"""
Figure export helpers for AgencityLab.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def save_figure(fig, path, *, dpi: int = 200, bbox_inches: str = "tight"):
    """
    Save a Matplotlib figure to disk.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
    return path


def export_figure(fig, path, *, dpi: int = 200, bbox_inches: str = "tight"):
    """
    Alias kept for readability in higher-level APIs.
    """
    return save_figure(fig, path, dpi=dpi, bbox_inches=bbox_inches)
