"""
Image-to-signal utilities.

The base implementation avoids forcing image libraries at install time.
If Pillow is installed, it can accept image files; otherwise it works with
array-like grayscale data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _require_pillow():
    """Import Pillow lazily if file-based image loading is requested."""
    try:
        from PIL import Image  # type: ignore
        return Image
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Pillow is required for loading image files. Install AgencityLab with the image extra."
        ) from exc


def image_to_signal(image, mode="raw"):
    
    arr = np.asarray(image, dtype=float)

    if arr.ndim == 3:
        arr = np.mean(arr, axis=-1)

    mode = mode.lower().strip()

    # 🔥 FULL IMAGE
    if mode == "raw":
        return arr  # (H, W)

    # 🔥 PROJECTIONS
    if mode == "row":
        return np.mean(arr, axis=1, keepdims=True)

    if mode == "col":
        return np.mean(arr, axis=0, keepdims=True).T

    # 🔥 GRADIENT (TRÈS IMPORTANT)
    if mode == "gradient":
        gx, gy = np.gradient(arr)
        return np.stack([gx, gy], axis=-1)

    raise ValueError("Unknown mode")