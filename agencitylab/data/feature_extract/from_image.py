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


def image_to_signal(image: Any, mode: str = "row_mean") -> np.ndarray:
    """
    Convert an image-like object into a one-dimensional signal.

    Parameters
    ----------
    image:
        Either a NumPy array or a path to an image file.
    mode:
        Reduction strategy:
        - row_mean
        - col_mean
        - flatten_mean
    """
    if isinstance(image, (str, Path)):
        Image = _require_pillow()
        img = Image.open(image).convert("L")
        arr = np.asarray(img, dtype=float)
    else:
        arr = np.asarray(image, dtype=float)

    if arr.ndim == 3:
        # Convert RGB-like tensors to grayscale by averaging channels.
        arr = np.mean(arr, axis=-1)

    if arr.ndim != 2:
        raise ValueError("image must be a 2D grayscale array or an RGB-like array.")

    mode = mode.lower().strip()

    if mode == "row_mean":
        return np.mean(arr, axis=1)
    if mode == "col_mean":
        return np.mean(arr, axis=0)
    if mode == "flatten_mean":
        return np.asarray([float(np.mean(arr))], dtype=float)

    raise ValueError("Unknown image-to-signal mode.")
