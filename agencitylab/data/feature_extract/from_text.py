"""
Text-to-signal utilities.

The base implementation uses very simple statistics to remain dependency-free.
"""

from __future__ import annotations

import re
from typing import Iterable, List

import numpy as np


_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)


def text_to_signal(text, mode="embedding"):
    
    tokens = _WORD_RE.findall(text)

    if not tokens:
        return np.zeros((0, 1))

    mode = mode.lower().strip()

    # 🔥 VECTORIAL REPRESENTATION
    if mode == "embedding":
        # fallback simple embedding (sans dépendance)
        return np.array([[len(w), sum(map(ord, w)) % 100] for w in tokens], dtype=float)

    # 🔥 SEQUENCE STRUCTURE
    if mode == "sequence":
        return np.array([len(w) for w in tokens])[:, None]

    raise ValueError("Unknown mode")