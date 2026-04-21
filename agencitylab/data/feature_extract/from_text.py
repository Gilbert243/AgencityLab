"""
Text-to-signal utilities.

The base implementation uses very simple statistics to remain dependency-free.
"""

from __future__ import annotations

import re
from typing import Iterable, List

import numpy as np


_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)


def text_to_signal(text: str, mode: str = "sentence_length") -> np.ndarray:
    """
    Convert text into a simple one-dimensional signal.

    Supported modes
    --------------
    - sentence_length: number of tokens per sentence
    - word_length: lengths of individual tokens
    - character_count: cumulative character count per sentence
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    mode = mode.lower().strip()

    if mode == "word_length":
        tokens = _WORD_RE.findall(text)
        return np.asarray([len(token) for token in tokens], dtype=float)

    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    if not sentences:
        return np.asarray([], dtype=float)

    if mode == "sentence_length":
        return np.asarray([len(_WORD_RE.findall(sentence)) for sentence in sentences], dtype=float)

    if mode == "character_count":
        return np.asarray([len(sentence) for sentence in sentences], dtype=float)

    raise ValueError("Unknown text-to-signal mode.")
