"""
Feature extraction utilities for multi-domain inputs.
"""

from .embeddings import build_embedding_signal
from .from_graph import graph_to_signal
from .from_image import image_to_signal
from .from_text import text_to_signal
from .signal_builder import build_signal_from_features

__all__ = [
    "build_embedding_signal",
    "build_signal_from_features",
    "graph_to_signal",
    "image_to_signal",
    "text_to_signal",
]
