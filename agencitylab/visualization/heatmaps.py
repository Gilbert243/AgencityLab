"""
Heatmap visualization for AgencityLab.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .styles import apply_default_style


def plot_heatmap(matrix, xlabels=None, ylabels=None, show: bool = True, title: Optional[str] = None):
    """
    Plot a generic heatmap.
    """
    apply_default_style()
    import matplotlib.pyplot as plt

    matrix = np.asarray(matrix, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 6))

    # 🔥 amélioration visuelle
    im = ax.imshow(matrix, aspect="auto", origin="lower", cmap="viridis")

    plt.colorbar(im, ax=ax)

    if xlabels is not None:
        ax.set_xticks(np.arange(len(xlabels)))
        ax.set_xticklabels(xlabels, rotation=45, ha="right")

    if ylabels is not None:
        ax.set_yticks(np.arange(len(ylabels)))
        ax.set_yticklabels(ylabels)

    if title is None:
        title = "AgencityLab heatmap"

    ax.set_title(title)
    fig.tight_layout()

    if show:
        try:
            plt.show()
        except Exception:
            pass

    return fig, ax