"""
Phase-space visualization for AgencityLab.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .styles import apply_default_style


def plot_phase_portrait(result, show: bool = True, title: Optional[str] = None):
    """
    Plot a phase portrait from the canonical state variables.
    """
    apply_default_style()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(result.X_star, result.A_star)
    axes[0].set_xlabel("X*")
    axes[0].set_ylabel("A*")
    axes[0].set_title("Activation phase portrait")

    axes[1].plot(result.M, result.O)
    axes[1].set_xlabel("M")
    axes[1].set_ylabel("O")
    axes[1].set_title("Memory-organization portrait")

    if title is None:
        title = "AgencityLab phase portrait"

    fig.suptitle(title)
    fig.tight_layout()

    if show:
        plt.show()

    return fig, axes
