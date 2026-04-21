"""
Component visualization for AgencityLab.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .styles import apply_default_style


def plot_components(result, show: bool = True, title: Optional[str] = None):
    """
    Plot the core components of the Agencity decomposition.
    """
    apply_default_style()
    import matplotlib.pyplot as plt

    xi = np.asarray(result.xi, dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(xi, result.u, label="u")
    axes[0].plot(xi, result.u_star, label="u*")
    axes[0].set_ylabel("signal")
    axes[0].legend()

    axes[1].plot(xi, result.X_star, label="X*")
    axes[1].plot(xi, result.A_star, label="A*")
    axes[1].set_ylabel("activation")
    axes[1].legend()

    axes[2].plot(xi, result.M, label="M")
    axes[2].plot(xi, result.O, label="O")
    axes[2].plot(xi, result.beta, label="beta")
    axes[2].plot(xi, result.b, label="b")
    axes[2].set_ylabel("agencity")
    axes[2].set_xlabel("xi")
    axes[2].legend()

    if title is None:
        title = "AgencityLab components"

    fig.suptitle(title)
    fig.tight_layout()

    if show:
        plt.show()

    return fig, axes
