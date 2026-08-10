"""Heatmap visualization of dimensionless Agencity components."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_heatmap(result, show: bool = True):
    """Plot the dimensionless ``M, O, D, S, J`` trajectories as a time heatmap."""
    xi = np.asarray(result.xi, dtype=float)
    matrix = np.vstack([result.M, result.O, result.D, result.S, result.J]).astype(float)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    image = ax.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        extent=[xi[0], xi[-1], -0.5, 4.5],
    )
    ax.set_yticks(range(5), labels=["M", "O", "D", "S", "J"])
    ax.set_xlabel("Coordinate")
    ax.set_title("Canonical dimensionless components")
    fig.colorbar(image, ax=ax, label="value")
    fig.tight_layout()
    if show:
        plt.show()
    return fig
