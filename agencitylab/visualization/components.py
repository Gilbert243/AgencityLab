"""Component visualization for AgencityLab."""

from __future__ import annotations

import matplotlib.pyplot as plt


def plot_components(result, show: bool = True):
    """Plot reduced kinematics and the CRM structural pair."""
    xi = result.xi
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(xi, result.X_star)
    axes[0].set_title("Reduced activation X*")
    axes[0].set_ylabel("X*")

    axes[1].plot(xi, result.A_star)
    axes[1].set_title("Reduced activity A*")
    axes[1].set_ylabel("A*")

    axes[2].plot(xi, result.M)
    axes[2].set_title("Memory M = CRM[u*]")
    axes[2].set_ylabel("M")

    axes[3].plot(xi, result.O)
    axes[3].set_title("Organisation O = CRM[u*, X*]")
    axes[3].set_ylabel("O")
    coordinate = getattr(result, "coordinate_unit", "") or ""
    axes[3].set_xlabel(f"Coordinate ({coordinate})" if coordinate else "Coordinate")

    fig.tight_layout()
    if show:
        plt.show()
    return fig
