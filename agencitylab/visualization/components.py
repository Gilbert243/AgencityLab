"""
Component visualization for AgencityLab.
"""

from __future__ import annotations

import matplotlib.pyplot as plt


def plot_components(result, show: bool = True):

    xi = result.xi

    fig, ax = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    ax[0].plot(xi, result.X_star)
    ax[0].set_title("Activation X(t)")

    ax[1].plot(xi, result.A_star)
    ax[1].set_title("Activity A(t)")

    ax[2].plot(xi, result.M)
    ax[2].set_title("Memory M(t)")

    ax[3].plot(xi, result.O)
    ax[3].set_title("Organization O(t)")
    ax[3].set_xlabel("ξ")

    plt.tight_layout()

    if show:
        plt.show()

    return fig