"""Intrinsic beta-plane visualization for AgencityLab."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_phase_portrait(result, show: bool = True):
    """Plot the intrinsic state trajectory in the complex beta plane."""
    beta = np.asarray(result.beta, dtype=complex)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(beta.real, beta.imag)
    ax.scatter(beta.real[0], beta.imag[0], s=24, label="start")
    ax.set_title("Intrinsic agencity trajectory beta")
    ax.set_xlabel("Re(beta)")
    ax.set_ylabel("Im(beta)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend()
    fig.tight_layout()
    if show:
        plt.show()
    return fig
