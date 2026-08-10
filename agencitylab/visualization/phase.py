"""
Phase portrait visualization for AgencityLab.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_phase_portrait(result, show: bool = True):

    beta = result.beta
    b = result.b

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot(beta, b)
    ax.set_title("Phase portrait (β vs b)")
    ax.set_xlabel("β (nat)")
    ax.set_ylabel("b (Bz)")

    plt.tight_layout()

    if show:
        plt.show()

    return fig