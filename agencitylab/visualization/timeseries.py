"""
Time series visualization for AgencityLab.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_timeseries(result, show: bool = True):

    xi = result.xi
    T = result.u
    b = result.b

    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # --------------------------------
    # SIGNAL
    # --------------------------------
    ax[0].plot(xi, T)
    ax[0].set_title("Signal u(ξ)")
    ax[0].set_ylabel("Amplitude")

    # --------------------------------
    # BETA
    # --------------------------------
    ax[1].plot(xi, result.beta)
    ax[1].set_title("β(t) (dimensionless)")
    ax[1].set_ylabel("β (nat)")

    # --------------------------------
    # AGENCITY
    # --------------------------------
    ax[2].plot(xi, b)
    ax[2].set_title("Agencity b(t)")
    ax[2].set_ylabel("Bz (W·nat)")
    ax[2].set_xlabel("ξ")

    plt.tight_layout()

    if show:
        plt.show()

    return fig