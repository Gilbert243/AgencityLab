"""
Causal relationship visualization between M and O.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_causal_mo(result, lag_max: int = 50, show: bool = True):
    """
    Cross-correlation analysis between M and O.

    Positive lag: M → O
    Negative lag: O → M
    """

    M = np.asarray(result.M)
    O = np.asarray(result.O)

    lags = np.arange(-lag_max, lag_max + 1)
    corr = np.zeros_like(lags, dtype=float)

    for i, lag in enumerate(lags):
        if lag < 0:
            corr[i] = np.corrcoef(M[:lag], O[-lag:])[0, 1]
        elif lag > 0:
            corr[i] = np.corrcoef(M[lag:], O[:-lag])[0, 1]
        else:
            corr[i] = np.corrcoef(M, O)[0, 1]

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(lags, corr)
    ax.axvline(0, linestyle="--")

    ax.set_title("Causal relation M ↔ O")
    ax.set_xlabel("Lag")
    ax.set_ylabel("Correlation")

    plt.tight_layout()

    if show:
        plt.show()

    return fig