"""Frequency-spectrum visualization for AgencityLab."""

from __future__ import annotations

import matplotlib.pyplot as plt

from agencitylab.analysis.spectrum import agencity_spectrum


def plot_spectrum(result, *, component: str = "magnitude", show: bool = True):
    """Plot a descriptive frequency spectrum of one explicit component of ``b``."""
    spectrum = agencity_spectrum(result.b, result.xi, component=component)
    if not spectrum:
        raise ValueError("signal is too short for a frequency spectrum")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(spectrum["frequency"], spectrum["amplitude"])
    ax.set_title(f"Frequency spectrum of b — {component}")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("FFT amplitude")
    fig.tight_layout()
    if show:
        plt.show()
    return fig
