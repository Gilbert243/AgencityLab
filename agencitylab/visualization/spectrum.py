"""
Spectrum visualization for AgencityLab.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_spectrum(result, show: bool = True):

    xi = result.xi
    b = result.b

    if len(b) < 2:
        raise ValueError("Signal too short for spectrum.")

    dt = np.mean(np.diff(xi))

    freq = np.fft.rfftfreq(len(b), dt)
    spectrum = np.abs(np.fft.rfft(b))

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(freq, spectrum)
    ax.set_title("Spectrum of b(t)")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("|FFT(b)|")

    plt.tight_layout()

    if show:
        plt.show()

    return fig