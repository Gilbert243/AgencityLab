"""
Spectrum visualization for AgencityLab.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from agencitylab.analysis.spectrum import agencity_spectrum
from .styles import apply_default_style


def plot_spectrum(result, show: bool = True, title: Optional[str] = None):
    """
    Plot the amplitude spectrum of the Agencity observable b.
    """
    apply_default_style()
    import matplotlib.pyplot as plt

    spectrum = agencity_spectrum(result.b, xi=result.xi)
    freq = spectrum["frequency"]
    amp = spectrum["amplitude"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(freq, amp)

    # 🔥 amélioration clé
    ax.set_xscale("log")

    ax.set_xlabel("frequency")
    ax.set_ylabel("amplitude")

    if title is None:
        title = "AgencityLab spectrum"

    ax.set_title(title)
    fig.tight_layout()

    if show:
        try:
            plt.show()
        except Exception:
            pass

    return fig, ax