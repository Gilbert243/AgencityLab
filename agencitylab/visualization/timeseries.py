"""Scientific time-series visualization for AgencityLab."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_timeseries(result, show: bool = True):
    """Plot ``u``, intrinsic ``beta`` and observable ``b`` without discarding phase."""
    xi = np.asarray(result.xi, dtype=float)
    beta = np.asarray(result.beta, dtype=complex)
    b = np.asarray(result.b, dtype=complex)
    coordinate = getattr(result, "coordinate_unit", "") or ""
    b_unit = getattr(result, "b_unit", "") or ""

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(xi, result.u)
    axes[0].set_title("Observable u")
    axes[0].set_ylabel(getattr(result, "unit", "") or "u")

    axes[1].plot(xi, beta.real, label="Re(beta)")
    axes[1].plot(xi, beta.imag, label="Im(beta)")
    axes[1].plot(xi, np.abs(beta), label="|beta|", linestyle="--")
    axes[1].set_title("Intrinsic agencity state beta")
    axes[1].set_ylabel("dimensionless")
    axes[1].legend()

    axes[2].plot(xi, b.real, label="Re(b)")
    axes[2].plot(xi, b.imag, label="Im(b)")
    axes[2].plot(xi, np.abs(b), label="|b|", linestyle="--")
    axes[2].set_title("Observable agencity flux b")
    axes[2].set_ylabel(b_unit or "b")
    axes[2].set_xlabel(f"Coordinate ({coordinate})" if coordinate else "Coordinate")
    axes[2].legend()

    fig.tight_layout()
    if show:
        plt.show()
    return fig
