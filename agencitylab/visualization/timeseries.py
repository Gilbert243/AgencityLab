"""
Time-series visualization for AgencityLab.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .styles import apply_default_style


def plot_timeseries(result, show: bool = True, ax=None, title: Optional[str] = None):
    """
    Plot the canonical Agencity quantities as aligned time series.

    Returns
    -------
    (fig, axes)
    """
    apply_default_style()
    import matplotlib.pyplot as plt

    xi = np.asarray(result.xi, dtype=float)

    if ax is None:
        fig, axes = plt.subplots(4, 2, figsize=(12, 10), sharex=True)
    else:
        fig = ax.figure
        axes = np.asarray(ax).reshape(4, 2)

    series = [
        ("u", np.asarray(result.u, dtype=float)),
        ("u*", np.asarray(result.u_star, dtype=float)),
        ("X*", np.asarray(result.X_star, dtype=float)),
        ("A*", np.asarray(result.A_star, dtype=float)),
        ("M", np.asarray(result.M, dtype=float)),
        ("O", np.asarray(result.O, dtype=float)),
        ("beta", np.asarray(result.beta, dtype=float)),
        ("b", np.asarray(result.b, dtype=float)),
    ]

    for i, (axis, (name, values)) in enumerate(zip(axes.ravel(), series)):
        axis.plot(xi, values)
        axis.set_title(name)
        axis.set_ylabel(name)
        axis.margins(x=0)

        # xlabel uniquement dernière ligne
        if i >= 6:
            axis.set_xlabel("xi")

    if title is None:
        title = "AgencityLab time series"

    fig.suptitle(title)
    fig.tight_layout()

    if show:
        try:
            plt.show()
        except Exception:
            pass

    return fig, axes