"""
Matplotlib styling helpers for AgencityLab.
"""

from __future__ import annotations


def set_matplotlib_style(style: str = "default") -> None:
    """
    Apply a named matplotlib style.
    """
    import matplotlib.pyplot as plt
    plt.style.use(style)


def apply_default_style() -> None:
    """
    Apply the canonical AgencityLab plotting style.

    Clean, readable, publication-friendly.
    """
    import matplotlib.pyplot as plt

    plt.style.use("default")
    plt.rcParams.update(
        {
            # Resolution
            "figure.dpi": 120,
            "savefig.dpi": 200,

            # Layout
            "figure.figsize": (10, 6),

            # Grid
            "axes.grid": True,
            "grid.alpha": 0.3,

            # Lines
            "lines.linewidth": 2.0,

            # Fonts
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,

            # Legend
            "legend.frameon": False,

            # Clean axes
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )