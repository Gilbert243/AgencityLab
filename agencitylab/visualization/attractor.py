"""
Dynamic attractor visualization.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_attractor(result, dims: tuple = ("beta", "b"), show: bool = True):
    """
    Plot attractor in 2D or 3D.

    dims options:
    - ("beta", "b")
    - ("X_star", "A_star")
    """

    d1 = getattr(result, dims[0])
    d2 = getattr(result, dims[1])

    x = np.asarray(d1)
    y = np.asarray(d2)

    if x.ndim > 1:
        x = np.linalg.norm(x, axis=1)

    if y.ndim > 1:
        y = np.linalg.norm(y, axis=1)

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot(x, y, linewidth=0.8)
    ax.set_title(f"Attractor ({dims[0]} vs {dims[1]})")
    ax.set_xlabel(dims[0])
    ax.set_ylabel(dims[1])

    plt.tight_layout()

    if show:
        plt.show()

    return fig


def plot_attractor_3d(result, show: bool = True):
    """
    3D attractor: (X, A, beta)
    """

    from mpl_toolkits.mplot3d import Axes3D  # noqa

    X = result.X_star
    A = result.A_star
    beta = result.beta

    x = np.linalg.norm(X, axis=1) if X.ndim > 1 else X
    y = np.linalg.norm(A, axis=1) if A.ndim > 1 else A
    z = beta

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(x, y, z, linewidth=0.6)

    ax.set_title("3D Attractor (X, A, β)")
    ax.set_xlabel("X")
    ax.set_ylabel("A")
    ax.set_zlabel("β")

    plt.tight_layout()

    if show:
        plt.show()

    return fig