"""Research-facing scientific visualizations for AgencityLab.

The functions in this module display canonical quantities and already-computed
analysis diagnostics. They never recompute or reinterpret the Theory of
Agencity equations.
"""

from __future__ import annotations

import numpy as np


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("matplotlib is required; install agencitylab[viz]") from exc
    return plt


def _coordinate_label(result) -> str:
    unit = getattr(result, "coordinate_unit", "") or ""
    return f"Coordinate ({unit})" if unit else "Coordinate"


def _b_label(result) -> str:
    unit = getattr(result, "b_unit", "") or ""
    return f"|b| ({unit})" if unit else "|b|"


def plot_scientific_overview(result, *, show: bool = True):
    """Plot the canonical signal-to-flux pipeline in six aligned panels."""
    plt = _plt()
    xi = np.asarray(result.xi, dtype=float)
    fig, axes = plt.subplots(6, 1, figsize=(11, 13), sharex=True)

    axes[0].plot(xi, result.u)
    axes[0].set_ylabel("u")
    axes[0].set_title("Observable")

    axes[1].plot(xi, result.X_star, label="X*")
    axes[1].plot(xi, result.A_star, label="A*")
    axes[1].set_ylabel("Reduced kinematics")
    axes[1].legend()

    axes[2].plot(xi, result.M, label="M")
    axes[2].plot(xi, result.O, label="O")
    axes[2].set_ylabel("CRM")
    axes[2].legend()

    axes[3].plot(xi, result.D, label="D")
    axes[3].plot(xi, result.S, label="S")
    axes[3].plot(xi, result.J, label="J")
    axes[3].set_ylabel("D, S, J")
    axes[3].legend()

    axes[4].plot(xi, np.abs(result.beta), label="|beta|")
    axes[4].plot(xi, np.abs(result.b), label="|b|")
    axes[4].set_ylabel("Magnitude")
    axes[4].legend()

    theta = np.asarray(result.theta, dtype=float)
    valid = np.asarray(result.S, dtype=float) > 0.0
    axes[5].plot(xi[valid], theta[valid], label="Theta (S > 0)")
    axes[5].set_ylabel("Theta (rad)")
    axes[5].set_xlabel(_coordinate_label(result))
    axes[5].legend()

    w = getattr(result, "memory_window", None)
    fig.suptitle(
        "Agencity scientific overview"
        + (f" — tau={result.tau:g}, w={w:g}" if w is not None else f" — tau={result.tau:g}")
    )
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_beta_geometry(result, *, analysis=None, show: bool = True):
    """Plot the intrinsic complex beta trajectory and its signed curvature diagnostic."""
    plt = _plt()
    beta = np.asarray(result.beta, dtype=complex)
    xi = np.asarray(result.xi, dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(beta.real, beta.imag)
    axes[0].scatter(beta.real[0], beta.imag[0], s=24, label="start")
    axes[0].set_xlabel("Re(beta)")
    axes[0].set_ylabel("Im(beta)")
    axes[0].set_title("Intrinsic beta trajectory")
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].legend()

    curvature = None
    if analysis is not None:
        curvature = analysis.get("geometry", {}).get("curvature")
    if curvature is None:
        from agencitylab.analysis.geometry import curvature as beta_curvature

        curvature = beta_curvature(beta, xi=xi)
    curvature = np.asarray(curvature, dtype=float)
    if curvature.size == xi.size:
        x_curv = xi
    else:
        # Structured reports exclude the finite-record CRM warmup interval.
        x_curv = xi[-curvature.size :] if curvature.size else np.asarray([], dtype=float)
    axes[1].plot(x_curv, curvature)
    axes[1].axhline(0.0, linewidth=1.0)
    axes[1].set_xlabel(_coordinate_label(result))
    axes[1].set_ylabel("kappa")
    axes[1].set_title("Signed curvature of beta")

    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_scientific_diagnostics(result, *, analysis, show: bool = True):
    """Plot structural and real-agencity diagnostics without inventing thresholds."""
    if analysis is None:
        raise ValueError("analysis is required for diagnostic visualization")
    plt = _plt()
    xi = np.asarray(result.xi, dtype=float)
    sigma = np.asarray(
        analysis.get("coherence", {}).get("structural_orientation", {}).get("sigma_theta", []),
        dtype=float,
    )
    real_diag = analysis.get("real_agencity", {})

    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(xi, result.S)
    axes[0].axhline(0.0, linewidth=1.0)
    axes[0].set_ylabel("S")
    axes[0].set_title("Structural intensity")

    if sigma.size == xi.size:
        axes[1].plot(xi, sigma)
    axes[1].set_ylabel("Sigma_Theta")
    axes[1].set_title("Local angular variance")
    theta_threshold = real_diag.get("theta_variance_threshold")
    if theta_threshold is not None:
        axes[1].axhline(float(theta_threshold), linestyle="--", label="contextual threshold")
        axes[1].legend()

    axes[2].plot(xi, np.abs(result.b))
    axes[2].set_ylabel(_b_label(result))
    axes[2].set_title("Observable agencity flux magnitude")
    b_threshold = real_diag.get("b_threshold")
    if b_threshold is not None:
        axes[2].axhline(float(b_threshold), linestyle="--", label="contextual threshold")
        axes[2].legend()

    axes[3].plot(xi, result.D, label="D")
    axes[3].plot(xi, result.S, label="S")
    axes[3].plot(xi, result.J, label="J")
    axes[3].axhline(0.0, linewidth=1.0)
    axes[3].set_ylabel("D, S, J")
    axes[3].set_xlabel(_coordinate_label(result))
    axes[3].set_title("Dynamic-structural contrast")
    axes[3].legend()

    status = real_diag.get("status", "undetermined")
    fig.suptitle(f"Scientific diagnostics — real-agencity: {status}")
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_multiscale_spectrum(spectrum, *, show: bool = True):
    """Plot the time-resolved magnitude of the theoretical ``b(t, tau)`` spectrum."""
    plt = _plt()
    b = np.asarray(spectrum["b"], dtype=complex)
    tau = np.asarray(spectrum["tau"], dtype=float)
    responses = spectrum.get("responses")
    if responses:
        xi = np.asarray(responses[0]["xi"], dtype=float)
    else:
        xi = np.arange(b.shape[1], dtype=float)

    fig, ax = plt.subplots(figsize=(10, 5))
    image = ax.pcolormesh(xi, tau, np.abs(b), shading="auto")
    ax.set_xlabel("Coordinate")
    ax.set_ylabel("tau")
    ax.set_title("Multiscale agencity magnitude |b(t, tau)|")
    fig.colorbar(image, ax=ax, label="|b|")
    fig.tight_layout()
    if show:
        plt.show()
    return fig
