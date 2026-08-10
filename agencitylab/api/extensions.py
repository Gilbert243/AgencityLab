"""Public multiscale, discrete, multivariate, and geometric extension APIs."""

from __future__ import annotations

import numpy as np

from agencitylab.core.multiscale import (
    agencity_spectrum,
    multivariate_agencity,
    optimize_memory_window,
)
from agencitylab.core.validation import validate_positive_scalar, validate_signal

from .compute import compute_agencity

RIEMANNIAN_EXTENSION_STATUS = "experimental: theoretical definition exists; detailed analysis is deferred"


def compute_agencity_spectrum(
    u,
    xi,
    taus,
    *,
    A_ref,
    P_c,
    windows=None,
    return_full: bool = False,
):
    """Compute the time-resolved ``b(t, tau)`` spectrum.

    With ``windows=None`` each scale uses the common convention ``w=tau``.
    Supplying a scalar or sequence keeps ``w`` explicit and independent, as
    defined by Volume 2 of the same Theory of Agencity.
    """
    return agencity_spectrum(
        xi,
        u,
        taus,
        A_ref=A_ref,
        P_c=P_c,
        windows=windows,
        return_full=return_full,
    )


def optimize_agencity_window(
    u,
    xi,
    *,
    tau,
    A_ref,
    P_c,
    candidates=None,
    n_candidates: int = 24,
):
    """Select the CRM width using the theory's Chapter 13 ``Phi2`` criterion."""
    return optimize_memory_window(
        xi,
        u,
        tau=tau,
        A_ref=A_ref,
        P_c=P_c,
        candidates=candidates,
        n_candidates=n_candidates,
    )


def compute_multivariate_agencity(
    u,
    xi,
    *,
    A_ref,
    tau,
    P_c,
    w=None,
    sample_axis: int = 0,
):
    """Compute the theory's Pc-weighted multivariate construction componentwise."""
    return multivariate_agencity(
        xi,
        u,
        A_ref=A_ref,
        tau=tau,
        P_c=P_c,
        w=w,
        sample_axis=sample_axis,
    )


def compute_discrete_agencity(
    u,
    *,
    delta,
    A_ref,
    tau,
    P_c,
    t0: float = 0.0,
    **kwargs,
):
    """Convenience entry point for a uniformly sampled scalar sequence.

    This is not a new physical equation. It constructs ``xi_n=t0+n*delta`` and
    delegates to :func:`compute_agencity`, which already implements the sampled
    derivatives and discrete CRM. The caller may supply ``w`` through ``kwargs``;
    when omitted, the public compute API uses the common ``w=tau`` convention.
    """
    values = validate_signal(u, name="u").ravel()
    delta = validate_positive_scalar(delta, name="delta")
    t0 = float(t0)
    if not np.isfinite(t0):
        raise ValueError("t0 must be finite")
    xi = t0 + delta * np.arange(values.size, dtype=float)
    return compute_agencity(
        u=values,
        xi=xi,
        A_ref=A_ref,
        tau=tau,
        P_c=P_c,
        **kwargs,
    )


def riemannian_extension_status() -> dict:
    """Report the implementation boundary of the Riemannian construction.

    Volume 2 defines covariant activation/activity and an intrinsic dynamic
    intensity, but explicitly defers the detailed analysis. AgencityLab therefore
    does not invent the missing general CRM/vector-state construction.
    """
    return {
        "status": RIEMANNIAN_EXTENSION_STATUS,
        "implemented": False,
        "reason": (
            "the source defines the geometric direction but defers the detailed analysis; "
            "a production pipeline would require additional accepted definitions and tests"
        ),
    }
