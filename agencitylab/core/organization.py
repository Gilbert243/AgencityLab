"""Canonical organisation operator ``O = CRM[u*, X*]``."""

from __future__ import annotations

from .crm import causal_moving_correlation
from .validation import validate_signal


def organization(
    observable_signal,
    activation_signal,
    tau,
    *,
    axis,
    window=None,
    activity_factor="auto",
    mechanism=None,
    domain=None,
    system_type=None,
    return_correlation: bool = False,
    verbose: bool = False,
):
    """Compute cross-CRM between the reduced observable and reduced activation."""
    u_star = validate_signal(observable_signal, name="observable_signal").ravel()
    X_star = validate_signal(activation_signal, name="activation_signal").ravel()
    if len(u_star) != len(X_star):
        raise ValueError("observable_signal and activation_signal must have the same length")

    crm = causal_moving_correlation(
        u_star,
        tau,
        axis=axis,
        other=X_star,
        window=window,
        activity_factor=activity_factor,
        mechanism=mechanism,
        domain=domain,
        system_type=system_type,
        verbose=verbose,
    )
    if return_correlation:
        return crm, crm.copy()
    return crm


def organization_from_signal(*args, **kwargs):
    """Pipeline readability alias."""
    return organization(*args, **kwargs)
