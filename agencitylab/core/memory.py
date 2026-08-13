"""Canonical memory operator ``M = CRM[u*]``."""

from __future__ import annotations

from .crm import causal_moving_correlation


def memory(
    observable_signal,
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
    """Compute memory directly from the reduced observable, without saturation."""
    crm = causal_moving_correlation(
        observable_signal,
        tau,
        axis=axis,
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


def memory_from_signal(*args, **kwargs):
    """Pipeline readability alias."""
    return memory(*args, **kwargs)
