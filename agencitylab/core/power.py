"""Characteristic-power utilities for AgencityLab."""

from __future__ import annotations

import numpy as np

from agencitylab.constants.characteristic_powers import (
    power_context_from_metadata,
    resolve_characteristic_power,
)
from .validation import validate_nonnegative_scalar, validate_positive_scalar


def characteristic_power(
    value=None,
    *,
    nominal_power=None,
    system=None,
    domain=None,
    reference_energy=None,
    inertia=None,
    tau=None,
    metadata=None,
    A_ref=None,
    default=None,
    verbose=False,
):
    """Resolve canonical characteristic power ``P_c``.

    The accepted numerical domain is finite ``P_c >= 0``. Supported physical
    routes are, in order: an explicit ``value``; an already specified metadata
    value; nominal sustainable power; documented reference energy divided by its
    structural time; ``inertia * reference_energy / tau``; or a deliberately
    registered physical convention. ``A_ref`` is accepted only for backwards-
    compatible call signatures and is never used to derive power.
    """
    del A_ref
    context = power_context_from_metadata(metadata)
    if system is None:
        system = context.get("system")
    if domain is None:
        domain = context.get("domain")

    if value is not None and str(value).strip().lower() not in {"auto", "canonical", "default"}:
        out = validate_nonnegative_scalar(value, name="P_c")
    elif context.get("Pc") is not None:
        out = validate_nonnegative_scalar(context["Pc"], name="P_c")
    elif nominal_power is not None:
        out = validate_nonnegative_scalar(nominal_power, name="nominal_power")
    elif reference_energy is not None and tau is not None:
        energy = validate_positive_scalar(reference_energy, name="reference_energy")
        tau_value = validate_positive_scalar(tau, name="tau")
        if inertia is not None:
            energy *= validate_positive_scalar(inertia, name="inertia")
        out = energy / tau_value
    else:
        out = resolve_characteristic_power(
            system=system,
            domain=domain,
            Pc="auto",
            default=default,
        )

    if verbose:
        print(f"[power] resolved P_c={out}")
    return float(out)


def estimate_characteristic_power(
    signal,
    *,
    tau,
    method: str = "rms",
    scale=None,
    A_ref=None,
    verbose: bool = False,
):
    """Experimental signal-derived power estimate; not canonical physics."""
    x = np.asarray(signal, dtype=float)
    if not np.all(np.isfinite(x)):
        raise ValueError("signal must contain only finite values")
    tau = validate_positive_scalar(tau, name="tau")

    if A_ref is not None:
        scale = validate_positive_scalar(A_ref, name="A_ref")
    elif scale is None:
        if method == "rms":
            scale = float(np.sqrt(np.mean(np.square(x))))
        elif method == "variance":
            scale = float(np.sqrt(np.var(x)))
        elif method == "amplitude":
            scale = float(np.max(x) - np.min(x))
        else:
            raise ValueError("Unknown power estimation method")
    scale = validate_positive_scalar(scale, name="scale")
    out = (scale**2) / tau
    if verbose:
        print(f"[power] experimental estimate P_c={out}")
    return float(out)
