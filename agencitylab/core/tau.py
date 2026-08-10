"""
tau.py

Characteristic time utilities for AgencityLab.

IMPORTANT
---------
In the stabilized Agencity theory:

    tau is NOT estimated from the observable signal u(t).

tau is a structural property of the containing system.

This module therefore provides:
    - canonical tau resolution,
    - physical helpers,
    - optional experimental heuristics.

The old signal-based estimation remains available
for exploratory workflows only.
"""

from __future__ import annotations

import numpy as np

from .autocorr import autocorrelation

from .safeguards import (
    EPS,
    ensure_positive,
)

from .validation import validate_axis

from agencitylab.constants.characteristic_times import (
    resolve_characteristic_time,
)


# ============================================================
# INTERNAL HELPERS
# ============================================================

def _median_step(axis):

    axis = validate_axis(axis)

    diffs = np.diff(axis)

    diffs = diffs[
        np.isfinite(diffs)
        & (np.abs(diffs) > EPS)
    ]

    if diffs.size == 0:
        return 1.0

    return float(
        np.median(np.abs(diffs))
    )


def _interpolate_threshold_crossing(
    lags,
    values,
    threshold,
):

    for i in range(1, len(values)):

        if values[i] <= threshold <= values[i - 1]:

            x0, x1 = lags[i - 1], lags[i]

            y0, y1 = values[i - 1], values[i]

            if abs(y1 - y0) <= EPS:
                return float(x1)

            alpha = (
                (threshold - y0)
                / (y1 - y0)
            )

            return float(
                x0 + alpha * (x1 - x0)
            )

    return None


# ============================================================
# CANONICAL API
# ============================================================

def characteristic_time(
    *,
    tau="auto",
    system=None,
    domain=None,
    verbose=False,
):
    """
    Resolve canonical characteristic time.

    Parameters
    ----------
    tau : float or "auto"

    system : str
        System identifier.

    domain : str
        Physical domain.

    Returns
    -------
    tau : float
    """

    tau_value = resolve_characteristic_time(
        tau=tau,
        system=system,
        domain=domain,
    )

    tau_value = ensure_positive(
        tau_value
    )

    if verbose:
        print(f"[tau] resolved tau={tau_value}")

    return tau_value


# ============================================================
# OPTIONAL HEURISTIC ESTIMATION
# ============================================================

def estimate_tau(
    activation_signal,
    *,
    axis=None,
    threshold: float = 0.5,
    fallback: str = "first_minimum",
    min_lag: int = 1,
    verbose: bool = False,
):
    """
    Experimental heuristic estimation of tau.

    WARNING
    -------
    This function is NOT part of the canonical theory.

    It estimates a temporal scale from the observable dynamics.
    """

    if verbose:
        print("[tau] heuristic estimation")

    lags, acf = autocorrelation(
        activation_signal,
        demean=True,
        normalize=True,
    )

    min_lag = max(
        1,
        int(min_lag),
    )

    crossing = _interpolate_threshold_crossing(
        lags[min_lag:],
        acf[min_lag:],
        threshold,
    )

    if crossing is None:

        if fallback == "first_minimum":

            idx = None

            for i in range(
                min_lag + 1,
                len(acf) - 1,
            ):

                if (
                    acf[i] <= acf[i - 1]
                    and acf[i] <= acf[i + 1]
                ):
                    idx = i
                    break

            if idx is None:
                idx = max(
                    min_lag,
                    len(acf) // 4,
                )

            crossing = float(idx)

        else:

            crossing = float(
                max(
                    min_lag,
                    len(acf) // 4,
                )
            )

    if axis is None:
        return float(crossing)

    step = _median_step(axis)

    tau_value = float(crossing * step)

    return tau_value