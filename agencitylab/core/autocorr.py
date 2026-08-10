"""
autocorr.py

Autocorrelation utilities for AgencityLab.

Purpose
-------
Used for:
    - exploratory temporal analysis,
    - empirical diagnostics,
    - multiscale inspection,
    - coherence analysis,
    - heuristic tau exploration.

IMPORTANT
---------
In stabilized Agencity theory:

    tau is structural
    and independent of u(t)

Therefore:
    autocorrelation-based tau estimation
    is NOT canonical.

Autocorrelation remains useful for:
    - empirical workflows,
    - unknown systems,
    - exploratory diagnostics,
    - comparative studies,
    - noise characterization.

Physical interpretation
-----------------------
Autocorrelation measures temporal persistence
of observable structures.

It does NOT define the structural time
of the containing system.
"""

from __future__ import annotations

import numpy as np

from .safeguards import (
    EPS,
    replace_non_finite,
)

from .validation import (
    validate_signal,
)


# ============================================================
# INTERNAL HELPERS
# ============================================================

def _safe_energy(x):
    """
    Stable signal energy.
    """

    return float(
        np.dot(x, x)
    )


def _safe_normalize(
    acf,
    eps=EPS,
):
    """
    Stable autocorrelation normalization.
    """

    ref = max(
        float(acf[0]),
        eps,
    )

    return acf / ref


def _lag_axis(n):
    """
    Generate lag axis.
    """

    return np.arange(
        n,
        dtype=float,
    )


# ============================================================
# AUTOCORRELATION
# ============================================================

def autocorrelation(
    signal,
    *,
    demean=True,
    normalize=True,
    unbiased=False,
    max_lag=None,
    replace_nan=True,
    return_energy=False,
    verbose=False,
):
    """
    Compute non-negative-lag autocorrelation.

    Parameters
    ----------
    signal :
        Input signal.

    demean :
        Remove mean before correlation.

    normalize :
        Normalize by lag-0 value.

    unbiased :
        Apply unbiased lag normalization.

    max_lag :
        Optional maximum lag.

    replace_nan :
        Replace invalid values.

    return_energy :
        Return signal energy.

    Returns
    -------
    lags :
        Lag axis.

    acf :
        Autocorrelation function.

    optionally:
        (lags, acf, energy)
    """

    # ========================================================
    # VALIDATION
    # ========================================================

    x = validate_signal(
        signal,
        min_length=2,
        name="signal",
    ).ravel()

    x = np.asarray(
        x,
        dtype=float,
    )

    n = x.size

    # ========================================================
    # CLEANUP
    # ========================================================

    if replace_nan:

        x = replace_non_finite(
            x,
            0.0,
        )

    # ========================================================
    # DEMEAN
    # ========================================================

    if demean:

        if verbose:

            print(
                "[autocorr] "
                "Removing mean"
            )

        x = x - np.mean(x)

    # ========================================================
    # ENERGY
    # ========================================================

    energy = _safe_energy(x)

    # ========================================================
    # DEGENERATE SIGNAL
    # ========================================================

    if energy <= EPS:

        if verbose:

            print(
                "[autocorr] "
                "Near-zero energy signal"
            )

        lags = _lag_axis(n)

        acf = np.zeros(
            n,
            dtype=float,
        )

        acf[0] = 1.0

        if return_energy:
            return lags, acf, energy

        return lags, acf

    # ========================================================
    # CORRELATION
    # ========================================================

    if verbose:

        print(
            "[autocorr] "
            "Computing autocorrelation"
        )

    corr = np.correlate(
        x,
        x,
        mode="full",
    )

    acf = corr[n - 1:]

    # ========================================================
    # UNBIASED NORMALIZATION
    # ========================================================

    if unbiased:

        if verbose:

            print(
                "[autocorr] "
                "Applying unbiased correction"
            )

        weights = np.arange(
            n,
            0,
            -1,
            dtype=float,
        )

        acf = acf / weights

    # ========================================================
    # NORMALIZATION
    # ========================================================

    if normalize:

        if verbose:

            print(
                "[autocorr] "
                "Normalizing"
            )

        acf = _safe_normalize(
            acf,
            eps=EPS,
        )

    # ========================================================
    # MAX LAG
    # ========================================================

    if max_lag is not None:

        max_lag = int(max_lag)

        if max_lag <= 0:

            raise ValueError(
                "max_lag must be positive"
            )

        acf = acf[:max_lag]

    # ========================================================
    # CLEANUP
    # ========================================================

    acf = replace_non_finite(
        acf,
        0.0,
    )

    acf = np.asarray(
        acf,
        dtype=float,
    )

    lags = _lag_axis(
        len(acf)
    )

    # ========================================================
    # DIAGNOSTICS
    # ========================================================

    if verbose:

        print(
            "[autocorr] "
            f"length={len(acf)}"
        )

        print(
            "[autocorr] "
            f"energy={energy:.6f}"
        )

        print(
            "[autocorr] "
            f"acf[0]={acf[0]:.6f}"
        )

        print(
            "[autocorr] "
            f"acf_min={np.min(acf):.6f}"
        )

        print(
            "[autocorr] "
            f"acf_max={np.max(acf):.6f}"
        )

    # ========================================================
    # RETURN
    # ========================================================

    if return_energy:
        return lags, acf, energy

    return lags, acf


# ============================================================
# NORMALIZED AUTOCORRELATION
# ============================================================

def normalized_autocorrelation(
    signal,
    *,
    unbiased=False,
    max_lag=None,
    verbose=False,
):
    """
    Convenience wrapper returning only
    normalized autocorrelation.
    """

    return autocorrelation(
        signal,
        demean=True,
        normalize=True,
        unbiased=unbiased,
        max_lag=max_lag,
        verbose=verbose,
    )[1]