"""
signature.py

Agencity scaling signature analysis.

This module extracts scale-invariant properties from the agencity
multi-scale response using log-log regression.

Core idea:
    log(beta_mean) ~ alpha * log(tau)

Where:
    alpha = scaling exponent (signature of the system)
"""

from __future__ import annotations
import numpy as np

from agencitylab.core.safeguards import EPS, replace_non_finite


# =========================================================
# INTERNAL UTILITIES
# =========================================================

def _safe_log(x, eps=EPS):
    """Safe logarithm avoiding zero/negative values."""
    x = np.asarray(x, dtype=float)
    x = replace_non_finite(x, eps)
    x = np.maximum(x, eps)
    return np.log(x)


def _linear_regression(x, y):
    """Least-squares linear regression."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]

    return float(slope), float(intercept)


def _compute_r2(x, y, slope, intercept):
    """Coefficient of determination (R²)."""
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    if ss_tot < EPS:
        return 0.0

    return float(1 - ss_res / ss_tot)


# =========================================================
# MAIN API
# =========================================================

def agencity_signature(
    tau,
    beta_mean,
    *,
    log_base=np.e,
    return_fit=True,
    verbose=False,
):
    """
    Compute the scaling signature of agencity.

    Parameters
    ----------
    tau : array-like
        Characteristic scales
    beta_mean : array-like
        Mean agencity magnitude per scale
    log_base : float
        Logarithm base (default: natural)
    return_fit : bool
        Whether to return fitted curve
    verbose : bool
        Print debug information

    Returns
    -------
    dict with:
        slope       : scaling exponent α
        intercept   : regression intercept
        r2          : goodness of fit
        tau_log     : log(tau)
        beta_log    : log(beta_mean)
        fit         : optional fitted curve
        regime      : qualitative interpretation
    """

    tau = np.asarray(tau, dtype=float)
    beta_mean = np.asarray(beta_mean, dtype=float)

    # --- safety ---
    tau = replace_non_finite(tau, EPS)
    beta_mean = replace_non_finite(beta_mean, EPS)

    tau = np.maximum(tau, EPS)
    beta_mean = np.maximum(beta_mean, EPS)

    # --- logs ---
    if log_base == np.e:
        lt = _safe_log(tau)
        lb = _safe_log(beta_mean)
    else:
        lt = _safe_log(tau) / np.log(log_base)
        lb = _safe_log(beta_mean) / np.log(log_base)

    # --- regression ---
    slope, intercept = _linear_regression(lt, lb)

    # --- goodness ---
    r2 = _compute_r2(lt, lb, slope, intercept)

    # --- regime classification ---
    if slope > 0.1:
        regime = "amplifying"
    elif slope < -0.1:
        regime = "dissipative"
    else:
        regime = "scale-invariant"

    # --- fit curve ---
    fit = None
    if return_fit:
        fit = slope * lt + intercept

    if verbose:
        print("[signature] ---")
        print(f"[signature] slope (alpha): {slope:.6f}")
        print(f"[signature] intercept   : {intercept:.6f}")
        print(f"[signature] R²          : {r2:.6f}")
        print(f"[signature] regime      : {regime}")

    return {
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "tau_log": lt,
        "beta_log": lb,
        "fit": fit,
        "regime": regime,
    }