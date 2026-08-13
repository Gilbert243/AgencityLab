"""Multiscale Agencity signature diagnostics.

This module is analytical, not canonical. It fits a log-log relation across
explicitly supplied scales and never modifies tau, beta, or the canonical
single-scale computation.
"""

from __future__ import annotations

import numpy as np


def _linear_regression(x, y):
    matrix = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(matrix, y, rcond=None)[0]
    return float(slope), float(intercept)


def _compute_r2(x, y, slope, intercept):
    predicted = slope * x + intercept
    residual = float(np.sum((y - predicted) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    if total == 0.0:
        return 1.0 if residual == 0.0 else 0.0
    return float(1.0 - residual / total)


def agencity_signature(
    tau,
    beta_mean,
    *,
    log_base=np.e,
    return_fit=True,
    slope_threshold: float | None = None,
    verbose=False,
):
    """Fit the diagnostic scaling relation ``log(beta_mean) ~ alpha log(tau)``.

    Only strictly positive finite pairs are admissible because the logarithm is
    part of this diagnostic regression. Invalid values are not replaced by an
    epsilon. ``slope_threshold`` is optional and contextual; without it, no
    qualitative regime label is inferred from the fitted slope.
    """
    tau = np.asarray(tau, dtype=float)
    beta_mean = np.asarray(beta_mean, dtype=float)
    if tau.ndim != 1 or beta_mean.ndim != 1 or tau.size != beta_mean.size:
        raise ValueError("tau and beta_mean must be one-dimensional with equal length")
    valid = np.isfinite(tau) & np.isfinite(beta_mean) & (tau > 0.0) & (beta_mean > 0.0)
    if np.count_nonzero(valid) < 2:
        raise ValueError("at least two strictly positive finite scale pairs are required")
    tau_valid = tau[valid]
    beta_valid = beta_mean[valid]

    base = float(log_base)
    if not np.isfinite(base) or base <= 0.0 or base == 1.0:
        raise ValueError("log_base must be positive, finite, and different from 1")
    lt = np.log(tau_valid) / np.log(base)
    lb = np.log(beta_valid) / np.log(base)
    slope, intercept = _linear_regression(lt, lb)
    r2 = _compute_r2(lt, lb, slope, intercept)

    if slope_threshold is None:
        regime = "undetermined"
        interpretation_status = "no slope threshold configured"
    else:
        threshold = float(slope_threshold)
        if not np.isfinite(threshold) or threshold < 0.0:
            raise ValueError("slope_threshold must be finite and non-negative")
        if slope > threshold:
            regime = "amplifying"
        elif slope < -threshold:
            regime = "dissipative"
        else:
            regime = "approximately_scale_invariant"
        interpretation_status = "diagnostic threshold configured"

    fit = slope * lt + intercept if return_fit else None
    if verbose:
        print(f"[signature] slope={slope:.6g}, r2={r2:.6g}, regime={regime}")

    return {
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "tau_log": lt,
        "beta_log": lb,
        "fit": fit,
        "regime": regime,
        "interpretation_status": interpretation_status,
        "slope_threshold": slope_threshold,
        "n_valid_scales": int(np.count_nonzero(valid)),
        "status": "multiscale diagnostic; not a canonical equation",
    }
