"""Causal moving correlation (CRM) for the Theory of Agencity.

The CRM width is ``w > 0``. Volume 2 often uses ``w = tau`` as a convenient
choice but keeps the quantities distinct, especially for window optimisation.
The discrete implementation compares the most recent block ``[t-w, t]`` with
the immediately preceding block ``[t-2w, t-w]``. A zero empirical variance
gives correlation zero by definition. No epsilon is inserted into the Pearson
denominator.
"""

from __future__ import annotations

import numpy as np

from .validation import validate_axis, validate_positive_scalar, validate_signal


def _uniform_step(axis):
    axis = validate_axis(axis)
    diffs = np.diff(axis)
    step = float(diffs[0])
    tolerance = np.finfo(float).eps * max(1.0, abs(step)) * 64.0
    if not np.allclose(diffs, step, rtol=1e-10, atol=tolerance):
        raise ValueError("discrete CRM requires uniformly sampled coordinates")
    return step


def _window_to_samples(window, axis):
    window = validate_positive_scalar(window, name="window")
    step = _uniform_step(axis)
    n = int(round(window / step))
    if n < 1:
        raise ValueError("CRM window is smaller than one sampling interval")

    represented = n * step
    tolerance = max(np.finfo(float).eps * max(1.0, abs(window)) * 128.0, abs(step) * 1e-9)
    if not np.isclose(represented, window, rtol=1e-9, atol=tolerance):
        raise ValueError("CRM window must be an integer multiple of the sampling interval")
    return n


def _pearson_corr(a, b):
    """Pearson correlation with the theory's exact zero-variance convention."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a0 = a - np.mean(a)
    b0 = b - np.mean(b)
    ss_a = float(np.dot(a0, a0))
    ss_b = float(np.dot(b0, b0))

    if ss_a == 0.0 or ss_b == 0.0:
        return 0.0

    corr = float(np.dot(a0, b0) / np.sqrt(ss_a * ss_b))
    # Only suppress round-off excursions outside the mathematical Pearson bounds.
    return float(np.clip(corr, -1.0, 1.0))


def causal_moving_correlation(
    signal,
    tau,
    *,
    axis,
    other=None,
    window=None,
    activity_factor="auto",
    mechanism=None,
    domain=None,
    system_type=None,
    force_compressed=False,
    verbose=False,
):
    """Compute auto- or cross-CRM at explicit ``tau`` and optional ``w``.

    ``other=None`` computes ``CRM[signal]``. Supplying ``other`` computes the
    cross-CRM whose recent block comes from ``signal`` and whose preceding block
    comes from ``other``. When ``window`` is omitted, this implementation uses
    the common convention ``w=tau``; an explicit positive window is preserved.

    Legacy compression/activity arguments are accepted only to reject their use;
    they never alter the CRM width.
    """
    del mechanism, domain, system_type
    if force_compressed or activity_factor not in {None, "auto"}:
        raise ValueError("CRM compression/activity factors are not part of the accepted theory")

    x = validate_signal(signal, name="signal").ravel()
    axis = validate_axis(axis, expected_length=len(x), name="axis")
    tau = validate_positive_scalar(tau, name="tau")
    w = tau if window is None else validate_positive_scalar(window, name="window")

    y = x if other is None else validate_signal(other, name="other").ravel()
    if len(y) != len(x):
        raise ValueError("signal and other must have the same length")

    n = _window_to_samples(w, axis)
    if len(x) < 2 * n:
        raise ValueError("signal too short for two CRM windows")

    out = np.zeros(len(x), dtype=float)
    for end in range(2 * n - 1, len(x)):
        recent = x[end - n + 1 : end + 1]
        previous = y[end - 2 * n + 1 : end - n + 1]
        out[end] = _pearson_corr(recent, previous)

    if verbose:
        mode = "cross" if other is not None else "auto"
        print(f"[crm] mode={mode}, w={w}, samples={n}")
    return out


def crm_tau(signal, tau, *, axis, other=None, verbose=False, **kwargs):
    """Compatibility helper that deliberately applies the ``w=tau`` convention."""
    if "window" in kwargs and kwargs["window"] is not None:
        if float(kwargs["window"]) != float(tau):
            raise ValueError("crm_tau is specifically the w=tau convenience helper")
    return causal_moving_correlation(
        signal,
        tau,
        axis=axis,
        other=other,
        window=tau,
        verbose=verbose,
    )
