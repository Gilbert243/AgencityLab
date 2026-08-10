"""Causal moving correlation (CRM) for the Theory of Agencity.

The CRM width is ``w > 0``. Volume 2 often uses ``w = tau`` as a convenient
choice but keeps the quantities distinct, especially for window optimisation.
The discrete implementation compares the most recent block ``[t-w, t]`` with
the immediately preceding block ``[t-2w, t-w]``. A zero empirical variance
gives correlation zero by definition. No epsilon is inserted into the Pearson
denominator.

The implementation uses rolling first and second moments for the ordinary
vectorised path. Numerically delicate windows are recomputed with the direct
centred definition. The fallback is a numerical safeguard only: it preserves
the same Pearson coefficient and never changes the physical zero convention.
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


def _condition_for_correlation(values):
    """Scale finite samples before centring to avoid overflow/underflow."""
    values = np.asarray(values, dtype=float)
    scale = float(np.max(np.abs(values)))
    if scale == 0.0:
        return np.zeros_like(values)
    conditioned = values / scale
    conditioned = conditioned - np.mean(conditioned)
    return conditioned


def _pearson_corr(a, b):
    """Pearson correlation with the theory's exact zero-variance convention."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size != b.size:
        raise ValueError("Pearson samples must have the same length")
    if a.size < 2 or np.all(a == a[0]) or np.all(b == b[0]):
        return 0.0

    a0 = _condition_for_correlation(a)
    b0 = _condition_for_correlation(b)
    ss_a = float(np.dot(a0, a0))
    ss_b = float(np.dot(b0, b0))

    # Non-constant floating-point samples can still produce a numerically null
    # sum of squares at extreme dynamic ranges. Long double is a safeguard for
    # that machine case; it does not introduce a physical epsilon threshold.
    if ss_a == 0.0 or ss_b == 0.0:
        a_ld = np.asarray(a, dtype=np.longdouble)
        b_ld = np.asarray(b, dtype=np.longdouble)
        a_scale = np.max(np.abs(a_ld))
        b_scale = np.max(np.abs(b_ld))
        a0_ld = a_ld / a_scale - np.mean(a_ld / a_scale)
        b0_ld = b_ld / b_scale - np.mean(b_ld / b_scale)
        ss_a_ld = np.dot(a0_ld, a0_ld)
        ss_b_ld = np.dot(b0_ld, b0_ld)
        if ss_a_ld == 0 or ss_b_ld == 0:
            return 0.0
        corr = np.dot(a0_ld, b0_ld) / (np.sqrt(ss_a_ld) * np.sqrt(ss_b_ld))
        return float(np.clip(corr, -1.0, 1.0))

    corr = float(np.dot(a0, b0) / (np.sqrt(ss_a) * np.sqrt(ss_b)))
    # Only suppress round-off excursions outside the mathematical Pearson bounds.
    return float(np.clip(corr, -1.0, 1.0))


def _rolling_sum(values, width):
    """Return all width-sized rolling sums in O(N) time and O(N) memory."""
    values = np.asarray(values, dtype=float)
    prefix = np.empty(values.size + 1, dtype=float)
    prefix[0] = 0.0
    np.cumsum(values, dtype=float, out=prefix[1:])
    return prefix[width:] - prefix[:-width]


def _constant_windows(values, width):
    """Detect exactly constant sampled windows without a tolerance."""
    values = np.asarray(values)
    if width == 1:
        return np.ones(values.size, dtype=bool)

    changes = np.not_equal(values[1:], values[:-1]).astype(np.int64, copy=False)
    prefix = np.empty(changes.size + 1, dtype=np.int64)
    prefix[0] = 0
    np.cumsum(changes, dtype=np.int64, out=prefix[1:])
    counts = prefix[width - 1 :] - prefix[: -(width - 1)]
    return counts == 0


def _rolling_pearson(x, y, width, *, auto):
    """Compute adjacent-window Pearson coefficients with stable fallbacks."""
    size = x.size
    out = np.zeros(size, dtype=float)
    if width == 1:
        return out, 0

    valid_count = size - 2 * width + 1

    # For very short windows, subtraction of cumulative moments can lose more
    # information than the direct O(width) centred formula. Since width is
    # bounded here, this remains linear in N while reproducing the definition.
    if width <= 4:
        for offset in range(valid_count):
            recent_start = width + offset
            out[2 * width - 1 + offset] = _pearson_corr(
                x[recent_start : recent_start + width],
                y[offset : offset + width],
            )
        return out, valid_count

    x_work = _condition_for_correlation(x)
    y_work = x_work if auto else _condition_for_correlation(y)

    sum_x_all = _rolling_sum(x_work, width)
    sum_x2_all = _rolling_sum(np.square(x_work), width)
    sum_x = sum_x_all[width:]
    sum_x2 = sum_x2_all[width:]

    if auto:
        sum_y = sum_x_all[:-width]
        sum_y2 = sum_x2_all[:-width]
    else:
        sum_y_all = _rolling_sum(y_work, width)
        sum_y2_all = _rolling_sum(np.square(y_work), width)
        sum_y = sum_y_all[:-width]
        sum_y2 = sum_y2_all[:-width]

    pair_products = x_work[width:] * y_work[:-width]
    sum_xy = _rolling_sum(pair_products, width)

    width_float = float(width)
    ss_x = sum_x2 - (sum_x * sum_x) / width_float
    ss_y = sum_y2 - (sum_y * sum_y) / width_float
    covariance = sum_xy - (sum_x * sum_y) / width_float

    constant_x_all = _constant_windows(x, width)
    constant_y_all = constant_x_all if auto else _constant_windows(y, width)
    nonconstant = (~constant_x_all[width:]) & (~constant_y_all[:-width])

    eps = np.finfo(float).eps
    scale_x = np.maximum(sum_x2, (sum_x * sum_x) / width_float)
    scale_y = np.maximum(sum_y2, (sum_y * sum_y) / width_float)
    stable = (
        nonconstant
        & (ss_x > 256.0 * eps * scale_x)
        & (ss_y > 256.0 * eps * scale_y)
    )

    local = np.zeros(valid_count, dtype=float)
    if np.any(stable):
        denominator = np.sqrt(ss_x[stable]) * np.sqrt(ss_y[stable])
        values = covariance[stable] / denominator
        finite_and_bounded = np.isfinite(values) & (np.abs(values) <= 1.0 + 1e-10)
        stable_positions = np.flatnonzero(stable)
        accepted_positions = stable_positions[finite_and_bounded]
        local[accepted_positions] = np.clip(values[finite_and_bounded], -1.0, 1.0)
        stable[stable_positions[~finite_and_bounded]] = False

    fallback_positions = np.flatnonzero(nonconstant & ~stable)
    for offset in fallback_positions:
        recent_start = width + int(offset)
        previous_start = int(offset)
        local[offset] = _pearson_corr(
            x[recent_start : recent_start + width],
            y[previous_start : previous_start + width],
        )

    out[2 * width - 1 :] = local
    return out, int(fallback_positions.size)


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

    out, fallback_count = _rolling_pearson(x, y, n, auto=other is None)

    if verbose:
        mode = "cross" if other is not None else "auto"
        print(f"[crm] mode={mode}, w={w}, samples={n}, stable_fallbacks={fallback_count}")
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
