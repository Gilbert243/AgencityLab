"""Observable normalization for AgencityLab.

Canonical normalization is exactly ``u* = u / A_ref``. ``A_ref`` is a fixed
physical reference amplitude and is never inferred from the observed signal.
Statistical normalizations remain available only as explicitly non-canonical helpers.
"""

from __future__ import annotations

import numpy as np

from agencitylab.constants.reference_amplitudes import (
    reference_context_from_metadata,
    resolve_reference_amplitude,
)
from .validation import validate_positive_scalar, validate_signal


def center_signal(signal, *, axis: int = 0):
    """Remove the mean (non-canonical preprocessing helper)."""
    arr = np.asarray(signal, dtype=float)
    return arr - np.mean(arr, axis=axis, keepdims=True)


def compute_reference_scale(signal, *, method: str = "std", axis: int = 0, eps=None):
    """Estimate a scale from a signal for experimental workflows only."""
    del eps  # compatibility argument; never injected into a physical scale
    arr = validate_signal(signal)
    key = str(method).strip().lower()

    if key == "std":
        scale = np.std(arr, axis=axis, keepdims=True)
    elif key == "mad":
        med = np.median(arr, axis=axis, keepdims=True)
        scale = np.median(np.abs(arr - med), axis=axis, keepdims=True)
    elif key == "range":
        scale = np.max(arr, axis=axis, keepdims=True) - np.min(arr, axis=axis, keepdims=True)
    else:
        raise ValueError(f"Unknown reference scale method '{method}'")

    if np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("estimated reference scale must be strictly positive")
    return scale


def normalize_signal(
    signal,
    *,
    A_ref=None,
    unit=None,
    observable_kind=None,
    domain=None,
    metadata=None,
    method: str = "canonical",
    center: bool = False,
    axis: int = 0,
    clip=None,
    replace_nan=False,
    verbose: bool = False,
):
    """Normalize an observable.

    In canonical mode the only operation is ``u_star = u / A_ref``. Centering,
    clipping, replacement of invalid samples, z-score and min-max transforms are
    explicitly non-canonical and cannot be enabled through canonical mode.
    """
    arr = np.asarray(signal, dtype=float)
    if replace_nan:
        arr = np.where(np.isfinite(arr), arr, 0.0)
    arr = validate_signal(arr)

    if metadata is not None:
        ctx = reference_context_from_metadata(metadata)
        unit = unit or ctx.get("unit")
        observable_kind = observable_kind or ctx.get("observable_kind")
        domain = domain or ctx.get("domain")
        if A_ref is None:
            A_ref = ctx.get("A_ref") or ctx.get("reference_amplitude")

    key = str(method).strip().lower()
    canonical = key in {"canonical", "a_ref", "auto", "default"}

    if canonical:
        if center or clip is not None or replace_nan:
            raise ValueError("canonical normalization cannot center, clip, or replace samples")
        ref = resolve_reference_amplitude(
            unit=unit,
            observable_kind=observable_kind,
            domain=domain,
            A_ref=A_ref,
            default=None,
        )
        ref = validate_positive_scalar(ref, name="A_ref")
        out = arr / ref
    elif key == "zscore":
        mean = np.mean(arr, axis=axis, keepdims=True)
        ref = np.std(arr, axis=axis, keepdims=True)
        if np.any(ref <= 0.0):
            raise ValueError("z-score scale must be strictly positive")
        out = (arr - mean) / ref
    elif key == "minmax":
        minimum = np.min(arr, axis=axis, keepdims=True)
        ref = np.max(arr, axis=axis, keepdims=True) - minimum
        if np.any(ref <= 0.0):
            raise ValueError("min-max scale must be strictly positive")
        out = (arr - minimum) / ref
    elif key == "centered":
        ref = 1.0
        out = center_signal(arr, axis=axis)
    else:
        raise ValueError(f"Unknown normalization method '{method}'")

    if not canonical and clip is not None:
        out = np.clip(out, -abs(float(clip)), abs(float(clip)))

    if not np.all(np.isfinite(out)):
        raise ValueError("normalization produced non-finite values")

    if verbose:
        print(f"[normalization] method={key}, A_ref={np.asarray(ref).reshape(-1)[0]:.6g}")

    return np.asarray(out, dtype=float), ref


def normalize_state(*args, **kwargs):
    """Readability alias for :func:`normalize_signal`."""
    return normalize_signal(*args, **kwargs)
