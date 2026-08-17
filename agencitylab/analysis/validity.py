"""Numerical validity intervals for analysis of canonical results.

This module centralises finite-record exclusions used by diagnostics.  The CRM
warm-up is a numerical/finite-record concern and never changes canonical
quantities.  The default rule excludes two complete CRM windows and optional
finite-difference edge samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True, slots=True)
class AnalysisInterval:
    """Resolved finite-record interval used by analysis diagnostics."""

    mask: np.ndarray
    start_index: int
    stop_index: int
    start_time: float | None
    stop_time: float | None
    memory_window: float
    memory_window_source: str
    warmup_windows: float
    edge_samples: int
    valid_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_index": self.start_index,
            "stop_index": self.stop_index,
            "start_time": self.start_time,
            "stop_time": self.stop_time,
            "memory_window": self.memory_window,
            "memory_window_source": self.memory_window_source,
            "warmup_windows": self.warmup_windows,
            "edge_samples": self.edge_samples,
            "valid_fraction": self.valid_fraction,
            "rule": (
                f"t  + {self.warmup_windows:g}*w, "
                "excluding numerical edge samples"
            ),
        }


def _lookup(value, name: str):
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _finite_positive_scalar(value, *, name: str) -> float:
    try:
        result = float(value)
    except Exception as exc:
        raise ValueError(f"{name} must be a finite positive scalar") from exc
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return result


def _resolve_memory_window(result, explicit_w) -> tuple[float, str]:
    if explicit_w is not None:
        return _finite_positive_scalar(explicit_w, name="w"), "explicit"

    for name in ("w", "memory_window"):
        value = _lookup(result, name)
        if value is not None:
            return _finite_positive_scalar(value, name=name), name

    metadata = _lookup(result, "metadata")
    if metadata is not None:
        value = _lookup(metadata, "memory_window")
        if value is not None:
            return _finite_positive_scalar(value, name="metadata.memory_window"), (
                "metadata.memory_window"
            )

    tau = _lookup(result, "tau")
    if tau is None:
        raise ValueError("w is required when result does not expose a memory window or tau")
    return _finite_positive_scalar(tau, name="tau"), "tau_fallback"


def resolve_analysis_interval(
    result,
    *,
    w=None,
    warmup_windows: float = 2.0,
    edge_samples: int = 0,
    extra_mask=None,
) -> AnalysisInterval:
    """Return the shared numerical-validity interval for analysis diagnostics."""

    xi = np.asarray(_lookup(result, "xi"), dtype=float)
    if xi.ndim != 1 or xi.size == 0:
        raise ValueError("result.xi must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(xi)) or np.any(np.diff(xi) <= 0.0):
        raise ValueError("result.xi must be finite and strictly increasing")

    try:
        windows = float(warmup_windows)
    except Exception as exc:
        raise ValueError("warmup_windows must be finite and non-negative") from exc
    if not np.isfinite(windows) or windows < 0.0:
        raise ValueError("warmup_windows must be finite and non-negative")

    if isinstance(edge_samples, (bool, np.bool_)) or not isinstance(
        edge_samples,
        (int, np.integer),
    ):
        raise ValueError("edge_samples must be a non-negative integer")
    edges = int(edge_samples)
    if edges < 0:
        raise ValueError("edge_samples must be a non-negative integer")

    window, source = _resolve_memory_window(result, w)
    mask = xi >= xi[0] + windows * window
    if edges:
        mask[: min(edges, mask.size)] = False
        mask[max(0, mask.size - edges) :] = False

    if extra_mask is not None:
        supplied = np.asarray(extra_mask, dtype=bool)
        if supplied.ndim != 1 or supplied.size != xi.size:
            raise ValueError("extra_mask must be one-dimensional and match xi")
        mask &= supplied

    valid_indices = np.flatnonzero(mask)
    if valid_indices.size:
        start = int(valid_indices[0])
        stop = int(valid_indices[-1] + 1)
        start_time = float(xi[start])
        stop_time = float(xi[stop - 1])
    else:
        start = int(xi.size)
        stop = int(xi.size)
        start_time = None
        stop_time = None

    return AnalysisInterval(
        mask=mask,
        start_index=start,
        stop_index=stop,
        start_time=start_time,
        stop_time=stop_time,
        memory_window=window,
        memory_window_source=source,
        warmup_windows=windows,
        edge_samples=edges,
        valid_fraction=float(np.mean(mask)),
    )


def analysis_valid_mask(result, **kwargs) -> np.ndarray:
    """Return only the Boolean mask from :func:`resolve_analysis_interval`."""

    return resolve_analysis_interval(result, **kwargs).mask
