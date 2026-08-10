"""Stateful streaming interface for the stable AgencityLab API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from agencitylab.exceptions import AgencityValidationError, StreamNotReadyError, StreamStateError
from agencitylab.models import AgencityResult

from .analyze import analyze_agencity
from .compute import compute_agencity
from .validation import validate_metadata


@dataclass
class AgencityStream:
    """Rolling scalar Agencity processor.

    The first five fields preserve the pre-v0.3 constructor order. Canonical
    physical parameters may now be supplied once on the stream object instead of
    being repeated on every update.
    """

    window_size: Optional[int] = None
    preset: str = "default"
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    analyze: bool = True

    A_ref: float | str | None = None
    tau: float | str | None = "auto"
    w: float | None = None
    P_c: float | str | None = "auto"
    unit: Optional[str] = None
    coordinate_unit: Optional[str] = None
    power_unit: Optional[str] = None

    _xi_buffer: list[np.ndarray] = field(default_factory=list, init=False, repr=False)
    _u_buffer: list[np.ndarray] = field(default_factory=list, init=False, repr=False)
    last_result: Optional[AgencityResult] = None
    last_analysis: Optional[Dict[str, Any]] = None
    last_error: Optional[str] = None

    def __post_init__(self) -> None:
        if self.window_size is not None:
            if not isinstance(self.window_size, int) or self.window_size < 3:
                raise StreamStateError("window_size must be an integer >= 3 or None")
        if not isinstance(self.config, dict):
            raise StreamStateError("config must be a dictionary")
        self.config = dict(self.config)
        self.metadata = validate_metadata(self.metadata)

    def set_preset(self, preset: str):
        self.preset = str(preset)
        return self

    def set_config(self, **kwargs):
        self.config.update(kwargs)
        return self

    def set_backend(self, backend: str = "auto", *, prefer_gpu: bool = False):
        self.config["backend"] = backend
        self.config["prefer_gpu"] = bool(prefer_gpu)
        return self

    def set_metadata(self, **kwargs):
        merged = dict(self.metadata)
        merged.update(kwargs)
        self.metadata = validate_metadata(merged)
        return self

    def set_physical_context(
        self,
        *,
        A_ref=None,
        tau=None,
        w=None,
        P_c=None,
        unit=None,
        coordinate_unit=None,
        power_unit=None,
    ):
        """Update persistent canonical physical/contextual parameters."""
        if A_ref is not None:
            self.A_ref = A_ref
        if tau is not None:
            self.tau = tau
        if w is not None:
            self.w = w
        if P_c is not None:
            self.P_c = P_c
        if unit is not None:
            self.unit = unit
        if coordinate_unit is not None:
            self.coordinate_unit = coordinate_unit
        if power_unit is not None:
            self.power_unit = power_unit
        return self

    def clear(self):
        self._xi_buffer.clear()
        self._u_buffer.clear()
        self.last_result = None
        self.last_analysis = None
        self.last_error = None
        return self

    def _all_xi(self) -> np.ndarray:
        return np.concatenate(self._xi_buffer) if self._xi_buffer else np.array([], dtype=float)

    def _all_u(self) -> np.ndarray:
        return np.concatenate(self._u_buffer) if self._u_buffer else np.array([], dtype=float)

    def _implicit_axis(self, size: int) -> np.ndarray:
        existing = self._all_xi()
        if existing.size == 0:
            start = 0.0
            step = 1.0
        elif existing.size == 1:
            start = float(existing[-1] + 1.0)
            step = 1.0
        else:
            step = float(existing[-1] - existing[-2])
            start = float(existing[-1] + step)
        return start + step * np.arange(size, dtype=float)

    def _append_chunk(self, xi, u) -> None:
        try:
            values = np.asarray(u, dtype=float)
        except Exception as exc:
            raise StreamStateError("u_chunk must be numeric") from exc
        if values.ndim != 1:
            raise StreamStateError("u_chunk must be one-dimensional")
        if values.size == 0:
            raise StreamStateError("u_chunk cannot be empty")
        if not np.all(np.isfinite(values)):
            raise StreamStateError("u_chunk must contain only finite values")

        if xi is None:
            axis = self._implicit_axis(values.size)
            if self.coordinate_unit is None and not self.metadata.get("coordinate_unit"):
                self.coordinate_unit = "sample"
        else:
            try:
                axis = np.asarray(xi, dtype=float)
            except Exception as exc:
                raise StreamStateError("xi_chunk must be numeric") from exc
            if axis.ndim != 1 or axis.shape != values.shape:
                raise StreamStateError("xi_chunk and u_chunk must be one-dimensional and equal length")
            if not np.all(np.isfinite(axis)) or np.any(np.diff(axis) <= 0.0):
                raise StreamStateError("xi_chunk must be finite and strictly increasing")
            existing = self._all_xi()
            if existing.size and axis[0] <= existing[-1]:
                raise StreamStateError(
                    "xi_chunk must start strictly after the previous stream coordinate"
                )

        self._xi_buffer.append(axis)
        self._u_buffer.append(values)

        if self.window_size is not None:
            axis_all = self._all_xi()
            values_all = self._all_u()
            if values_all.size > self.window_size:
                axis_all = axis_all[-self.window_size :]
                values_all = values_all[-self.window_size :]
            self._xi_buffer = [axis_all]
            self._u_buffer = [values_all]

    def _compute_kwargs(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        kwargs = {
            "A_ref": self.A_ref,
            "tau": self.tau,
            "w": self.w,
            "P_c": self.P_c,
            "unit": self.unit,
            "coordinate_unit": self.coordinate_unit,
            "power_unit": self.power_unit,
        }
        kwargs.update(overrides)
        return kwargs

    def _compute_current(self, *, verbose: bool, overrides: Dict[str, Any]):
        values = self._all_u()
        axis = self._all_xi()
        if values.size < 3:
            raise StreamNotReadyError(
                f"stream contains {values.size} samples; at least 3 are required"
            )

        try:
            return compute_agencity(
                u=values,
                xi=axis,
                preset=self.preset,
                config=dict(self.config),
                metadata=self.metadata,
                verbose=verbose,
                **self._compute_kwargs(overrides),
            )
        except AgencityValidationError as exc:
            message = str(exc)
            if "signal too short for two CRM windows" in message:
                raise StreamNotReadyError(
                    "stream buffer does not yet contain two complete CRM windows"
                ) from exc
            raise

    def update(
        self,
        u_chunk,
        xi_chunk=None,
        *,
        verbose: bool = False,
        run_analysis: Optional[bool] = None,
        **kwargs,
    ):
        """Append one chunk and return the rolling result when the buffer is ready."""
        self._append_chunk(xi_chunk, u_chunk)
        if verbose:
            print(f"[stream] buffer size = {self._all_u().size}")

        try:
            result = self._compute_current(verbose=verbose, overrides=kwargs)
        except Exception as exc:
            self.last_error = str(exc)
            raise

        self.last_error = None
        self.last_result = result
        should_analyze = self.analyze if run_analysis is None else bool(run_analysis)
        if should_analyze:
            self.last_analysis = analyze_agencity(result, verbose=verbose)
            result.attach_analysis(self.last_analysis)
            result.signature = self.last_analysis.get("signature")
            result.multiscale = self.last_analysis.get("multiscale")
        else:
            self.last_analysis = None
        return result

    def push(self, u_chunk, xi_chunk=None, *, verbose: bool = False, **kwargs):
        """Compatibility alias for :meth:`update`."""
        return self.update(u_chunk, xi_chunk=xi_chunk, verbose=verbose, **kwargs)

    def flush(self, *, verbose: bool = False, **kwargs):
        """Compute on the current buffer without appending new samples."""
        if not self._u_buffer:
            raise StreamNotReadyError("stream buffer is empty")
        result = self._compute_current(verbose=verbose, overrides=kwargs)
        self.last_result = result
        self.last_error = None
        if self.analyze:
            self.last_analysis = analyze_agencity(result, verbose=verbose)
            result.attach_analysis(self.last_analysis)
            result.signature = self.last_analysis.get("signature")
            result.multiscale = self.last_analysis.get("multiscale")
        return result

    def summary(self) -> Dict[str, Any]:
        return {} if self.last_result is None else self.last_result.summary()

    def snapshot(self) -> Dict[str, Any]:
        """Return a compact, serialization-safe stream state snapshot."""
        return {
            "buffer_length": int(self._all_u().size),
            "window_size": self.window_size,
            "preset": self.preset,
            "has_result": self.last_result is not None,
            "has_analysis": self.last_analysis is not None,
            "last_error": self.last_error,
            "config": dict(self.config),
            "metadata": dict(self.metadata),
        }


def stream_agencity(
    u_chunk,
    xi_chunk=None,
    *,
    stream: Optional[AgencityStream] = None,
    verbose: bool = False,
    **kwargs,
):
    """Convenience function for one streaming update."""
    if stream is None:
        stream = AgencityStream()
    return stream.update(u_chunk, xi_chunk=xi_chunk, verbose=verbose, **kwargs)
