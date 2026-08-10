"""
Streaming Agencity API.

This module provides a lightweight real-time processor that accepts chunks
of signal data and produces rolling Agencity results.

It is designed for near-real-time workflows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from .compute import compute_agencity, AgencityResult
from .analyze import analyze_agencity
from .validation import validate_metadata


@dataclass
class AgencityStream:
    """
    Stateful stream processor.

    Parameters
    ----------
    window_size:
        Maximum number of recent samples kept in memory.
        If None, the full history is kept.
    preset:
        Compute preset name.
    config:
        Optional compute configuration overrides.
    metadata:
        Metadata attached to all computed results.
    analyze:
        Whether to run analysis after each update.
    """
    window_size: Optional[int] = None
    preset: str = "default"
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    analyze: bool = True

    _xi_buffer: list = field(default_factory=list, init=False, repr=False)
    _u_buffer: list = field(default_factory=list, init=False, repr=False)
    last_result: Optional[AgencityResult] = None
    last_analysis: Optional[Dict[str, Any]] = None

    def set_preset(self, preset: str):
        self.preset = str(preset)
        return self

    def set_config(self, **kwargs):
        self.config.update(kwargs)
        return self

    def set_backend(self, backend: str = "auto", *, prefer_gpu: bool = False):
        """
        Configure backend for streamed computations.
        """
        self.config["backend"] = backend
        self.config["prefer_gpu"] = bool(prefer_gpu)
        return self

    def set_metadata(self, **kwargs):
        self.metadata.update(kwargs)
        self.metadata = validate_metadata(self.metadata)
        return self

    def clear(self):
        self._xi_buffer.clear()
        self._u_buffer.clear()
        self.last_result = None
        self.last_analysis = None
        return self

    def _append_chunk(self, xi, u):
        xi = np.asarray(xi, dtype=float).ravel()
        u = np.asarray(u, dtype=float).ravel()

        if xi.shape != u.shape:
            raise ValueError("xi and u must have the same shape")

        self._xi_buffer.append(xi)
        self._u_buffer.append(u)

        if self.window_size is not None:
            xi_all = np.concatenate(self._xi_buffer)
            u_all = np.concatenate(self._u_buffer)

            if xi_all.size > self.window_size:
                xi_all = xi_all[-self.window_size :]
                u_all = u_all[-self.window_size :]

            self._xi_buffer = [xi_all]
            self._u_buffer = [u_all]

    def update(
        self,
        u_chunk,
        xi_chunk=None,
        *,
        verbose: bool = False,
        run_analysis: Optional[bool] = None,
        **kwargs,
    ):
        """
        Push a new chunk and compute a rolling result.
        """
        if xi_chunk is None:
            xi_chunk = np.arange(len(np.asarray(u_chunk).ravel()), dtype=float)

        self._append_chunk(xi_chunk, u_chunk)

        xi_all = np.concatenate(self._xi_buffer)
        u_all = np.concatenate(self._u_buffer)

        if verbose:
            print(f"[stream] buffer size = {len(u_all)}")

        self.last_result = compute_agencity(
            data=u_all,
            xi=xi_all,
            preset=self.preset,
            config=dict(self.config),
            metadata=self.metadata,
            verbose=verbose,
            **kwargs,
        )

        should_analyze = self.analyze if run_analysis is None else bool(run_analysis)
        if should_analyze:
            self.last_analysis = analyze_agencity(self.last_result, verbose=verbose)
            self.last_result.attach_analysis(self.last_analysis)
            self.last_result.signature = self.last_analysis.get("signature")
            self.last_result.multiscale = self.last_analysis.get("multiscale")

        return self.last_result

    def push(self, u_chunk, xi_chunk=None, *, verbose: bool = False, **kwargs):
        """
        Alias for update().
        """
        return self.update(u_chunk, xi_chunk=xi_chunk, verbose=verbose, **kwargs)

    def flush(self, *, verbose: bool = False, **kwargs):
        """
        Compute on the current full buffer and return the result.
        """
        if not self._u_buffer:
            raise ValueError("Stream buffer is empty")

        xi_all = np.concatenate(self._xi_buffer)
        u_all = np.concatenate(self._u_buffer)

        self.last_result = compute_agencity(
            data=u_all,
            xi=xi_all,
            preset=self.preset,
            config=dict(self.config),
            metadata=self.metadata,
            verbose=verbose,
            **kwargs,
        )

        if self.analyze:
            self.last_analysis = analyze_agencity(self.last_result, verbose=verbose)
            self.last_result.attach_analysis(self.last_analysis)
            self.last_result.signature = self.last_analysis.get("signature")
            self.last_result.multiscale = self.last_analysis.get("multiscale")

        return self.last_result

    def summary(self) -> Dict[str, Any]:
        if self.last_result is None:
            return {}
        return self.last_result.summary()

    def snapshot(self) -> Dict[str, Any]:
        """
        Return a compact snapshot of stream state.
        """
        return {
            "buffer_length": int(np.concatenate(self._u_buffer).size) if self._u_buffer else 0,
            "window_size": self.window_size,
            "preset": self.preset,
            "has_result": self.last_result is not None,
            "has_analysis": self.last_analysis is not None,
            "config": dict(self.config),
        }


def stream_agencity(
    u_chunk,
    xi_chunk=None,
    *,
    stream: Optional[AgencityStream] = None,
    verbose: bool = False,
    **kwargs,
):
    """
    Convenience function for one-shot streaming updates.
    """
    if stream is None:
        stream = AgencityStream()

    return stream.update(u_chunk, xi_chunk=xi_chunk, verbose=verbose, **kwargs)