"""
High-level preprocessing pipeline for AgencityLab.

The pipeline turns raw inputs into a canonical signal representation:
u(ξ) -> preprocessing -> validated SignalData
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

from .loaders.numpy_loader import load_numpy_signal
from .validation import SignalData, validate_input, validate_signal_input
from .transforms.normalize import normalize_signal
from .transforms.interpolate import interpolate_signal
from .transforms.resample import resample_signal
from .transforms.smoothing import smooth_signal
from .transforms.detrend import detrend_signal
from .transforms.windowing import apply_window


@dataclass(slots=True)
class DataPipeline:
    """
    Chainable preprocessing pipeline.

    The pipeline remains intentionally simple and explicit to preserve
    scientific traceability.
    """
    xi: Optional[np.ndarray] = None
    u: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    steps: List[str] = field(default_factory=list)

    def from_arrays(self, xi: Any, u: Any) -> "DataPipeline":
        """Load a signal from explicit arrays."""
        self.xi, self.u = validate_signal_input(xi, u)
        self.steps.append("from_arrays")
        return self

    def from_numpy(self, data: Any) -> "DataPipeline":
        """
        Load a signal from a NumPy-like container.

        Accepted shapes:
        - (n, 2): first column xi, second column u
        - {"xi": ..., "u": ...}: dict-like object
        """
        xi, u = load_numpy_signal(data)
        return self.from_arrays(xi, u)

    def normalize(self, method: str = "zscore") -> "DataPipeline":
        """Normalize the signal values."""
        self._require_signal()
        self.u = normalize_signal(self.u, method=method)
        self.steps.append(f"normalize:{method}")
        return self

    def detrend(self, method: str = "linear") -> "DataPipeline":
        """Remove slow trends from the signal values."""
        self._require_signal()
        self.u = detrend_signal(self.xi, self.u, method=method)
        self.steps.append(f"detrend:{method}")
        return self

    def smooth(self, method: str = "moving_average", window_size: int = 5) -> "DataPipeline":
        """Smooth the signal values."""
        self._require_signal()
        self.u = smooth_signal(self.u, method=method, window_size=window_size)
        self.steps.append(f"smooth:{method}:{window_size}")
        return self

    def resample(self, num_points: int) -> "DataPipeline":
        """Resample the signal on a new uniform grid."""
        self._require_signal()
        self.xi, self.u = resample_signal(self.xi, self.u, num_points=num_points)
        self.steps.append(f"resample:{num_points}")
        return self

    def interpolate(self, new_xi: Any, method: str = "linear") -> "DataPipeline":
        """Interpolate the signal onto a new coordinate grid."""
        self._require_signal()
        self.xi, self.u = interpolate_signal(self.xi, self.u, new_xi, method=method)
        self.steps.append(f"interpolate:{method}")
        return self

    def window(self, kind: str = "hann") -> "DataPipeline":
        """Apply a tapering window to the signal values."""
        self._require_signal()
        self.u = apply_window(self.u, kind=kind)
        self.steps.append(f"window:{kind}")
        return self

    def build(self) -> SignalData:
        """Return a canonical SignalData object."""
        self._require_signal()
        self.steps.append("build")
        return SignalData(xi=self.xi.copy(), u=self.u.copy(), metadata=dict(self.metadata))

    def _require_signal(self) -> None:
        """Ensure that a signal has already been loaded."""
        if self.xi is None or self.u is None:
            raise RuntimeError("No signal loaded. Use from_arrays() or from_numpy() first.")


def prepare_signal(
    xi: Any,
    u: Any,
    *,
    normalize: bool = True,
    detrend: bool = False,
    smooth: bool = False,
    resample_points: Optional[int] = None,
    window: Optional[str] = None,
    normalization_method: str = "zscore",
    detrend_method: str = "linear",
    smoothing_method: str = "moving_average",
    smoothing_window_size: int = 5,
) -> SignalData:
    """
    Convenience function to build a processed SignalData object in one call.
    """
    pipeline = DataPipeline().from_arrays(xi, u)

    if resample_points is not None:
        pipeline.resample(int(resample_points))

    if detrend:
        pipeline.detrend(method=detrend_method)

    if smooth:
        pipeline.smooth(method=smoothing_method, window_size=smoothing_window_size)

    if window is not None:
        pipeline.window(kind=window)

    if normalize:
        pipeline.normalize(method=normalization_method)

    return pipeline.build()
