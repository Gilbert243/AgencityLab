"""
Signal model for AgencityLab.

This object represents the canonical input signal u(ξ) after preprocessing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from .metadata import ExperimentMetadata


@dataclass(slots=True)
class AgencitySignal:
    """
    Canonical signal container.

    Parameters
    ----------
    xi:
        Coordinate array.
    u:
        Signal values.
    metadata:
        Optional metadata attached to the signal.
    """
    xi: np.ndarray
    u: np.ndarray
    metadata: ExperimentMetadata = field(default_factory=ExperimentMetadata)

    def __post_init__(self) -> None:
        self.xi = np.asarray(self.xi, dtype=float)
        self.u = np.asarray(self.u, dtype=float)

        if self.xi.ndim != 1 or self.u.ndim != 1:
            raise ValueError("xi and u must be one-dimensional.")
        if self.xi.shape[0] != self.u.shape[0]:
            raise ValueError("xi and u must have the same length.")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the signal into a dictionary."""
        return {
            "xi": self.xi.tolist(),
            "u": self.u.tolist(),
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgencitySignal":
        """Create a signal from a dictionary representation."""
        metadata = data.get("metadata", {})
        if not isinstance(metadata, ExperimentMetadata):
            metadata = ExperimentMetadata.from_dict(metadata)
        return cls(
            xi=np.asarray(data["xi"], dtype=float),
            u=np.asarray(data["u"], dtype=float),
            metadata=metadata,
        )
