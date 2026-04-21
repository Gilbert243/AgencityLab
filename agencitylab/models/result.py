"""
Result model for AgencityLab.

AgencityResult is the central object returned by the computation and analysis
pipelines. It stores the canonical quantities of the theory in a reproducible
and serializable form.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .metadata import ExperimentMetadata


@dataclass(slots=True)
class AgencityResult:
    """
    Complete analysis result.

    Attributes
    ----------
    xi:
        Canonical coordinate array.
    u:
        Input signal values.
    u_star:
        Normalized signal.
    X_star:
        Reduced activation.
    A_star:
        Reduced activity.
    tau:
        Characteristic scale.
    t_star:
        Reduced coordinate.
    M:
        Memory variable.
    O:
        Organization variable.
    beta:
        Agencement variable.
    b_reduced:
        Reduced Agencity observable.
    b:
        Full Agencity observable.
    P_c:
        Characteristic power.
    metadata:
        Experiment metadata.
    """
    xi: np.ndarray
    u: np.ndarray
    u_star: np.ndarray
    X_star: np.ndarray
    A_star: np.ndarray
    tau: float
    t_star: np.ndarray
    M: np.ndarray
    O: np.ndarray
    beta: np.ndarray
    b_reduced: np.ndarray
    b: np.ndarray
    P_c: np.ndarray
    metadata: ExperimentMetadata = field(default_factory=ExperimentMetadata)
    config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.xi = np.asarray(self.xi, dtype=float)
        self.u = np.asarray(self.u, dtype=float)
        self.u_star = np.asarray(self.u_star, dtype=float)
        self.X_star = np.asarray(self.X_star, dtype=float)
        self.A_star = np.asarray(self.A_star, dtype=float)
        self.t_star = np.asarray(self.t_star, dtype=float)
        self.M = np.asarray(self.M, dtype=float)
        self.O = np.asarray(self.O, dtype=float)
        self.beta = np.asarray(self.beta, dtype=float)
        self.b_reduced = np.asarray(self.b_reduced, dtype=float)
        self.b = np.asarray(self.b, dtype=float)
        self.P_c = np.asarray(self.P_c, dtype=float)

        n = self.xi.shape[0]
        for name in ("u", "u_star", "X_star", "A_star", "t_star", "M", "O", "beta", "b_reduced", "b", "P_c"):
            arr = getattr(self, name)
            if arr.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional.")
            if arr.shape[0] != n:
                raise ValueError(f"{name} must have the same length as xi.")

        if float(self.tau) <= 0.0:
            raise ValueError("tau must be strictly positive.")

    @property
    def eta(self) -> np.ndarray:
        """Return the reduced efficiency eta = b / P_c."""
        eps = 1e-12
        return self.b / np.maximum(np.abs(self.P_c), eps)

    def summary(self) -> Dict[str, Any]:
        """Return a compact numeric summary."""
        return {
            "n_samples": int(self.xi.size),
            "tau": float(self.tau),
            "b_mean": float(np.mean(self.b)),
            "b_std": float(np.std(self.b)),
            "beta_min": float(np.min(self.beta)) if self.beta.size else 0.0,
            "beta_max": float(np.max(self.beta)) if self.beta.size else 0.0,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the result to a dictionary."""
        return {
            "xi": self.xi.tolist(),
            "u": self.u.tolist(),
            "u_star": self.u_star.tolist(),
            "X_star": self.X_star.tolist(),
            "A_star": self.A_star.tolist(),
            "tau": float(self.tau),
            "t_star": self.t_star.tolist(),
            "M": self.M.tolist(),
            "O": self.O.tolist(),
            "beta": self.beta.tolist(),
            "b_reduced": self.b_reduced.tolist(),
            "b": self.b.tolist(),
            "P_c": self.P_c.tolist(),
            "metadata": self.metadata.to_dict(),
            "config": dict(self.config),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgencityResult":
        """Rebuild a result object from a dictionary."""
        metadata = data.get("metadata", {})
        if not isinstance(metadata, ExperimentMetadata):
            metadata = ExperimentMetadata.from_dict(metadata)
        return cls(
            xi=np.asarray(data["xi"], dtype=float),
            u=np.asarray(data["u"], dtype=float),
            u_star=np.asarray(data["u_star"], dtype=float),
            X_star=np.asarray(data["X_star"], dtype=float),
            A_star=np.asarray(data["A_star"], dtype=float),
            tau=float(data["tau"]),
            t_star=np.asarray(data["t_star"], dtype=float),
            M=np.asarray(data["M"], dtype=float),
            O=np.asarray(data["O"], dtype=float),
            beta=np.asarray(data["beta"], dtype=float),
            b_reduced=np.asarray(data["b_reduced"], dtype=float),
            b=np.asarray(data["b"], dtype=float),
            P_c=np.asarray(data["P_c"], dtype=float),
            metadata=metadata,
            config=dict(data.get("config", {})),
        )

    def save_json(self, path: str | Path) -> Path:
        """Save the result using the project IO layer."""
        from agencitylab.io.save import save
        return save(self.to_dict(), path)

    @classmethod
    def load_json(cls, path: str | Path) -> "AgencityResult":
        """Load a result using the project IO layer."""
        from agencitylab.io.load import load
        payload = load(path)
        return cls.from_dict(payload)
