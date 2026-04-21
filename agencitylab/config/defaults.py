"""
Default configuration values for AgencityLab.

The defaults are intentionally conservative and reproducible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

from .modes import AgencityMode


@dataclass(slots=True)
class AgencityConfig:
    """
    Main configuration object for the framework.

    Parameters
    ----------
    mode:
        Operational mode of the framework.
    normalization_method:
        Strategy used to normalize the input signal u(ξ).
    tau_threshold:
        Threshold used to estimate the characteristic scale τ from autocorrelation.
    activity_window:
        Half-window used in the discrete activity computation.
    crm_window:
        Window used for causal moving correlation.
    epsilon:
        Numerical safeguard used to avoid divisions by zero.
    use_numba:
        Optional performance accelerator.
    use_jax:
        Optional research accelerator.
    """

    mode: AgencityMode = AgencityMode.CANONICAL
    normalization_method: str = "zscore"
    tau_threshold: float = 0.5
    activity_window: int = 1
    crm_window: int = 1
    epsilon: float = 1e-12
    use_numba: bool = False
    use_jax: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the config to a plain serializable dictionary."""
        payload = asdict(self)
        payload["mode"] = self.mode.value
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgencityConfig":
        """Create a configuration object from a dictionary."""
        payload = dict(data)
        if "mode" in payload:
            payload["mode"] = AgencityMode.from_value(payload["mode"])
        return cls(**payload)


DEFAULT_CONFIG = AgencityConfig()
