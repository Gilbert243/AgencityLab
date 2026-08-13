"""Software/runtime configuration for AgencityLab.

This module contains software behaviour only. Physical quantities such as
``A_ref``, ``tau``, ``w`` and ``P_c`` are explicit scientific inputs and are not
accepted as global configuration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any

from .modes import AgencityMode


@dataclass(slots=True)
class AgencityConfig:
    """Runtime software options that never redefine scientific equations."""

    mode: AgencityMode = AgencityMode.CANONICAL
    backend: str = "numpy"
    prefer_gpu: bool = False
    compute_signature: bool = True
    compute_multiscale: bool = True
    report_language: str = "en"
    streaming_enabled: bool = False
    batch_parallel: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.mode = AgencityMode.from_value(self.mode)
        self.backend = str(self.backend).strip().lower()
        self.prefer_gpu = bool(self.prefer_gpu)
        self.compute_signature = bool(self.compute_signature)
        self.compute_multiscale = bool(self.compute_multiscale)
        self.report_language = str(self.report_language).strip().lower()
        self.streaming_enabled = bool(self.streaming_enabled)
        self.batch_parallel = bool(self.batch_parallel)
        self.metadata = dict(self.metadata or {})

    def to_dict(self) -> dict[str, Any]:
        """Return a detached serializable representation."""
        payload = asdict(self)
        payload["mode"] = self.mode.value
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "AgencityConfig":
        """Construct a config and reject unknown options instead of hiding typos."""
        payload = dict(data or {})
        allowed = {item.name for item in fields(cls)}
        unexpected = sorted(set(payload) - allowed)
        if unexpected:
            names = ", ".join(unexpected)
            raise TypeError(f"unexpected AgencityConfig option(s): {names}")
        return cls(**payload)

    def with_updates(self, **kwargs: Any) -> "AgencityConfig":
        """Return a validated copy with selected software options updated."""
        payload = self.to_dict()
        payload.update(kwargs)
        return AgencityConfig.from_dict(payload)


DEFAULT_CONFIG = AgencityConfig()
