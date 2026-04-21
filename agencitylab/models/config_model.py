"""
Configuration models for analysis-level utilities.

This module complements agencitylab.config by providing lightweight
analysis-oriented configuration objects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass(slots=True)
class AnalysisConfig:
    """
    Analysis configuration used by the higher-level scientific helpers.
    """
    regime_window: int = 32
    spectrum_nfft: int = 256
    diagnostics_threshold: float = 1e-9
    report_language: str = "en"
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the config to a serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisConfig":
        """Create an AnalysisConfig from a dictionary."""
        return cls(**data)
