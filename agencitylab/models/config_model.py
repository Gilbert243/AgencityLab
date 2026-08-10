"""
Configuration models for analysis-level utilities.

This module complements agencitylab.config by providing lightweight
analysis-oriented configuration objects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict


@dataclass(slots=True)
class AnalysisConfig:
    """
    Analysis configuration used by the higher-level scientific helpers.
    """
    regime_window: int = 32
    spectrum_nfft: int = 256
    diagnostics_threshold: float = 3.0
    report_language: str = "en"
    compute_signature: bool = True
    compute_multiscale: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the config to a serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisConfig":
        """Create an AnalysisConfig from a dictionary."""
        data = dict(data or {})

        known_fields = {
            "regime_window",
            "spectrum_nfft",
            "diagnostics_threshold",
            "report_language",
            "compute_signature",
            "compute_multiscale",
            "extra",
        }

        known = {k: data.pop(k) for k in list(data.keys()) if k in known_fields}
        extra = dict(known.get("extra", {}))
        extra.update(data)

        return cls(
            regime_window=known.get("regime_window", cls.regime_window),
            spectrum_nfft=known.get("spectrum_nfft", cls.spectrum_nfft),
            diagnostics_threshold=known.get("diagnostics_threshold", cls.diagnostics_threshold),
            report_language=known.get("report_language", cls.report_language),
            compute_signature=known.get("compute_signature", cls.compute_signature),
            compute_multiscale=known.get("compute_multiscale", cls.compute_multiscale),
            extra=extra,
        )

    def with_updates(self, **kwargs) -> "AnalysisConfig":
        """Return a new config with updated fields."""
        payload = self.to_dict()
        payload.update(kwargs)
        return AnalysisConfig.from_dict(payload)