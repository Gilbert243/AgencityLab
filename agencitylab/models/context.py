"""
Execution context for AgencityLab.

This container groups a signal, a result, analyses and artifacts together.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass(slots=True)
class Context:
    """
    Generic execution context.

    Useful for pipelines, notebooks and batch processing.
    """
    name: str = "default"
    metadata: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)

    signal: Optional[Any] = None
    result: Optional[Any] = None
    analysis: Dict[str, Any] = field(default_factory=dict)
    report: Optional[str] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)

    def attach_signal(self, signal: Any) -> None:
        self.signal = signal

    def attach_result(self, result: Any) -> None:
        self.result = result

    def attach_analysis(self, analysis: Dict[str, Any]) -> None:
        self.analysis = dict(analysis or {})

    def attach_report(self, report: str) -> None:
        self.report = report

    def add_artifact(self, name: str, value: Any) -> None:
        self.artifacts[str(name)] = value

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the context to a dictionary."""
        return {
            "name": self.name,
            "metadata": dict(self.metadata),
            "config": dict(self.config),
            "signal": self.signal.to_dict() if hasattr(self.signal, "to_dict") else self.signal,
            "result": self.result.to_dict() if hasattr(self.result, "to_dict") else self.result,
            "analysis": dict(self.analysis),
            "report": self.report,
            "artifacts": dict(self.artifacts),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Context":
        """Rebuild a context from a dictionary."""
        data = dict(data or {})

        signal = data.get("signal")
        result = data.get("result")

        if isinstance(signal, dict):
            from .signal import AgencitySignal
            signal = AgencitySignal.from_dict(signal)

        if isinstance(result, dict):
            from .result import AgencityResult
            result = AgencityResult.from_dict(result)

        return cls(
            name=data.get("name", "default"),
            metadata=dict(data.get("metadata", {})),
            config=dict(data.get("config", {})),
            signal=signal,
            result=result,
            analysis=dict(data.get("analysis", {})),
            report=data.get("report"),
            artifacts=dict(data.get("artifacts", {})),
        )