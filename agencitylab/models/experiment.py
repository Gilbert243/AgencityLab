"""
Experiment model for AgencityLab.

The experiment object connects a dataset, configuration, results and
reproducibility information.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .config_model import AnalysisConfig
from .dataset import AgencityDataset
from .metadata import ExperimentMetadata
from .result import AgencityResult


@dataclass(slots=True)
class AgencityExperiment:
    """
    High-level scientific experiment container.
    """
    dataset: AgencityDataset
    config: AnalysisConfig = field(default_factory=AnalysisConfig)
    metadata: ExperimentMetadata = field(default_factory=ExperimentMetadata)
    result: Optional[AgencityResult] = None
    analysis: Dict[str, Any] = field(default_factory=dict)
    report_text: Optional[str] = None

    def attach_result(self, result: AgencityResult) -> None:
        """Attach an analysis result to the experiment."""
        self.result = result

    def attach_analysis(self, analysis: Dict[str, Any]) -> None:
        """Attach a structured analysis dictionary."""
        self.analysis = dict(analysis or {})

    def attach_report(self, text: str) -> None:
        """Attach a human-readable report."""
        self.report_text = text

    def summary(self) -> Dict[str, Any]:
        """Compact experiment summary."""
        out = {
            "dataset": self.dataset.summary(),
            "config": self.config.to_dict(),
            "metadata": self.metadata.to_dict(),
            "has_result": self.result is not None,
        }
        if self.result is not None:
            out["result_summary"] = self.result.summary()
        return out

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the experiment to a dictionary."""
        return {
            "dataset": self.dataset.to_dict(),
            "config": self.config.to_dict(),
            "metadata": self.metadata.to_dict(),
            "result": None if self.result is None else self.result.to_dict(),
            "analysis": dict(self.analysis),
            "report_text": self.report_text,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgencityExperiment":
        """Rebuild an experiment from a dictionary."""
        dataset = data.get("dataset", {})
        config = data.get("config", {})
        metadata = data.get("metadata", {})
        result = data.get("result", None)

        if not isinstance(dataset, AgencityDataset):
            dataset = AgencityDataset.from_dict(dataset)

        if not isinstance(config, AnalysisConfig):
            config = AnalysisConfig.from_dict(config)

        if not isinstance(metadata, ExperimentMetadata):
            metadata = ExperimentMetadata.from_dict(metadata)

        if isinstance(result, dict):
            result = AgencityResult.from_dict(result)

        return cls(
            dataset=dataset,
            config=config,
            metadata=metadata,
            result=result,
            analysis=dict(data.get("analysis", {})),
            report_text=data.get("report_text"),
        )