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

    def attach_result(self, result: AgencityResult) -> None:
        """Attach an analysis result to the experiment."""
        self.result = result

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the experiment to a dictionary."""
        return {
            "dataset": self.dataset.to_dict(),
            "config": self.config.to_dict(),
            "metadata": self.metadata.to_dict(),
            "result": None if self.result is None else self.result.to_dict(),
        }
