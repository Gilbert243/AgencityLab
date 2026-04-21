"""
Dataset model for AgencityLab.

A dataset is a collection of canonical signals with shared metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from .metadata import ExperimentMetadata
from .signal import AgencitySignal


@dataclass(slots=True)
class AgencityDataset:
    """
    Group of signals analyzed together.
    """
    items: list[AgencitySignal] = field(default_factory=list)
    metadata: ExperimentMetadata = field(default_factory=ExperimentMetadata)

    def add(self, signal: AgencitySignal) -> None:
        """Append a signal to the dataset."""
        if not isinstance(signal, AgencitySignal):
            raise TypeError("signal must be an AgencitySignal instance.")
        self.items.append(signal)

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the dataset to a dictionary."""
        return {
            "items": [item.to_dict() for item in self.items],
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgencityDataset":
        """Create a dataset from a dictionary representation."""
        metadata = data.get("metadata", {})
        if not isinstance(metadata, ExperimentMetadata):
            metadata = ExperimentMetadata.from_dict(metadata)

        items = [AgencitySignal.from_dict(item) for item in data.get("items", [])]
        return cls(items=items, metadata=metadata)
