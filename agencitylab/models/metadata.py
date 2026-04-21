"""
Metadata models for AgencityLab.

Metadata is kept separate from the numerical payload to preserve clarity
and to support reproducible scientific workflows.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


@dataclass(slots=True)
class ExperimentMetadata:
    """
    Metadata attached to a signal or an analysis result.
    """
    title: str = ""
    description: str = ""
    author: str = ""
    domain: str = ""
    source: str = ""
    coordinate_name: str = "xi"
    signal_name: str = "u"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    tags: list[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to a serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentMetadata":
        """Create metadata from a dictionary."""
        return cls(**data)
