"""
Atomic pipeline steps for AgencityLab.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from agencitylab.models.result import AgencityResult


@dataclass(slots=True)
class PipelineContext:
    """Mutable execution context passed between pipeline steps."""
    data: Any = None
    result: Optional[AgencityResult] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PipelineStep:
    """An atomic step in the Agencity pipeline."""
    name: str
    func: Callable[[PipelineContext], PipelineContext]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def run(self, context: PipelineContext) -> PipelineContext:
        """Execute the step against the current context."""
        return self.func(context)
