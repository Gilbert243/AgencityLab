"""
Pipeline hook system for AgencityLab.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .steps import PipelineContext, PipelineStep


@dataclass(slots=True)
class PipelineHooks:
    """Optional callback bundle for pipeline execution."""
    before_step: Optional[Callable[[PipelineStep, PipelineContext], None]] = None
    after_step: Optional[Callable[[PipelineStep, PipelineContext], None]] = None
    on_error: Optional[Callable[[PipelineStep, PipelineContext, Exception], None]] = None

    def emit_before(self, step: PipelineStep, context: PipelineContext) -> None:
        """Invoke the before-step hook if present."""
        if self.before_step is not None:
            self.before_step(step, context)

    def emit_after(self, step: PipelineStep, context: PipelineContext) -> None:
        """Invoke the after-step hook if present."""
        if self.after_step is not None:
            self.after_step(step, context)

    def emit_error(self, step: PipelineStep, context: PipelineContext, error: Exception) -> None:
        """Invoke the error hook if present."""
        if self.on_error is not None:
            self.on_error(step, context, error)
