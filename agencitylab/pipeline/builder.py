"""
Pipeline builder for AgencityLab.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from agencitylab.api.analyze import analyze_agencity
from agencitylab.api.compute import compute_agencity

from .cache import PipelineCache
from .execution import execute_pipeline
from .hooks import PipelineHooks
from .steps import PipelineContext, PipelineStep


@dataclass
class AgencityPipeline:
    """
    Chainable pipeline object.
    """
    _steps: List[PipelineStep] = field(default_factory=list)
    _hooks: PipelineHooks = field(default_factory=PipelineHooks)
    _cache: Optional[PipelineCache] = None
    _initial_data: Any = None
    _initial_xi: Any = None
    _initial_u: Any = None
    _compute_kwargs: Dict[str, Any] = field(default_factory=dict)
    _analysis_enabled: bool = True

    def from_arrays(self, xi, u) -> "AgencityPipeline":
        """Register a raw signal pair as the initial input."""
        self._initial_xi = xi
        self._initial_u = u
        self._initial_data = None
        return self

    def from_data(self, data) -> "AgencityPipeline":
        """Register a raw data object as the initial input."""
        self._initial_data = data
        self._initial_xi = None
        self._initial_u = None
        return self

    def preprocess(self, **kwargs) -> "AgencityPipeline":
        """Append preprocessing options to the compute stage."""
        self._compute_kwargs.update(kwargs)
        return self

    def compute(self, **kwargs) -> "AgencityPipeline":
        """Append compute options to the compute stage."""
        self._compute_kwargs.update(kwargs)
        return self

    def analyze(self, enabled: bool = True) -> "AgencityPipeline":
        """Enable or disable the analysis stage."""
        self._analysis_enabled = bool(enabled)
        return self

    def with_cache(self, cache: Optional[PipelineCache] = None) -> "AgencityPipeline":
        """Attach a cache to the pipeline."""
        self._cache = cache or PipelineCache()
        return self

    def with_hooks(self, hooks: PipelineHooks) -> "AgencityPipeline":
        """Attach pipeline hooks."""
        self._hooks = hooks
        return self

    def step(self, name: str, func: Callable[[PipelineContext], PipelineContext], **metadata) -> "AgencityPipeline":
        """Append a custom pipeline step."""
        self._steps.append(PipelineStep(name=name, func=func, metadata=metadata))
        return self

    def build(self) -> List[PipelineStep]:
        """Build a runnable list of pipeline steps."""
        steps = list(self._steps)

        if self._initial_data is not None or (self._initial_xi is not None and self._initial_u is not None):
            def _load(context: PipelineContext) -> PipelineContext:
                context.data = self._initial_data if self._initial_data is not None else {
                    "xi": self._initial_xi,
                    "u": self._initial_u,
                }
                return context
            steps.insert(0, PipelineStep("load", _load))

        def _compute(context: PipelineContext) -> PipelineContext:
            if context.data is not None:
                if isinstance(context.data, dict) and "xi" in context.data and "u" in context.data:
                    context.result = compute_agencity(data=context.data, **self._compute_kwargs)
                else:
                    context.result = compute_agencity(data=context.data, **self._compute_kwargs)
            elif self._initial_xi is not None and self._initial_u is not None:
                context.result = compute_agencity(self._initial_xi, self._initial_u, **self._compute_kwargs)
            else:
                raise RuntimeError("No input data available for compute stage.")
            context.artifacts["result"] = context.result
            return context

        steps.append(PipelineStep("compute", _compute))

        if self._analysis_enabled:
            def _analyze(context: PipelineContext) -> PipelineContext:
                if context.result is None:
                    raise RuntimeError("Cannot analyze without a computed result.")
                context.artifacts["analysis"] = analyze_agencity(context.result)
                return context
            steps.append(PipelineStep("analyze", _analyze))

        return steps

    def run(self) -> PipelineContext:
        """Execute the pipeline and return the final context."""
        context = PipelineContext()
        return execute_pipeline(self.build(), context=context, hooks=self._hooks, cache=self._cache)


PipelineBuilder = AgencityPipeline
