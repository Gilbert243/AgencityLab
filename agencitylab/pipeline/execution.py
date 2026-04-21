"""
Pipeline execution engine for AgencityLab.
"""

from __future__ import annotations

from typing import Iterable, Optional

from .cache import PipelineCache
from .hooks import PipelineHooks
from .steps import PipelineContext, PipelineStep


def execute_pipeline(
    steps: Iterable[PipelineStep],
    context: Optional[PipelineContext] = None,
    hooks: Optional[PipelineHooks] = None,
    cache: Optional[PipelineCache] = None,
) -> PipelineContext:
    """
    Execute a sequence of pipeline steps.
    """
    context = context or PipelineContext()
    hooks = hooks or PipelineHooks()

    for step in steps:
        hooks.emit_before(step, context)

        try:
            if cache is not None:
                cache_key = cache.make_key(step.name, context.artifacts, context.metadata)
                cached = cache.get(cache_key, default=None)
                if cached is not None:
                    context = cached
                    hooks.emit_after(step, context)
                    continue

            context = step.run(context)

            if cache is not None:
                cache_key = cache.make_key(step.name, context.artifacts, context.metadata)
                cache.set(cache_key, context)

            hooks.emit_after(step, context)

        except Exception as error:
            hooks.emit_error(step, context, error)
            raise

    return context
