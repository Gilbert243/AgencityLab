"""
Pipeline façade for AgencityLab.
"""

from __future__ import annotations

from agencitylab.pipeline.builder import AgencityPipeline


def pipeline() -> AgencityPipeline:
    """Return a new pipeline builder instance."""
    return AgencityPipeline()
