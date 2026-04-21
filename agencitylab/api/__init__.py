"""
High-level public API for AgencityLab.

This package exposes the simplest entry points for users:
- compute_agencity()
- analyze_agencity()
- visualize_agencity()
- pipeline()
"""

from .analyze import analyze_agencity, textual_analysis
from .compute import compute_agencity
from .pipeline_api import pipeline
from .shortcuts import run, summarize
from .visualize import visualize_agencity

__all__ = [
    "analyze_agencity",
    "compute_agencity",
    "pipeline",
    "run",
    "summarize",
    "textual_analysis",
    "visualize_agencity",
]
