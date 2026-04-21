"""
User-friendly shortcuts for AgencityLab.
"""

from __future__ import annotations

from .analyze import textual_analysis
from .compute import compute_agencity
from .visualize import visualize_agencity


def run(data=None, u=None, **kwargs):
    """Compute Agencity and return the result object."""
    return compute_agencity(data=data, u=u, **kwargs)


def summarize(result) -> str:
    """Return a text summary for an AgencityResult."""
    return textual_analysis(result)
