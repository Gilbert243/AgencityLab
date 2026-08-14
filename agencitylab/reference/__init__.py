"""Scientific reference inputs for AgencityLab.

``signals`` generate observables, ``datasets`` manage materialized data, and
``scenarios`` bind an observable to explicit context before delegating to the
canonical public computation API.
"""

from __future__ import annotations

from . import datasets, scenarios, signals

__all__ = ["signals", "datasets", "scenarios"]
