"""Public package entry point for AgencityLab 1.0.

The package root is intentionally small. Canonical scalar computation is
available directly through :func:`compute_agencity`; specialized functionality
lives in explicit scientific namespaces such as :mod:`agencitylab.fields` and
:mod:`agencitylab.gravity`.

AgencityLab 1.0 is the first stable public API contract. Repository snapshots
that preceded 1.0 are treated as development history and do not create legacy
aliases at the package root.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .api.compute import compute_agencity
from .exceptions import (
    AgencityError,
    AgencityValidationError,
    PhysicalParameterError,
    UnitValidationError,
)
from .models import (
    AgencityResult,
    DynamicalAgencityFieldSolution,
    DynamicalAgencityFieldState,
    ExperimentMetadata,
    ObservableAgencityFieldResult,
)
from .scientific_status import ScientificStatus
from .version import __version__

_LAZY_PUBLIC: dict[str, tuple[str, str | None]] = {
    "analysis": ("agencitylab.analysis", None),
    "api": ("agencitylab.api", None),
    "applications": ("agencitylab.applications", None),
    "extensions": ("agencitylab.extensions", None),
    "fields": ("agencitylab.fields", None),
    "gravity": ("agencitylab.gravity", None),
    "models": ("agencitylab.models", None),
    "quantum": ("agencitylab.quantum", None),
    "thermodynamics": ("agencitylab.thermodynamics", None),
    "analyze_agencity": ("agencitylab.api.analyze", "analyze_agencity"),
    "compute_agencity_field": ("agencitylab.fields", "compute_agencity_field"),
    "scientific_workflow": ("agencitylab.api.scientific", "scientific_workflow"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_PUBLIC.get(name)
    if target is None:
        raise AttributeError(f"module 'agencitylab' has no attribute {name!r}")

    module_name, attribute = target
    module = import_module(module_name)
    value = module if attribute is None else getattr(module, attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_PUBLIC))


__all__ = [
    "__version__",
    "compute_agencity",
    "ScientificStatus",
    "AgencityResult",
    "ExperimentMetadata",
    "ObservableAgencityFieldResult",
    "DynamicalAgencityFieldState",
    "DynamicalAgencityFieldSolution",
    "AgencityError",
    "AgencityValidationError",
    "PhysicalParameterError",
    "UnitValidationError",
    "analysis",
    "api",
    "applications",
    "extensions",
    "fields",
    "gravity",
    "models",
    "quantum",
    "thermodynamics",
    "analyze_agencity",
    "compute_agencity_field",
    "scientific_workflow",
]
