"""Experimental observable spatial Agencity fields.

This package intentionally exposes only the v1.1 observable-field orchestration.
Autonomous dynamical fields and PDE solvers remain future research work.
"""

from agencitylab.models.field_result import ObservableAgencityFieldResult

from .local_field import compute_agencity_field

# Compatibility alias for the historical placeholder name.  New code should use
# ObservableAgencityFieldResult so it cannot be confused with a future dynamical phi field.
AgencityField = ObservableAgencityFieldResult

__all__ = [
    "AgencityField",
    "ObservableAgencityFieldResult",
    "compute_agencity_field",
]
