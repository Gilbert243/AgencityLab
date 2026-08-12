"""Named numerical presets for Agencity field-physics research.

Presets are explicit numerical conventions, never universal physical constants.
"""

from __future__ import annotations

from agencitylab.models.field_extensions import ParameterProvenance, ParameterSource

from .potential import QuarticAgencityPotential


def dimensionless_benchmark() -> QuarticAgencityPotential:
    """Return the explicit ``lambda=1, mu=1`` dimensionless benchmark.

    This is a numerical benchmark in a dimensionless convention, not a universal
    physical parameter set. Provenance is stored on the returned potential.
    """
    provenance = ParameterProvenance(
        source=ParameterSource.DIMENSIONLESS_BENCHMARK,
        note="Numerical benchmark only; not a universal physical parameter.",
        reference="AgencityLab field-physics benchmark convention",
    )
    return QuarticAgencityPotential(
        lambda_=1.0,
        mu=1.0,
        lambda_provenance=provenance,
        mu_provenance=provenance,
        units_convention="dimensionless",
    )
