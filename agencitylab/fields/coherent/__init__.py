"""Research references for coherent autonomous-field structures and topology.

This subpackage implements static/reference structures from Volume 2 of the
Agencity theory. It has scientific status ``research``: these helpers are not
an experimental validation of autonomous-field physics and they do not depend
on ``agencitylab.fields.dynamics``.

The one-dimensional domain wall is deliberately documented as a real-sector
(Z2) reference solution. The full complex quartic field has a connected U(1)
vacuum manifold, so that wall must not be interpreted as a generally stable
U(1) topological defect.
"""

from agencitylab.scientific_status import ScientificStatus

from .domain_wall import domain_wall_profile, domain_wall_residual
from .topology import field_zero_mask, phase_winding
from .vortex import vortex_field, vortex_radial_residual

SCIENTIFIC_STATUS = ScientificStatus.RESEARCH

__all__ = [
    "SCIENTIFIC_STATUS",
    "domain_wall_profile",
    "domain_wall_residual",
    "field_zero_mask",
    "phase_winding",
    "vortex_field",
    "vortex_radial_residual",
]
