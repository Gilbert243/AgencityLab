"""Spatial and dynamical Agencity field interfaces."""

from agencitylab.models.field_result import ObservableAgencityFieldResult

from . import coherent as _coherent
from . import dynamics as _dynamics
from . import effective_beta as _effective_beta
from . import numerics as _numerics
from . import physics as _physics
from .coherent import *  # noqa: F401,F403
from .dynamics import *  # noqa: F401,F403
from .effective_beta import *  # noqa: F401,F403
from .local_field import compute_agencity_field
from .numerics import *  # noqa: F401,F403
from .physics import *  # noqa: F401,F403

_MODULES = (_coherent, _dynamics, _effective_beta, _numerics, _physics)
__all__ = ["ObservableAgencityFieldResult", "compute_agencity_field"]
__all__ += [
    name
    for module in _MODULES
    for name in module.__all__
    if name != "SCIENTIFIC_STATUS" and name not in __all__
]

del _MODULES
