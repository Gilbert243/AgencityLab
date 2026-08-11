"""Reusable NumPy-only numerical infrastructure for spatial fields.

This subpackage is mathematically generic.  It intentionally defines no
Agencity-specific field physics and is not re-exported from ``agencitylab.fields``
in v1.1.x.
"""

from .boundaries import DirichletBoundary, NeumannBoundary, PeriodicBoundary
from .grid import UniformRectilinearGrid
from .integrators import rk4_step, velocity_verlet_step
from .operators import gradient, gradient_norm_squared, integrate_spatial, laplacian
from .stability import diffusion_cfl_limit, wave_cfl_limit

__all__ = [
    "DirichletBoundary",
    "NeumannBoundary",
    "PeriodicBoundary",
    "UniformRectilinearGrid",
    "diffusion_cfl_limit",
    "gradient",
    "gradient_norm_squared",
    "integrate_spatial",
    "laplacian",
    "rk4_step",
    "velocity_verlet_step",
    "wave_cfl_limit",
]
