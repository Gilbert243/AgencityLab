import numpy as np

import agencitylab.fields as fields
import agencitylab.fields.dynamics as dynamics
import agencitylab.gravity as gravity
from agencitylab.fields.numerics import UniformRectilinearGrid
from agencitylab.fields.physics import QuarticAgencityPotential


def test_fields_dynamics_documentation_matches_public_exports():
    assert "not re-exported" not in (dynamics.__doc__ or "")
    assert fields.simulate_klein_gordon is dynamics.simulate_klein_gordon
    assert (
        fields.simulate_dissipative_klein_gordon
        is dynamics.simulate_dissipative_klein_gordon
    )
    assert fields.simulate_tdgl is dynamics.simulate_tdgl
    assert dynamics.FLAT_FIELD_METRIC_SIGNATURE == (1, -1, -1, -1)
    assert gravity.GRAVITY_METRIC_SIGNATURE == (-1, 1, 1, 1)
    assert dynamics.FLAT_FIELD_METRIC_SIGNATURE != gravity.GRAVITY_METRIC_SIGNATURE


def test_field_solution_records_flat_metric_convention():
    grid = UniformRectilinearGrid(shape=(8,), spacings=(0.25,))
    potential = QuarticAgencityPotential(lambda_=-1.0, mu=1.0)
    solution = fields.simulate_klein_gordon(
        np.zeros(grid.shape),
        np.zeros(grid.shape),
        grid,
        potential,
        dt=0.01,
        n_steps=1,
    )
    assert tuple(solution.solver_metadata["metric_signature"]) == (1, -1, -1, -1)
    assert solution.solver_metadata["metric_convention"] == "Chapter 16 flat field (+,-,-,-)"
