"""Cross-layer release gates for speculative quantum and cosmology extensions."""

from __future__ import annotations

import numpy as np
import pytest

import agencitylab
from agencitylab.applications import cosmology
from agencitylab.fields.physics import QuarticAgencityPotential, vacuum_amplitude
import agencitylab.quantum as quantum
from agencitylab.quantum import SCIENTIFIC_STATUS as QUANTUM_STATUS
from agencitylab.scientific_status import ScientificStatus


def test_speculative_apis_are_explicitly_namespaced() -> None:
    assert QUANTUM_STATUS is ScientificStatus.SPECULATIVE
    assert cosmology.SCIENTIFIC_STATUS is ScientificStatus.SPECULATIVE
    assert callable(quantum.radial_mass_squared)
    assert callable(quantum.annihilation_operator)
    assert callable(quantum.agencity_uncertainty_lower_bound)
    assert callable(cosmology.simulate_flat_flrw)
    assert cosmology.FlatFLRWSolution.__module__.startswith(
        "agencitylab.applications.cosmology"
    )
    assert not hasattr(agencitylab, "radial_mass_squared")
    assert not hasattr(agencitylab, "simulate_flat_flrw")


def test_shared_quartic_potential_connects_classical_quantum_and_cosmology() -> None:
    potential = QuarticAgencityPotential(lambda_=2.0, mu=0.5)
    vacuum = vacuum_amplitude(potential.lambda_, potential.mu)

    assert vacuum == pytest.approx(2.0)
    assert quantum.radial_mass_squared(potential.lambda_) == pytest.approx(4.0)
    assert float(cosmology.homogeneous_energy_density(vacuum, 0.0, potential)) == pytest.approx(
        -2.0
    )


def test_broken_quartic_negative_vacuum_is_not_repaired_for_flrw() -> None:
    potential = QuarticAgencityPotential(lambda_=1.0, mu=1.0)
    vacuum = vacuum_amplitude(1.0, 1.0)
    rho = float(cosmology.homogeneous_energy_density(vacuum, 0.0, potential))

    assert rho == pytest.approx(-0.25)
    with pytest.raises(ValueError, match="negative rho"):
        cosmology.initial_hubble_from_friedmann(
            rho,
            gravitational_constant=1.0,
            branch="expanding",
        )


def test_quantum_fock_cutoff_defect_is_visible_not_hidden() -> None:
    cutoff = 5
    annihilation = quantum.annihilation_operator(cutoff)
    creation = annihilation.conj().T
    commutator = annihilation @ creation - creation @ annihilation

    np.testing.assert_allclose(np.diag(commutator)[:-1], 1.0)
    assert commutator[-1, -1] == pytest.approx(-(cutoff - 1))


def test_uncertainty_bound_preserves_zero_characteristic_power() -> None:
    assert quantum.agencity_uncertainty_lower_bound(
        characteristic_power=0.0,
        tau=2.0,
        hbar=1.0,
    ) == 0.0


def test_flrw_solver_reports_constraint_instead_of_projecting_it() -> None:
    potential = QuarticAgencityPotential(lambda_=-1.0, mu=1.0)
    solution = cosmology.simulate_flat_flrw(
        phi0=0.2 + 0.1j,
        phi_dot0=0.0,
        scale_factor0=1.0,
        potential=potential,
        gravitational_constant=0.01,
        dt=0.005,
        steps=20,
        branch="expanding",
    )

    assert solution.scientific_status is ScientificStatus.SPECULATIVE
    np.testing.assert_allclose(solution.friedmann_residual[0], 0.0, atol=1e-15)
    assert solution.friedmann_residual.shape == solution.times.shape


def test_canonical_scalar_flux_contract_is_unchanged() -> None:
    xi = np.arange(64.0)
    result = agencitylab.compute_agencity(
        u=np.sin(0.2 * xi),
        xi=xi,
        A_ref=1.0,
        tau=4.0,
        w=3.0,
        P_c=2.0,
    )

    np.testing.assert_allclose(result.b, result.P_c * result.beta)
    mask = result.S == 0.0
    np.testing.assert_array_equal(result.beta[mask], 0.0j)
