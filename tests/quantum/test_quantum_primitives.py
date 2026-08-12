from __future__ import annotations

import numpy as np
import pytest

import agencitylab
from agencitylab.fields.physics import QuarticAgencityPotential, vacuum_amplitude
from agencitylab.quantum import (
    SCIENTIFIC_STATUS,
    agencity_uncertainty_lower_bound,
    annihilation_operator,
    broken_symmetry_vacuum_amplitude,
    creation_operator,
    fock_state,
    goldstone_angular_frequency,
    goldstone_mass_squared,
    goldstone_propagator,
    number_operator,
    occupation_expectation,
    one_loop_quartic_beta,
    radial_angular_frequency,
    radial_mass,
    radial_mass_squared,
    radial_propagator,
    truncated_commutator,
    truncation_commutator_defect,
    vacuum_state,
)
from agencitylab.scientific_status import ScientificStatus


def test_quantum_layer_is_explicitly_speculative_and_selected_api_is_public() -> None:
    assert SCIENTIFIC_STATUS is ScientificStatus.SPECULATIVE
    assert agencitylab.radial_mass_squared is radial_mass_squared
    assert agencitylab.annihilation_operator is annihilation_operator
    assert agencitylab.agencity_uncertainty_lower_bound is agencity_uncertainty_lower_bound


def test_broken_symmetry_modes_reuse_shared_potential_contract() -> None:
    potential = QuarticAgencityPotential(lambda_=2.0, mu=0.5)
    assert broken_symmetry_vacuum_amplitude(potential) == vacuum_amplitude(2.0, 0.5)
    assert radial_mass_squared(2.0) == 4.0
    assert radial_mass(2.0) == 2.0
    assert goldstone_mass_squared() == 0.0


@pytest.mark.parametrize("lambda_", [0.0, -1.0, np.nan, np.inf])
def test_radial_mode_requires_positive_finite_lambda(lambda_) -> None:
    with pytest.raises(ValueError):
        radial_mass_squared(lambda_)


def test_radial_and_goldstone_dispersion_relations() -> None:
    k = np.array([0.0, 3.0, 4.0])
    np.testing.assert_allclose(radial_angular_frequency(k, 2.0), np.sqrt(k**2 + 4.0))
    np.testing.assert_array_equal(goldstone_angular_frequency(k), k)
    with pytest.raises(ValueError):
        goldstone_angular_frequency(np.array([0.0, -1.0]))


def test_fock_vacuum_and_number_states() -> None:
    cutoff = 5
    annihilation = annihilation_operator(cutoff)
    creation = creation_operator(cutoff)
    number = number_operator(cutoff)
    vacuum = vacuum_state(cutoff)

    np.testing.assert_array_equal(annihilation @ vacuum, np.zeros(cutoff))
    np.testing.assert_array_equal(creation, annihilation.conj().T)

    for occupation in range(cutoff):
        state = fock_state(occupation, cutoff)
        np.testing.assert_allclose(number @ state, occupation * state)
        assert occupation_expectation(state) == pytest.approx(float(occupation))


def test_occupation_expectation_is_norm_independent() -> None:
    state = 3.0 * (fock_state(0, 4) + 1j * fock_state(2, 4))
    assert occupation_expectation(state) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        occupation_expectation(np.zeros(4))


def test_finite_fock_commutator_exposes_cutoff_defect() -> None:
    cutoff = 4
    commutator = truncated_commutator(cutoff)
    expected = np.eye(cutoff, dtype=complex)
    expected[-1, -1] = -(cutoff - 1)
    np.testing.assert_allclose(commutator, expected)

    defect = truncation_commutator_defect(cutoff)
    expected_defect = np.zeros((cutoff, cutoff), dtype=complex)
    expected_defect[-1, -1] = -cutoff
    np.testing.assert_allclose(defect, expected_defect, atol=1e-15)

    for occupation in range(cutoff - 1):
        state = fock_state(occupation, cutoff)
        np.testing.assert_allclose(commutator @ state, state)


def test_propagators_match_the_explicit_chapter_21_forms() -> None:
    k_squared = np.array([-2.0, 0.0, 5.0])
    epsilon = 0.25
    np.testing.assert_allclose(
        radial_propagator(k_squared, 2.0, epsilon=epsilon),
        1j / (k_squared - 4.0 + 1j * epsilon),
    )
    np.testing.assert_allclose(
        goldstone_propagator(k_squared, epsilon=epsilon),
        1j / (k_squared + 1j * epsilon),
    )


@pytest.mark.parametrize("epsilon", [0.0, -1.0, np.nan, np.inf])
def test_propagator_regulator_is_explicit_and_strictly_positive(epsilon) -> None:
    with pytest.raises(ValueError):
        goldstone_propagator(1.0, epsilon=epsilon)


def test_agencity_uncertainty_bound_and_zero_characteristic_power() -> None:
    assert agencity_uncertainty_lower_bound(
        characteristic_power=6.0,
        tau=3.0,
        hbar=2.0,
    ) == pytest.approx(2.0)
    assert agencity_uncertainty_lower_bound(
        characteristic_power=0.0,
        tau=3.0,
        hbar=2.0,
    ) == 0.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"characteristic_power": -1.0, "tau": 1.0, "hbar": 1.0}, "non-negative"),
        ({"characteristic_power": 1.0, "tau": 0.0, "hbar": 1.0}, "tau"),
        ({"characteristic_power": 1.0, "tau": 1.0, "hbar": 0.0}, "hbar"),
    ],
)
def test_agencity_uncertainty_bound_validates_explicit_parameters(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        agencity_uncertainty_lower_bound(**kwargs)


def test_one_loop_beta_returns_only_stated_leading_term() -> None:
    mu = 0.3
    assert one_loop_quartic_beta(mu) == pytest.approx(5.0 * mu**2 / (16.0 * np.pi**2))
    with pytest.raises(ValueError):
        one_loop_quartic_beta(0.0)
