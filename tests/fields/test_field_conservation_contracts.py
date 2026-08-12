from __future__ import annotations

import numpy as np
import pytest

from agencitylab.fields.physics import (
    FLAT_FIELD_METRIC_SIGNATURE,
    QuarticAgencityPotential,
    appendix_b_beta_energy_momentum_tensor,
    appendix_b_beta_equation_residual,
    appendix_b_beta_lagrangian_density,
    appendix_b_beta_noether_current,
    flat_energy_momentum_tensor,
    flat_field_lagrangian_density,
    flat_field_minkowski_metric,
    phase_noether_current,
    radial_equation_residual,
    u1_noether_current,
)


def _potential() -> QuarticAgencityPotential:
    return QuarticAgencityPotential(lambda_=2.0, mu=0.5)


def test_chapter16_metric_signature_is_explicit() -> None:
    assert FLAT_FIELD_METRIC_SIGNATURE == (1, -1, -1, -1)
    np.testing.assert_array_equal(
        flat_field_minkowski_metric(),
        np.diag([1.0, -1.0, -1.0, -1.0]),
    )


def test_flat_lagrangian_uses_plus_minus_minus_minus_contraction() -> None:
    phi = np.array([1.0 + 0.5j])
    derivatives = np.array([[2.0 + 1.0j, 1.0, -0.5j, 0.25]])
    potential = _potential()
    contraction = (
        np.abs(derivatives[:, 0]) ** 2
        - np.abs(derivatives[:, 1]) ** 2
        - np.abs(derivatives[:, 2]) ** 2
        - np.abs(derivatives[:, 3]) ** 2
    )
    expected = 0.5 * contraction - potential.value(phi)
    np.testing.assert_allclose(
        flat_field_lagrangian_density(phi, derivatives, potential),
        expected,
    )


def test_energy_momentum_tensor_is_symmetric_and_u1_invariant() -> None:
    phi = np.array([0.7 + 0.2j])
    derivatives = np.array([[0.3 + 0.4j, -0.2j, 0.1, 0.05 + 0.1j]])
    potential = _potential()
    tensor = flat_energy_momentum_tensor(phi, derivatives, potential)
    np.testing.assert_allclose(tensor, np.swapaxes(tensor, -1, -2))

    phase = np.exp(0.81j)
    rotated = flat_energy_momentum_tensor(
        phase * phi,
        phase * derivatives,
        potential,
    )
    np.testing.assert_allclose(rotated, tensor, atol=1e-14, rtol=1e-14)


def test_noether_current_matches_amplitude_phase_identity() -> None:
    radius = np.array([1.7])
    theta_derivatives = np.array([[0.4, -0.2, 0.1, 0.3]])
    theta = 0.63
    phi = radius * np.exp(1j * theta)
    derivatives = 1j * phi[..., np.newaxis] * theta_derivatives

    direct = u1_noether_current(phi, derivatives)
    decomposed = phase_noether_current(radius, theta_derivatives)
    np.testing.assert_allclose(direct, decomposed, atol=1e-14, rtol=1e-14)


def test_radial_equation_residual_uses_existing_potential_gradient() -> None:
    potential = _potential()
    radius = np.array([0.0, 1.0, 2.0])
    box = np.array([0.2, -0.1, 0.4])
    phase_contraction = np.array([0.3, -0.2, 0.5])
    expected = box - radius * phase_contraction + potential.gradient(radius)
    np.testing.assert_allclose(
        radial_equation_residual(radius, box, phase_contraction, potential),
        expected,
    )


def test_appendix_b_equation_is_kept_distinct_with_pc_squared() -> None:
    potential = _potential()
    beta = np.array([0.2 + 0.3j, -0.4j])
    box = np.array([0.1 - 0.2j, 0.3 + 0.1j])
    power = 2.5
    expected = box + potential.gradient(beta) / power**2
    np.testing.assert_allclose(
        appendix_b_beta_equation_residual(beta, box, potential, P_c=power),
        expected,
    )


def test_appendix_b_lagrangian_tensor_and_current_scale_by_pc_squared() -> None:
    potential = _potential()
    beta = np.array([0.8 * np.exp(0.4j)])
    theta_derivatives = np.array([[0.3, 0.1, -0.2, 0.05]])
    derivatives = 1j * beta[..., np.newaxis] * theta_derivatives
    power = 3.0

    base_lagrangian_kinetic = 0.5 * (
        np.abs(derivatives[:, 0]) ** 2
        - np.abs(derivatives[:, 1]) ** 2
        - np.abs(derivatives[:, 2]) ** 2
        - np.abs(derivatives[:, 3]) ** 2
    )
    expected_lagrangian = power**2 * base_lagrangian_kinetic - potential.value(beta)
    np.testing.assert_allclose(
        appendix_b_beta_lagrangian_density(beta, derivatives, potential, P_c=power),
        expected_lagrangian,
    )

    current = appendix_b_beta_noether_current(beta, derivatives, P_c=power)
    base_current = phase_noether_current(np.abs(beta), theta_derivatives)
    np.testing.assert_allclose(current, power**2 * base_current)

    tensor = appendix_b_beta_energy_momentum_tensor(
        beta,
        derivatives,
        potential,
        P_c=power,
    )
    np.testing.assert_allclose(tensor, np.swapaxes(tensor, -1, -2))


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
def test_appendix_b_pc_must_be_positive_where_source_divides_by_pc(bad: float) -> None:
    potential = _potential()
    with pytest.raises(ValueError):
        appendix_b_beta_equation_residual(
            np.array([0.1]),
            np.array([0.0]),
            potential,
            P_c=bad,
        )
