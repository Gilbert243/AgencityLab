import numpy as np
import pytest

from agencitylab.api import compute_discrete_agencity
from agencitylab.core.discrete import volume2_first_difference, volume2_second_difference


def test_volume2_stencils_constant_linear_and_quadratic_including_boundaries():
    delta = 0.1
    t = np.arange(-1.0, 1.0 + delta / 2.0, delta)

    constant = np.full_like(t, 3.0)
    np.testing.assert_allclose(volume2_first_difference(constant, delta), 0.0, atol=1e-14)
    np.testing.assert_allclose(volume2_second_difference(constant, delta), 0.0, atol=1e-12)

    linear = 2.5 * t - 0.4
    np.testing.assert_allclose(volume2_first_difference(linear, delta), 2.5, atol=1e-13)
    np.testing.assert_allclose(volume2_second_difference(linear, delta), 0.0, atol=1e-12)

    quadratic = t**2
    np.testing.assert_allclose(volume2_first_difference(quadratic, delta), 2.0 * t, atol=1e-13)
    np.testing.assert_allclose(volume2_second_difference(quadratic, delta), 2.0, atol=1e-12)


def test_discrete_api_rejects_non_scalar_signal_shape():
    with pytest.raises(ValueError, match="one-dimensional"):
        compute_discrete_agencity(
            np.zeros((8, 1)),
            delta=0.1,
            A_ref=1.0,
            tau=1.0,
            w=0.5,
            P_c=1.0,
        )


def test_discrete_api_quadratic_uses_reduced_volume2_derivatives_exactly():
    delta = 0.1
    t = np.arange(0.0, 8.0 + delta / 2.0, delta)
    result = compute_discrete_agencity(
        t**2,
        delta=delta,
        t0=0.0,
        A_ref=4.0,
        tau=2.0,
        w=1.0,
        P_c=3.0,
    )
    # u*=t^2/4 and t*=t/2, so du*/dt*=t and d2u*/dt*2=2.
    # The endpoint formula is analytically exact for a quadratic; the tolerance
    # below only admits ordinary floating cancellation at the largest sample.
    np.testing.assert_allclose(result.X_star, t, rtol=0.0, atol=2e-13)
    np.testing.assert_allclose(result.A_star, 2.0, rtol=0.0, atol=5e-12)
    assert result.config["formulation"] == "volume2_discrete"
    assert result.config["delta_star"] == pytest.approx(delta / 2.0)


def test_sinus_interior_transfer_matches_volume2_amplitude_and_phase():
    delta = 0.05
    omega = 1.7
    t = np.arange(0.0, 6.0, delta)
    u = np.sin(omega * t)
    z = omega * delta

    X = volume2_first_difference(u, delta)
    A = volume2_second_difference(u, delta)

    expected_X = omega * (np.sin(z) / z) * np.cos(omega * t)
    expected_A = -omega**2 * (4.0 * np.sin(z / 2.0) ** 2 / z**2) * np.sin(omega * t)
    np.testing.assert_allclose(X[1:-1], expected_X[1:-1], rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(A[1:-1], expected_A[1:-1], rtol=2e-12, atol=2e-12)


def test_volume2_second_difference_is_not_silently_gradient_of_gradient():
    delta = 0.1
    t = np.arange(0.0, 6.0, delta)
    u = np.sin(1.3 * t)
    direct = volume2_second_difference(u, delta)
    successive = np.gradient(np.gradient(u, delta), delta)
    assert np.max(np.abs(direct - successive)) > 1e-3


def test_discrete_sinus_converges_second_order_with_documented_boundaries():
    errors_X = []
    errors_A = []
    for delta in (0.2, 0.1, 0.05, 0.025):
        t = np.arange(0.0, 2.0 * np.pi + delta / 2.0, delta)
        u = np.sin(t)
        errors_X.append(np.max(np.abs(volume2_first_difference(u, delta) - np.cos(t))))
        errors_A.append(np.max(np.abs(volume2_second_difference(u, delta) + np.sin(t))))

    for coarse, fine in zip(errors_X[:-1], errors_X[1:]):
        assert coarse / fine > 3.0
    for coarse, fine in zip(errors_A[:-1], errors_A[1:]):
        assert coarse / fine > 3.0


def test_discrete_pipeline_propagates_explicit_stencils_through_b():
    delta = 0.05
    t = np.arange(0.0, 10.0 + delta / 2.0, delta)
    result = compute_discrete_agencity(
        np.sin(0.8 * t),
        delta=delta,
        A_ref=2.0,
        tau=1.0,
        w=0.5,
        P_c=2.5,
    )

    np.testing.assert_allclose(result.S, np.hypot(result.M, result.O))
    np.testing.assert_allclose(
        result.D,
        np.hypot(result.X_star, result.A_star * result.X_star),
    )
    np.testing.assert_allclose(result.J, np.log((np.e + result.D) / (np.e + result.S)))
    valid = result.S > 0.0
    np.testing.assert_allclose(result.U[valid], (result.M[valid] + 1j * result.O[valid]) / result.S[valid])
    np.testing.assert_array_equal(result.U[~valid], 0.0j)
    np.testing.assert_allclose(result.beta[valid], result.J[valid] * result.U[valid])
    np.testing.assert_array_equal(result.beta[~valid], 0.0j)
    np.testing.assert_allclose(result.b, 2.5 * result.beta)


def _van_der_pol(t, mu=1.0):
    delta = float(t[1] - t[0])
    x = np.empty_like(t)
    v = np.empty_like(t)
    x[0], v[0] = 1.0, 0.0

    def rhs(state):
        xx, vv = state
        return np.array([vv, mu * (1.0 - xx**2) * vv - xx])

    for i in range(t.size - 1):
        y = np.array([x[i], v[i]])
        k1 = rhs(y)
        k2 = rhs(y + 0.5 * delta * k1)
        k3 = rhs(y + 0.5 * delta * k2)
        k4 = rhs(y + delta * k3)
        y_next = y + delta * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        x[i + 1], v[i + 1] = y_next
    return x


def _filtered_noise(size, seed=123):
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=size)
    out = np.empty(size)
    out[0] = raw[0]
    for i in range(1, size):
        out[i] = 0.94 * out[i - 1] + 0.12 * raw[i]
    return out


@pytest.mark.parametrize(
    "kind",
    ["constant", "sinus", "damped", "vdp", "unstable", "filtered_noise"],
)
def test_discrete_stress_signals_remain_finite_and_preserve_flux_identity(kind):
    delta = 0.05
    t = np.arange(0.0, 20.0 + delta / 2.0, delta)
    if kind == "constant":
        u = np.ones_like(t)
    elif kind == "sinus":
        u = np.sin(t)
    elif kind == "damped":
        u = np.exp(-0.08 * t) * np.sin(t)
    elif kind == "vdp":
        u = _van_der_pol(t)
    elif kind == "unstable":
        u = np.exp(0.04 * t) * np.sin(t)
    else:
        u = _filtered_noise(t.size)

    result = compute_discrete_agencity(
        u,
        delta=delta,
        A_ref=max(1.0, float(np.max(np.abs(u)))),
        tau=1.0,
        w=0.5,
        P_c=1.0,
    )
    for name in ("X_star", "A_star", "M", "O", "D", "S", "J"):
        assert np.all(np.isfinite(getattr(result, name)))
    for name in ("U", "beta", "b"):
        assert np.all(np.isfinite(getattr(result, name)))
    np.testing.assert_allclose(result.b, result.P_c * result.beta)


def test_discrete_zero_power_keeps_intrinsic_state_but_zeroes_observable_flux():
    delta = 0.05
    t = np.arange(0.0, 8.0 + delta / 2.0, delta)
    result = compute_discrete_agencity(
        np.sin(t),
        delta=delta,
        A_ref=1.0,
        tau=1.0,
        w=0.5,
        P_c=0.0,
    )
    assert np.any(np.abs(result.beta) > 0.0)
    np.testing.assert_array_equal(result.b, 0.0j)
