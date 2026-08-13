import numpy as np

from agencitylab import compute_agencity


def test_sine_signal_regression_preserves_canonical_identities():
    xi = np.linspace(0.0, 10.0, 101)
    u = np.sin(xi)
    result = compute_agencity(u=u, xi=xi, A_ref=1.0, tau=2.0, P_c=3.0)

    np.testing.assert_allclose(result.S, np.hypot(result.M, result.O))
    np.testing.assert_allclose(result.J, np.log((np.e + result.D) / (np.e + result.S)))
    np.testing.assert_allclose(result.b, 3.0 * result.beta)
