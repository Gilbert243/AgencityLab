import numpy as np

from agencitylab import run_batch


def test_batch_accepts_legacy_list_pair_items():
    xi = np.arange(8.0)
    u = np.sin(xi)

    results = run_batch(
        [[xi, u]],
        A_ref=1.0,
        tau=2.0,
        P_c=1.0,
    )

    assert len(results) == 1
    np.testing.assert_array_equal(results[0].xi, xi)
    np.testing.assert_array_equal(results[0].u, u)
