import numpy as np

from agencitylab import compute_agencity
from agencitylab.api import AgencityStream, compute_agencity_spectrum, run_batch


_CANONICAL_FIELDS = ("X_star", "A_star", "M", "O", "D", "S", "J", "U", "beta", "b")


def _signal(size):
    xi = np.arange(size, dtype=float)
    u = np.sin(0.017 * xi) + 0.2 * np.cos(0.031 * xi)
    return xi, u


def _compute(xi, u, **overrides):
    kwargs = {"A_ref": 1.5, "tau": 64.0, "w": 64.0, "P_c": 2.5}
    kwargs.update(overrides)
    return compute_agencity(u=u, xi=xi, **kwargs)


def test_long_signal_pipeline_is_finite_and_preserves_canonical_identities():
    xi, u = _signal(100_000)

    result = _compute(xi, u)

    for field in _CANONICAL_FIELDS:
        values = getattr(result, field)
        assert values.shape == u.shape
        assert np.all(np.isfinite(values))

    np.testing.assert_allclose(result.S, np.hypot(result.M, result.O), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(result.beta, result.J * result.U, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(result.b, 2.5 * result.beta, rtol=0.0, atol=0.0)

    defined = result.S > 0.0
    np.testing.assert_allclose(np.abs(result.U[defined]), 1.0, rtol=2e-15, atol=2e-15)
    np.testing.assert_array_equal(result.U[~defined], 0.0)


def test_batch_preserves_order_per_item_physics_and_parallel_equivalence():
    xi, u = _signal(2048)
    items = [
        {
            "xi": xi,
            "u": u * scale,
            "A_ref": scale,
            "tau": tau,
            "w": tau,
            "P_c": power,
        }
        for scale, tau, power in ((1.0, 32.0, 2.0), (1.5, 64.0, 3.0), (2.0, 96.0, 5.0))
    ]

    serial = run_batch(items, analyze=False)
    parallel = run_batch(items, analyze=False, parallel=True, executor="thread", max_workers=2)

    assert [item.tau for item in serial] == [32.0, 64.0, 96.0]
    assert [item.P_c for item in serial] == [2.0, 3.0, 5.0]
    assert [item.A_ref for item in serial] == [1.0, 1.5, 2.0]

    for serial_item, parallel_item in zip(serial, parallel):
        for field in _CANONICAL_FIELDS:
            np.testing.assert_array_equal(
                getattr(parallel_item, field),
                getattr(serial_item, field),
            )


def test_full_history_streaming_matches_one_shot_pipeline_at_final_update():
    xi, u = _signal(4096)
    expected = _compute(xi, u)
    stream = AgencityStream(
        analyze=False,
        A_ref=1.5,
        tau=64.0,
        w=64.0,
        P_c=2.5,
    )

    actual = None
    for indices in np.array_split(np.arange(u.size), 8):
        actual = stream.update(u[indices], xi[indices])

    assert actual is not None
    assert stream.snapshot()["buffer_length"] == u.size
    for field in _CANONICAL_FIELDS:
        np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))


def test_multiscale_rows_match_independent_canonical_computations():
    xi, u = _signal(4096)
    scales = np.array([32.0, 48.0, 64.0, 96.0])

    spectrum = compute_agencity_spectrum(
        u,
        xi,
        scales,
        A_ref=1.5,
        P_c=2.5,
        windows=scales,
    )

    assert spectrum["b"].shape == (scales.size, u.size)
    for index, scale in enumerate(scales):
        expected = _compute(xi, u, tau=float(scale), w=float(scale))
        np.testing.assert_allclose(spectrum["b"][index], expected.b, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            spectrum["beta"][index], expected.beta, rtol=0.0, atol=0.0
        )
