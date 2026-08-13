"""Isolated smoke checks used by the optional-extra CI matrix."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

from agencitylab import compute_agencity


def _result():
    xi = np.arange(256, dtype=float)
    u = np.sin(0.07 * xi)
    return compute_agencity(
        u,
        xi,
        A_ref=1.0,
        tau=16.0,
        w=16.0,
        P_c=2.0,
    )


def _assert_unit_correlation(value, dtype) -> None:
    precision = np.finfo(np.dtype(dtype))
    np.testing.assert_allclose(value, 1.0, rtol=0.0, atol=8.0 * precision.eps)


def smoke_scientific() -> None:
    from agencitylab.analysis.events import detect_dynamic_peaks

    dynamic_intensity = np.array([0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0])
    peaks = detect_dynamic_peaks(dynamic_intensity, prominence=1.5)
    np.testing.assert_array_equal(peaks, np.array([3]))


def smoke_data() -> None:
    result = _result()
    frame = result.to_dataframe()
    dataset = result.to_xarray()
    assert frame.shape[0] == len(result)
    assert dataset.sizes["xi"] == len(result)


def smoke_viz() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from agencitylab.api import visualize_agencity

    figure = visualize_agencity(_result(), kind="overview", show=False)
    assert figure is not None
    plt.close("all")


def smoke_export() -> None:
    from agencitylab.api import export_excel, export_pdf

    result = _result()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        workbook = export_excel(result, root / "result.xlsx")
        report = export_pdf("AgencityLab optional export smoke test", root / "report.pdf")
        assert workbook.is_file() and workbook.stat().st_size > 0
        assert report.is_file() and report.stat().st_size > 0


def smoke_numba() -> None:
    from agencitylab.backends import backend_capabilities, select_backend

    capabilities = backend_capabilities("numba")
    assert capabilities["available"] is True
    assert capabilities["status"] == "experimental"
    backend = select_backend("numba")
    derivative = backend["central_difference"](np.arange(16, dtype=float), 1.0)
    crm = np.asarray(
        backend["causal_moving_correlation"](
            1e-20 * np.array([0.0, 1.0, 0.0, 1.0]), window=2
        )
    )
    assert np.all(np.isfinite(derivative))
    _assert_unit_correlation(crm[-1], crm.dtype)


def smoke_jax() -> None:
    from agencitylab.backends import backend_capabilities, select_backend

    capabilities = backend_capabilities("jax")
    assert capabilities["available"] is True
    assert capabilities["status"] == "experimental"
    backend = select_backend("jax")
    crm = np.asarray(
        backend["causal_moving_correlation"](
            np.array([0.0, 1.0, 0.0, 1.0]), window=2
        )
    )
    assert np.all(np.isfinite(crm))
    _assert_unit_correlation(crm[-1], crm.dtype)


SMOKE_CHECKS = {
    "scientific": smoke_scientific,
    "data": smoke_data,
    "viz": smoke_viz,
    "export": smoke_export,
    "numba": smoke_numba,
    "jax": smoke_jax,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("extra", choices=sorted(SMOKE_CHECKS))
    args = parser.parse_args()
    SMOKE_CHECKS[args.extra]()
    print(f"optional extra '{args.extra}': PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
