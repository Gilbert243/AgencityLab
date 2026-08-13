import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agencitylab import compute_agencity
from agencitylab.api import analyze_agencity, compute_agencity_spectrum
from agencitylab.api.visualize import visualize_agencity, visualize_multiscale_spectrum


def _result():
    xi = np.linspace(0.0, 20.0, 401)
    result = compute_agencity(
        u=np.sin(xi),
        xi=xi,
        A_ref=1.0,
        tau=2.0,
        P_c=2.0,
        coordinate_unit="s",
        power_unit="W",
    )
    return result, analyze_agencity(result)


def test_overview_exposes_the_canonical_pipeline():
    result, _ = _result()
    figure = visualize_agencity(result, kind="overview", show=False)
    assert len(figure.axes) == 6
    assert figure.axes[2].get_legend() is not None
    plt.close(figure)


def test_geometry_uses_intrinsic_beta_coordinates():
    result, analysis = _result()
    figure = visualize_agencity(result, kind="geometry", analysis=analysis, show=False)
    line = figure.axes[0].lines[0]
    np.testing.assert_allclose(line.get_xdata(), result.beta.real)
    np.testing.assert_allclose(line.get_ydata(), result.beta.imag)
    assert "beta" in figure.axes[0].get_title().lower()
    plt.close(figure)


def test_diagnostic_figure_consumes_analysis_without_creating_thresholds():
    result, analysis = _result()
    assert analysis["real_agencity"]["status"] == "undetermined"
    figure = visualize_agencity(
        result,
        kind="diagnostics",
        analysis=analysis,
        show=False,
    )
    assert len(figure.axes) == 4
    assert "undetermined" in figure._suptitle.get_text()
    plt.close(figure)


def test_timeseries_handles_complex_beta_and_b_explicitly():
    result, _ = _result()
    figure = visualize_agencity(result, kind="timeseries", show=False)
    assert len(figure.axes[1].lines) == 3
    assert len(figure.axes[2].lines) == 3
    plt.close(figure)


def test_frequency_spectrum_accepts_complex_flux():
    result, _ = _result()
    figure = visualize_agencity(result, kind="frequency_spectrum", show=False)
    assert len(figure.axes[0].lines) == 1
    plt.close(figure)


def test_multiscale_figure_uses_theoretical_b_tau_spectrum():
    result, _ = _result()
    spectrum = compute_agencity_spectrum(
        result.u,
        result.xi,
        [1.0, 2.0, 3.0],
        A_ref=1.0,
        P_c=2.0,
        return_full=True,
    )
    figure = visualize_multiscale_spectrum(spectrum, show=False)
    assert "b(t, tau)" in figure.axes[0].get_title()
    plt.close(figure)
