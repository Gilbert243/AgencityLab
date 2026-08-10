import numpy as np

from agencitylab import analyze_agencity, compute_agencity


def test_analysis_smoke():
    t = np.linspace(0.0, 10.0, 101)
    u = np.sin(t)
    result = compute_agencity(u=u, xi=t, A_ref=1.0, tau=2.0, P_c=1.0)
    report = analyze_agencity(result)
    assert "summary" in report and "regime" in report
