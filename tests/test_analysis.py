import numpy as np

from agencitylab import analyze_agencity, compute_agencity


def test_analysis_smoke():
    t = np.linspace(0, 10, 100)
    u = np.sin(t)
    result = compute_agencity(u=u, xi=t)
    report = analyze_agencity(result)
    assert "summary" in report and "regime" in report
