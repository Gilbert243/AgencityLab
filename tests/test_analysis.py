import numpy as np
from agencitylab.api import compute_agencity
from agencitylab.analysis.reports import analyze_result

def test_analysis_smoke():
    t = np.linspace(0, 10, 100)
    u = np.sin(t)
    r = compute_agencity(t, u)
    report = analyze_result(r)
    assert "summary" in report and "regime" in report
