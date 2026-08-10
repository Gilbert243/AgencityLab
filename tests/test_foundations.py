from importlib.metadata import version

import agencitylab
from agencitylab import compute_agencity
from agencitylab.api import compute_agencity as api_compute_agencity


def test_runtime_version_matches_package_metadata():
    assert agencitylab.__version__ == version("agencitylab")


def test_public_compute_api_is_exposed_consistently():
    assert compute_agencity is api_compute_agencity
    assert "compute_agencity" in agencitylab.__all__
    assert callable(compute_agencity)
