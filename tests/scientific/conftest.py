import pytest

from benchmarks.scientific.reference_bench import reference_suite


@pytest.fixture(scope="session")
def scientific_cases():
    return reference_suite()


@pytest.fixture(scope="session")
def scientific_results(scientific_cases):
    return {name: case.compute() for name, case in scientific_cases.items()}
