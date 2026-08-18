"""Runtime regressions uncovered while eliminating repository-wide mypy debt."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from agencitylab.analysis.information.vopson import (
    C,
    SCIENTIFIC_STATUS,
    information_mass,
    vopson_mass_equivalent,
)
from agencitylab.api.analyze import analyze_information
from agencitylab.constants.physics import BOLTZMANN_CONSTANT, SPEED_OF_LIGHT
from agencitylab.models.config_model import AnalysisConfig
from agencitylab.models.dataset import AgencityDataset
from agencitylab.models.experiment import AgencityExperiment
from agencitylab.models.context import Context
from agencitylab.scientific_status import ScientificStatus
from agencitylab.utils.normalization import normalize


def test_vopson_helper_uses_numeric_physical_constants_without_changing_hypothesis():
    expected = BOLTZMANN_CONSTANT.value * 300.0 * np.log(2.0) / SPEED_OF_LIGHT.value**2

    assert SCIENTIFIC_STATUS is ScientificStatus.SPECULATIVE
    assert C == SPEED_OF_LIGHT.value
    assert information_mass(1.0, 300.0) == pytest.approx(expected)
    assert vopson_mass_equivalent(float(np.log(2.0)), 300.0) == pytest.approx(expected)
    assert information_mass(0.0, 300.0) == 0.0


@pytest.mark.parametrize(
    ("bits", "temperature"),
    [(-1.0, 300.0), (1.0, -1.0), (np.nan, 300.0), (1.0, np.inf)],
)
def test_vopson_helper_rejects_invalid_landauer_inputs(bits, temperature):
    with pytest.raises(ValueError):
        information_mass(bits, temperature)


def test_information_summary_verbose_api_is_runtime_valid(capsys):
    result = SimpleNamespace(b=np.array([0.5 + 0.0j, 1.0 + 0.5j, 0.75 - 0.25j]))
    summary = analyze_information(result, verbose=True)

    assert "entropy_b" in summary
    assert "density_b" in summary
    assert "[info]" in capsys.readouterr().out


def test_analysis_config_from_dict_uses_instance_defaults_with_slots():
    config = AnalysisConfig.from_dict({})
    defaults = AnalysisConfig()

    assert config.to_dict() == defaults.to_dict()


def test_experiment_summary_uses_dataset_summary_contract():
    dataset = AgencityDataset()
    summary = AgencityExperiment(dataset=dataset).summary()

    assert summary["dataset"]["n_signals"] == 0
    assert summary["dataset"]["n_samples_total"] == 0


def test_context_serialization_accepts_absent_signal_and_result():
    payload = Context().to_dict()

    assert payload["signal"] is None
    assert payload["result"] is None


def test_legacy_utils_normalize_alias_targets_current_normalization_api():
    normalized, reference = normalize(np.array([2.0, 4.0, 6.0]), A_ref=2.0)

    np.testing.assert_allclose(normalized, [1.0, 2.0, 3.0])
    assert float(reference) == 2.0
