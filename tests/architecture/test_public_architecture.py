"""Architecture gates for the first stable AgencityLab 1.0 API."""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

import agencitylab as al
from agencitylab.api.compute import compute_agencity as api_compute_agencity
from agencitylab.config import AgencityConfig, DEFAULT_CONFIG

ROOT = Path(__file__).resolve().parents[2]


def _compute(func=al.compute_agencity):
    xi = np.arange(96, dtype=float)
    return func(
        np.sin(0.2 * xi) + 0.1 * np.cos(0.05 * xi),
        xi,
        A_ref=1.0,
        tau=4.0,
        w=3.0,
        P_c=2.0,
    )


def test_top_level_keeps_reference_compute_and_exact_api_result():
    assert al.compute_agencity is api_compute_agencity
    top = _compute(al.compute_agencity)
    direct = _compute(api_compute_agencity)
    for name in ("u_star", "X_star", "A_star", "M", "O", "D", "S", "J", "U", "beta", "b"):
        np.testing.assert_array_equal(getattr(top, name), getattr(direct, name))


def test_top_level_discovery_is_small_and_namespace_first():
    assert "compute_agencity" in al.__all__
    assert "fields" in al.__all__
    assert "gravity" in al.__all__
    assert len(al.__all__) < 30
    for specialized in (
        "minkowski_metric",
        "annihilation_operator",
        "simulate_flat_flrw",
        "export_pdf",
        "pipeline",
        "run_batch",
    ):
        assert specialized not in al.__all__
        with pytest.raises(AttributeError):
            getattr(al, specialized)


def test_primary_scientific_namespaces_import_cleanly():
    names = (
        "agencitylab.analysis",
        "agencitylab.api",
        "agencitylab.fields",
        "agencitylab.thermodynamics",
        "agencitylab.gravity",
        "agencitylab.quantum",
        "agencitylab.applications",
        "agencitylab.applications.cosmology",
    )
    for name in names:
        assert importlib.import_module(name) is not None
    assert al.analysis is importlib.import_module("agencitylab.analysis")
    assert al.fields is importlib.import_module("agencitylab.fields")


def test_pre1_incorrect_dynamics_and_obsolete_model_are_absent():
    dynamics = importlib.import_module("agencitylab.dynamics")
    assert not (ROOT / "agencitylab" / "dynamics" / "system.py").exists()
    assert not (ROOT / "agencitylab" / "models" / "physical_system.py").exists()
    assert "AgencityState" not in dynamics.__dict__
    assert "agencity_rhs" not in dynamics.__dict__


def test_rk4_has_one_authoritative_location():
    dynamics_integrators = importlib.import_module("agencitylab.dynamics.integrators")
    from agencitylab.fields.numerics import rk4_step

    assert not hasattr(dynamics_integrators, "rk4_step")
    assert callable(rk4_step)


def test_runtime_config_rejects_physical_and_unknown_options():
    for name in ("A_ref", "tau", "w", "P_c", "temperature", "crm_window", "epsilon"):
        assert not hasattr(DEFAULT_CONFIG, name)

    with pytest.raises(TypeError, match="unexpected AgencityConfig option"):
        AgencityConfig.from_dict({"tau": 2.0})
    with pytest.raises(TypeError, match="unexpected AgencityConfig option"):
        AgencityConfig.from_dict({"made_up_option": True})


def test_compute_has_no_pre1_aliases_or_hidden_runtime_configuration():
    xi = np.arange(8.0)
    u = np.sin(xi)
    with pytest.raises(TypeError):
        al.compute_agencity(data=u, xi=xi, A_ref=1.0, tau=2.0, P_c=1.0)
    with pytest.raises(TypeError):
        al.compute_agencity(u, xi, A_ref=1.0, tau=2.0, Pc=1.0)
    with pytest.raises(TypeError):
        al.compute_agencity(u, xi, A_ref=1.0, tau=2.0, P_c=1.0, config={})


def test_stable_distribution_is_typed_and_contains_no_compatibility_module():
    assert (ROOT / "agencitylab" / "py.typed").is_file()
    assert not (ROOT / "agencitylab" / "_compat.py").exists()
    assert not (ROOT / "agencitylab" / "api" / "shortcuts.py").exists()


def test_no_pre1_deprecation_machinery_remains_in_package_source():
    matches = []
    for path in (ROOT / "agencitylab").rglob("*.py"):
        if "DeprecationWarning" in path.read_text(encoding="utf-8"):
            matches.append(path.relative_to(ROOT).as_posix())
    assert matches == []


def test_dead_packaged_configs_examples_and_root_entrypoint_are_absent():
    config_dir = ROOT / "agencitylab" / "config"
    assert not list(config_dir.glob("*.yaml"))
    assert not (ROOT / "agencitylab" / "examples").exists()
    assert not (ROOT / "main.py").exists()
