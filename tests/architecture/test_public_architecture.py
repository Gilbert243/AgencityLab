"""Architecture and compatibility gates for the 1.1.7 API consolidation."""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

import agencitylab as al
from agencitylab.api.compute import compute_agencity as api_compute_agencity
from agencitylab.config import AgencityConfig, DEFAULT_CONFIG

ROOT = Path(__file__).resolve().parents[2]


def _compute(func=al.compute_agencity, *, config=None):
    xi = np.arange(96, dtype=float)
    return func(
        u=np.sin(0.2 * xi) + 0.1 * np.cos(0.05 * xi),
        xi=xi,
        A_ref=1.0,
        tau=4.0,
        w=3.0,
        P_c=2.0,
        config=config,
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
    for specialized in ("minkowski_metric", "annihilation_operator", "simulate_flat_flrw", "export_pdf"):
        assert specialized not in al.__all__


def test_primary_scientific_namespaces_import_cleanly():
    names = (
        "agencitylab.analysis",
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


def test_published_specialized_top_level_alias_is_deprecated_not_broken():
    al.__dict__.pop("minkowski_metric", None)
    with pytest.warns(DeprecationWarning, match="agencitylab.gravity.minkowski_metric"):
        legacy = al.minkowski_metric
    from agencitylab.gravity import minkowski_metric

    assert legacy is minkowski_metric


def test_legacy_agencity_dynamics_cannot_execute_as_canonical():
    from agencitylab.dynamics import system

    state = system.AgencityState(X_star=1.0, A_star=0.2, M=0.3, O=0.4, P_c=2.0)
    with pytest.warns(DeprecationWarning), pytest.raises(RuntimeError, match=r"beta = J \* U"):
        system.beta_from_state(state)
    assert "agencity_rhs" not in importlib.import_module("agencitylab.dynamics").__all__
    assert not (ROOT / "agencitylab" / "dynamics" / "canonical").exists()


def test_legacy_rk4_forwards_to_single_authoritative_implementation():
    from agencitylab.dynamics.integrators import rk4_step as legacy_rk4
    from agencitylab.fields.numerics import rk4_step

    rhs = lambda t, y: -0.5 * y
    state = np.array([1.0, 2.0])
    expected = rk4_step(rhs, 0.0, state, 0.1)
    with pytest.warns(DeprecationWarning, match="fields.numerics.rk4_step"):
        actual = legacy_rk4(rhs, 0.0, state, 0.1)
    np.testing.assert_array_equal(actual, expected)


def test_legacy_physical_config_keys_are_metadata_only_and_do_not_change_compute():
    baseline = _compute()
    with pytest.warns(DeprecationWarning, match="ignored by computation"):
        configured = _compute(
            config={
                "temperature": 900.0,
                "crm_window": 99,
                "agencity_scale": 100.0,
                "epsilon": 1e-3,
            }
        )
    for name in ("M", "O", "D", "S", "J", "U", "beta", "b"):
        np.testing.assert_array_equal(getattr(configured, name), getattr(baseline, name))


def test_runtime_config_has_no_physical_defaults():
    for name in ("A_ref", "tau", "w", "P_c", "temperature", "crm_window", "epsilon"):
        assert not hasattr(DEFAULT_CONFIG, name)
    with pytest.warns(DeprecationWarning):
        cfg = AgencityConfig(temperature=300.0, tau=2.0)
    assert cfg.metadata["legacy_config"] == {"temperature": 300.0, "tau": 2.0}


def test_dead_packaged_configs_examples_and_root_entrypoint_are_removed():
    config_dir = ROOT / "agencitylab" / "config"
    assert not list(config_dir.glob("*.yaml"))
    assert not (ROOT / "agencitylab" / "examples").exists()
    assert not (ROOT / "main.py").exists()
