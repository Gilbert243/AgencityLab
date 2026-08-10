"""Fluent high-level pipeline for the stable AgencityLab API."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from agencitylab.config.runtime import get_runtime_config
from agencitylab.config.schema import validate_config

from .analyze import analyze_agencity, analyze_multiscale, analyze_signature, textual_analysis
from .compute import compute_agencity
from .presets import resolve_compute_config
from .validation import prepare_inputs, validate_metadata


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("PyYAML required for config loading") from exc

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("YAML config must be a dictionary")
    return data


def _flatten_dict(data, parent_key="", sep="."):
    out = {}
    for key, value in data.items():
        full_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            out.update(_flatten_dict(value, full_key, sep))
        else:
            out[full_key] = value
    return out


def _map_config(flat_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Map supported hierarchical configuration keys to compute configuration."""
    mapping = {
        "backend.name": "backend",
        "backend.prefer_gpu": "prefer_gpu",
        "normalization.method": "normalization_method",
    }
    return {target: flat_cfg[source] for source, target in mapping.items() if source in flat_cfg}


@dataclass
class AgencityPipeline:
    """Fluent ``data -> compute -> analyze -> report`` interface."""

    preset: str = "default"
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    xi: Any = None
    u: Any = None
    result: Optional[Any] = None
    analysis: Optional[Dict[str, Any]] = None
    report_text: Optional[str] = None

    def from_arrays(self, xi, u):
        self.xi, self.u = prepare_inputs(u=u, xi=xi)
        return self

    def from_signal(self, u, xi=None):
        self.xi, self.u = prepare_inputs(u=u, xi=xi)
        return self

    def reset(self):
        self.result = None
        self.analysis = None
        self.report_text = None
        return self

    def set_preset(self, preset: str):
        self.preset = str(preset)
        return self

    def set_tau(self, tau: float):
        """Compatibility alias for :meth:`set_characteristic_time`."""
        return self.set_characteristic_time(tau)

    def set_power(self, Pc):
        """Compatibility alias for :meth:`set_characteristic_power`."""
        return self.set_characteristic_power(Pc)

    def set_backend(self, backend: str = "auto", *, prefer_gpu: bool = False):
        self.config["backend"] = backend
        self.config["prefer_gpu"] = bool(prefer_gpu)
        return self

    def set_config(self, **kwargs):
        self.config.update(kwargs)
        return self

    def load_config(self, path: str | Path):
        flat_cfg = _flatten_dict(_load_yaml(path))
        self.config.update(_map_config(flat_cfg))
        return self

    def use_runtime_config(self):
        runtime_cfg = get_runtime_config()
        if runtime_cfg is not None:
            self.config.update(runtime_cfg.to_dict())
        return self

    def set_metadata(self, **kwargs):
        merged = dict(self.metadata)
        merged.update(kwargs)
        self.metadata = validate_metadata(merged)
        return self

    def set_unit(self, unit: str, *, kind: Optional[str] = None):
        self.metadata["unit"] = str(unit)
        if kind is not None:
            self.metadata["observable_kind"] = str(kind)
        return self

    def set_coordinate_unit(self, unit: str):
        self.metadata["coordinate_unit"] = str(unit)
        return self

    def set_power_unit(self, unit: str):
        self.metadata["power_unit"] = str(unit)
        return self

    def set_reference_amplitude(self, A_ref: float):
        self.metadata["reference_amplitude"] = float(A_ref)
        return self

    def set_characteristic_time(self, tau: float):
        self.metadata["characteristic_time"] = float(tau)
        return self

    def set_characteristic_power(self, Pc: float):
        self.metadata["characteristic_power"] = float(Pc)
        return self

    def set_activity_factor(self, A_fact: float):
        """Store legacy metadata without modifying canonical computation."""
        warnings.warn(
            "activity_factor is legacy metadata and no longer changes canonical computation",
            DeprecationWarning,
            stacklevel=2,
        )
        self.metadata["activity_factor"] = float(A_fact)
        return self

    def set_system_type(self, system_type: str):
        self.metadata["system_type"] = str(system_type)
        return self

    def set_mechanism(self, mechanism: str):
        self.metadata["mechanism"] = str(mechanism)
        return self

    def set_resolution_scale(self, value: float):
        """Store observational resolution metadata only.

        No smoothing or coarse-graining is inserted into the canonical pipeline.
        """
        self.metadata["resolution_scale"] = float(value)
        return self

    def _resolve_config(self) -> Dict[str, Any]:
        runtime_cfg = get_runtime_config()
        merged: Dict[str, Any] = {}
        if runtime_cfg is not None:
            merged.update(runtime_cfg.to_dict())
        merged.update(self.config)
        return validate_config(
            resolve_compute_config(self.preset, config=merged)
        ).to_dict()

    def compute(self, *, verbose: bool = False, **kwargs):
        if self.u is None:
            raise ValueError("No signal loaded")

        self.result = compute_agencity(
            u=self.u,
            xi=self.xi,
            preset=self.preset,
            config=self._resolve_config(),
            metadata=self.metadata,
            verbose=verbose,
            **kwargs,
        )
        return self

    def analyze(self, *, verbose: bool = False):
        if self.result is None:
            self.compute(verbose=verbose)
        signature = analyze_signature(self.result, verbose=verbose)
        multiscale = analyze_multiscale(self.result, verbose=verbose)
        self.analysis = analyze_agencity(
            self.result,
            signature=signature,
            multiscale=multiscale,
            verbose=verbose,
        )
        self.result.attach_analysis(self.analysis)
        self.result.signature = signature
        self.result.multiscale = multiscale
        return self

    def report_dict(self):
        if self.analysis is None:
            self.analyze()
        return self.analysis

    def report(self, *, refresh: bool = False):
        if self.result is None:
            self.compute()
        if self.report_text is None or refresh:
            self.report_text = textual_analysis(self.result)
        return self.report_text

    def run(self, *, verbose: bool = False):
        return self.compute(verbose=verbose).analyze(verbose=verbose).result

    @property
    def summary(self):
        return self.result.summary() if self.result is not None else {}

    @property
    def b(self):
        return self.result.b if self.result is not None else None

    @property
    def beta(self):
        return self.result.beta if self.result is not None else None

    def inspect(self):
        return {
            "has_data": self.u is not None,
            "has_result": self.result is not None,
            "config": dict(self.config),
            "metadata": dict(self.metadata),
        }


def pipeline() -> AgencityPipeline:
    """Return a new fluent pipeline instance."""
    return AgencityPipeline()
