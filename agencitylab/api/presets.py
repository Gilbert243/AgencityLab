"""
Preset configurations for AgencityLab public API.

Presets define user-facing defaults for compute and analysis behaviors.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional


PRESETS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "default": {
        "compute": {
            "normalization_method": "A_ref",
            "A_ref": None,
            "tau_threshold": 0.5,
            "power_method": "rms",
        },
        "analysis": {
            "event_threshold": 3.0,
            "transition_threshold": 2.0,
            "spectrum_component": "magnitude",
        },
        "description": "Balanced default preset for general use.",
    },
    "fast": {
        "compute": {
            "normalization_method": "A_ref",
            "A_ref": None,
            "tau_threshold": 0.5,
            "power_method": "rms",
        },
        "analysis": {
            "event_threshold": 3.5,
            "transition_threshold": 2.5,
            "spectrum_component": "magnitude",
        },
        "description": "Lightweight preset for quick experiments.",
    },
    "research": {
        "compute": {
            "normalization_method": "A_ref",
            "A_ref": None,
            "tau_threshold": 0.5,
            "power_method": "variance",
        },
        "analysis": {
            "event_threshold": 2.5,
            "transition_threshold": 1.8,
            "spectrum_component": "phase",
        },
        "description": "More sensitive preset for scientific exploration.",
    },
    "multiscale": {
        "compute": {
            "normalization_method": "A_ref",
            "A_ref": None,
            "tau_threshold": 0.5,
            "power_method": "rms",
            "multiscale_scales": tuple(
                float(x) for x in __import__("numpy").exp(__import__("numpy").linspace(__import__("numpy").log(0.5), __import__("numpy").log(2.5), 12))
            ),
        },
        "analysis": {
            "event_threshold": 3.0,
            "transition_threshold": 2.0,
            "spectrum_component": "magnitude",
        },
        "description": "Preset tuned for multiscale studies.",
    },
}


def list_presets():
    return sorted(PRESETS.keys())


def has_preset(name: str) -> bool:
    return str(name) in PRESETS


def get_preset(name: str = "default") -> Dict[str, Dict[str, Any]]:
    """
    Return a deep copy of a preset definition.
    """
    name = str(name)
    if name not in PRESETS:
        raise KeyError(f"Unknown preset '{name}'. Available: {list_presets()}")
    return deepcopy(PRESETS[name])


def resolve_compute_config(
    preset: str | Dict[str, Any] = "default",
    *,
    config: Optional[Dict[str, Any]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Merge preset compute config with user config and overrides.
    """
    if isinstance(preset, dict):
        base = deepcopy(preset)
    else:
        base = deepcopy(get_preset(preset)["compute"])

    if config is not None:
        if not isinstance(config, dict):
            raise ValueError("config must be a dictionary or None")
        base.update(config)

    if overrides:
        base.update(overrides)

    return base


def resolve_analysis_config(
    preset: str | Dict[str, Any] = "default",
    *,
    config: Optional[Dict[str, Any]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Merge preset analysis config with user config and overrides.
    """
    if isinstance(preset, dict):
        base = deepcopy(preset)
    else:
        base = deepcopy(get_preset(preset)["analysis"])

    if config is not None:
        if not isinstance(config, dict):
            raise ValueError("config must be a dictionary or None")
        base.update(config)

    if overrides:
        base.update(overrides)

    return base


def describe_preset(name: str = "default") -> Dict[str, Any]:
    """
    Return preset metadata.
    """
    preset = get_preset(name)
    return {
        "name": name,
        "description": preset.get("description", ""),
        "compute": preset["compute"],
        "analysis": preset["analysis"],
    }