"""Software and diagnostic presets for the public API.

Compute presets never provide physical values such as ``A_ref``, ``tau``,
``w`` or ``P_c``. Analysis thresholds remain diagnostic choices rather than
universal scientific constants.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional


PRESETS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "default": {
        "compute": {"normalization_method": "A_ref"},
        "analysis": {
            "event_threshold": 3.0,
            "transition_threshold": 2.0,
            "spectrum_component": "magnitude",
        },
        "description": "Default software behaviour with explicit physical inputs.",
    },
    "fast": {
        "compute": {"normalization_method": "A_ref"},
        "analysis": {
            "event_threshold": 3.5,
            "transition_threshold": 2.5,
            "spectrum_component": "magnitude",
        },
        "description": "Lighter diagnostic settings; canonical equations are unchanged.",
    },
    "research": {
        "compute": {"normalization_method": "A_ref"},
        "analysis": {
            "event_threshold": 2.5,
            "transition_threshold": 1.8,
            "spectrum_component": "phase",
        },
        "description": "More sensitive diagnostic settings; not a physics preset.",
    },
    "multiscale": {
        "compute": {"normalization_method": "A_ref"},
        "analysis": {
            "event_threshold": 3.0,
            "transition_threshold": 2.0,
            "spectrum_component": "magnitude",
        },
        "description": "Diagnostic preset for explicit multiscale studies.",
    },
}


def list_presets():
    return sorted(PRESETS.keys())


def has_preset(name: str) -> bool:
    return str(name) in PRESETS


def get_preset(name: str = "default") -> Dict[str, Dict[str, Any]]:
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
    preset = get_preset(name)
    return {
        "name": name,
        "description": preset.get("description", ""),
        "compute": preset["compute"],
        "analysis": preset["analysis"],
    }
