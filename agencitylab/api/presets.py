"""Diagnostic presets for high-level analysis helpers.

Presets configure diagnostics only. Canonical physical quantities and equations
are never selected through a preset.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


PRESETS: dict[str, dict[str, Any]] = {
    "default": {
        "analysis": {
            "event_threshold": 3.0,
            "transition_threshold": 2.0,
            "spectrum_component": "magnitude",
        },
        "description": "Default diagnostic behaviour with explicit physical inputs.",
    },
    "fast": {
        "analysis": {
            "event_threshold": 3.5,
            "transition_threshold": 2.5,
            "spectrum_component": "magnitude",
        },
        "description": "Lighter diagnostic settings; canonical equations are unchanged.",
    },
    "research": {
        "analysis": {
            "event_threshold": 2.5,
            "transition_threshold": 1.8,
            "spectrum_component": "phase",
        },
        "description": "More sensitive diagnostic settings; not a physics preset.",
    },
    "multiscale": {
        "analysis": {
            "event_threshold": 3.0,
            "transition_threshold": 2.0,
            "spectrum_component": "magnitude",
        },
        "description": "Diagnostic preset for explicit multiscale studies.",
    },
}


def list_presets() -> list[str]:
    return sorted(PRESETS)


def has_preset(name: str) -> bool:
    return str(name) in PRESETS


def get_preset(name: str = "default") -> dict[str, Any]:
    key = str(name)
    if key not in PRESETS:
        raise KeyError(f"Unknown preset '{key}'. Available: {list_presets()}")
    return deepcopy(PRESETS[key])


def resolve_analysis_config(
    preset: str | dict[str, Any] = "default",
    *,
    config: dict[str, Any] | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
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


def describe_preset(name: str = "default") -> dict[str, Any]:
    preset = get_preset(name)
    return {
        "name": name,
        "description": preset.get("description", ""),
        "analysis": preset["analysis"],
    }
