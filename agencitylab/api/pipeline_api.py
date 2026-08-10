"""
Pipeline façade for AgencityLab.

Fluent high-level interface for:
data → compute → analyze → report

Features
--------
- runtime config integration
- YAML config loading
- backend-aware execution
- canonical A_ref support
- canonical tau support
- canonical Pc support
- canonical A_fact support
- physical resolution-scale support
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field

from pathlib import Path

from typing import Any
from typing import Dict
from typing import Optional

from agencitylab.config.runtime import (
    get_runtime_config,
)

from agencitylab.config.schema import (
    validate_config,
)

from .compute import (
    compute_agencity,
)

from .analyze import (
    analyze_agencity,
    textual_analysis,
    analyze_signature,
    analyze_multiscale,
)

from .validation import (
    prepare_inputs,
    validate_metadata,
)

from .presets import (
    resolve_compute_config,
)


# ============================================================
# YAML LOADER
# ============================================================

def _load_yaml(
    path: str | Path
) -> Dict[str, Any]:

    try:
        import yaml

    except ImportError as exc:

        raise ImportError(
            "PyYAML required for config loading"
        ) from exc

    path = Path(path)

    if not path.exists():

        raise FileNotFoundError(
            f"Config file not found: {path}"
        )

    with open(path, "r") as f:

        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):

        raise ValueError(
            "YAML config must be a dictionary"
        )

    return data


# ============================================================
# CONFIG UTILITIES
# ============================================================

def _flatten_dict(
    d,
    parent_key="",
    sep=".",
):
    out = {}

    for k, v in d.items():

        key = (
            f"{parent_key}{sep}{k}"
            if parent_key
            else k
        )

        if isinstance(v, dict):

            out.update(
                _flatten_dict(
                    v,
                    key,
                    sep,
                )
            )

        else:

            out[key] = v

    return out


def _map_config(
    flat_cfg: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Map hierarchical YAML config to compute config.
    """

    cfg = {}

    # ========================================================
    # BACKEND
    # ========================================================

    if "backend.name" in flat_cfg:
        cfg["backend"] = flat_cfg["backend.name"]

    if "backend.prefer_gpu" in flat_cfg:
        cfg["prefer_gpu"] = flat_cfg[
            "backend.prefer_gpu"
        ]

    # ========================================================
    # NORMALIZATION
    # ========================================================

    if "normalization.method" in flat_cfg:
        cfg["normalization_method"] = flat_cfg[
            "normalization.method"
        ]

    # ========================================================
    # TAU
    # ========================================================

    if "dynamics.tau.threshold" in flat_cfg:
        cfg["tau_threshold"] = flat_cfg[
            "dynamics.tau.threshold"
        ]

    # ========================================================
    # POWER
    # ========================================================

    if "power.method" in flat_cfg:
        cfg["power_method"] = flat_cfg[
            "power.method"
        ]

    # ========================================================
    # MULTISCALE
    # ========================================================

    if "multiscale.enabled" in flat_cfg:
        cfg["multiscale_enabled"] = flat_cfg[
            "multiscale.enabled"
        ]

    # ========================================================
    # FIELD
    # ========================================================

    if "field.enabled" in flat_cfg:
        cfg["field_enabled"] = flat_cfg[
            "field.enabled"
        ]

    # ========================================================
    # MODEL
    # ========================================================

    if "model.core.formulation" in flat_cfg:
        cfg["model_type"] = flat_cfg[
            "model.core.formulation"
        ]

    return cfg


# ============================================================
# PIPELINE
# ============================================================

@dataclass
class AgencityPipeline:
    """
    Fluent scientific pipeline interface.
    """

    preset: str = "default"

    config: Dict[str, Any] = field(
        default_factory=dict
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    xi: Any = None

    u: Any = None

    result: Optional[Any] = None

    analysis: Optional[Dict[str, Any]] = None

    report_text: Optional[str] = None

    # ========================================================
    # INPUT
    # ========================================================

    def from_arrays(
        self,
        xi,
        u,
    ):

        self.xi, self.u = prepare_inputs(
            u=u,
            xi=xi,
        )

        return self

    def from_signal(
        self,
        u,
        xi=None,
    ):

        self.xi, self.u = prepare_inputs(
            u=u,
            xi=xi,
        )

        return self

    def reset(self):

        self.result = None

        self.analysis = None

        self.report_text = None

        return self

    # ========================================================
    # CONFIG
    # ========================================================

    def set_preset(
        self,
        preset: str,
    ):

        self.preset = str(preset)

        return self

    def set_tau(
        self,
        tau: float,
    ):

        self.config["tau"] = tau

        return self

    def set_power(
        self,
        Pc,
    ):

        self.config["Pc"] = Pc

        return self

    def set_backend(
        self,
        backend: str = "auto",
        *,
        prefer_gpu: bool = False,
    ):

        self.config["backend"] = backend

        self.config["prefer_gpu"] = bool(
            prefer_gpu
        )

        return self

    def set_config(
        self,
        **kwargs,
    ):

        self.config.update(kwargs)

        return self

    def load_config(
        self,
        path: str | Path,
    ):

        yaml_cfg = _load_yaml(path)

        flat_cfg = _flatten_dict(yaml_cfg)

        mapped_cfg = _map_config(flat_cfg)

        self.config.update(flat_cfg)

        self.config.update(mapped_cfg)

        return self

    def use_runtime_config(self):

        runtime_cfg = get_runtime_config()

        if runtime_cfg is not None:

            self.config.update(
                runtime_cfg.to_dict()
            )

        return self

    # ========================================================
    # METADATA
    # ========================================================

    def set_metadata(
        self,
        **kwargs,
    ):
        """
        Attach scientific metadata.
        """

        self.metadata.update(kwargs)

        self.metadata = validate_metadata(
            self.metadata
        )

        return self

    def set_unit(
        self,
        unit: str,
        *,
        kind: Optional[str] = None,
    ):
        """
        Convenience helper for A_ref inference.
        """

        self.metadata["unit"] = unit

        if kind is not None:

            self.metadata[
                "observable_kind"
            ] = kind

        return self

    def set_reference_amplitude(
        self,
        A_ref: float,
    ):
        """
        Explicit A_ref override.
        """

        self.metadata[
            "reference_amplitude"
        ] = float(A_ref)

        return self

    # ========================================================
    # STRUCTURAL PARAMETERS
    # ========================================================

    def set_characteristic_time(
        self,
        tau: float,
    ):
        """
        Structural characteristic time.
        """

        self.metadata[
            "characteristic_time"
        ] = float(tau)

        return self

    def set_characteristic_power(
        self,
        Pc: float,
    ):
        """
        Structural characteristic power.
        """

        self.metadata[
            "characteristic_power"
        ] = float(Pc)

        return self

    def set_activity_factor(
        self,
        A_fact: float,
    ):
        """
        Structural activity factor.
        """

        self.metadata[
            "activity_factor"
        ] = float(A_fact)

        return self

    # ========================================================
    # PHYSICAL CONTEXT
    # ========================================================

    def set_system_type(
        self,
        system_type: str,
    ):
        """
        Set system type.
        """

        self.metadata[
            "system_type"
        ] = str(system_type)

        return self

    def set_mechanism(
        self,
        mechanism: str,
    ):
        """
        Set dominant mechanism.
        """

        self.metadata[
            "mechanism"
        ] = str(mechanism)

        return self

    # ========================================================
    # RESOLUTION SCALE
    # ========================================================

    def set_resolution_scale(
        self,
        value: float,
    ):
        """
        Physical observation scale.

        NOT arbitrary denoising.
        """

        self.metadata[
            "resolution_scale"
        ] = float(value)

        return self

    # ========================================================
    # CONFIG RESOLUTION
    # ========================================================

    def _resolve_config(
        self,
        overrides: Dict[str, Any],
    ) -> Dict[str, Any]:

        runtime_cfg = get_runtime_config()

        merged = {}

        if runtime_cfg is not None:

            merged.update(
                runtime_cfg.to_dict()
            )

        merged.update(self.config)

        merged.update(overrides)

        cfg = resolve_compute_config(
            self.preset,
            config=merged,
        )

        cfg = validate_config(
            cfg
        ).to_dict()

        if "backend" not in cfg:
            cfg["backend"] = "auto"

        if "prefer_gpu" not in cfg:
            cfg["prefer_gpu"] = False

        return cfg

    # ========================================================
    # COMPUTE
    # ========================================================

    def compute(
        self,
        *,
        verbose: bool = False,
        **kwargs,
    ):

        if self.u is None:

            raise ValueError(
                "No signal loaded."
            )

        cfg = self._resolve_config(
            kwargs
        )

        if verbose:

            print(
                "[pipeline] Computing..."
            )

            print(
                "[pipeline] "
                f"backend = "
                f"{cfg.get('backend')}"
            )

        self.result = compute_agencity(

            data=self.u,

            xi=self.xi,

            preset=self.preset,

            config=cfg,

            metadata=self.metadata,

            resolution_scale=self.metadata.get(
                "resolution_scale",
                None,
            ),

            verbose=verbose,
        )

        return self

    # ========================================================
    # ANALYZE
    # ========================================================

    def analyze(
        self,
        *,
        verbose: bool = False,
    ):

        if self.result is None:

            self.compute(
                verbose=verbose
            )

        sig = analyze_signature(
            self.result,
            verbose=verbose,
        )

        ms = analyze_multiscale(
            self.result,
            verbose=verbose,
        )

        self.analysis = analyze_agencity(
            self.result,
            signature=sig,
            multiscale=ms,
            verbose=verbose,
        )

        self.result.attach_analysis(
            self.analysis
        )

        self.result.signature = sig

        self.result.multiscale = ms

        return self

    # ========================================================
    # REPORT
    # ========================================================

    def report_dict(self):

        if self.analysis is None:

            self.analyze()

        return self.analysis

    def report(
        self,
        *,
        refresh: bool = False,
    ):

        if self.result is None:

            self.compute()

        if (
            self.report_text is None
            or refresh
        ):

            self.report_text = textual_analysis(
                self.result
            )

        return self.report_text

    # ========================================================
    # EXECUTION
    # ========================================================

    def run(
        self,
        *,
        verbose: bool = False,
    ):

        return (
            self.compute(
                verbose=verbose
            )
            .analyze(
                verbose=verbose
            )
            .result
        )

    # ========================================================
    # QUICK ACCESS
    # ========================================================

    @property
    def summary(self):

        return (
            self.result.summary()
            if self.result
            else {}
        )

    @property
    def b(self):

        return (
            self.result.b
            if self.result
            else None
        )

    @property
    def beta(self):

        return (
            self.result.beta
            if self.result
            else None
        )

    # ========================================================
    # DEBUG
    # ========================================================

    def inspect(self):

        return {

            "has_data":
                self.u is not None,

            "has_result":
                self.result is not None,

            "config":
                dict(self.config),

            "metadata":
                dict(self.metadata),
        }


# ============================================================
# FACTORY
# ============================================================

def pipeline() -> AgencityPipeline:

    return AgencityPipeline()