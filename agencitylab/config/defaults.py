"""Software/runtime configuration without hidden physical defaults."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict
import warnings

from .modes import AgencityMode


# These historical keys are accepted only so old configuration dictionaries do
# not fail abruptly. They are recorded as metadata and never become scientific
# inputs to the canonical or extension equations.
_LEGACY_SCIENTIFIC_KEYS = frozenset(
    {
        "tau_threshold",
        "activity_window",
        "crm_window",
        "epsilon",
        "reduced_time_step",
        "use_riemann_metric",
        "metric_type",
        "agencity_scale",
        "temperature",
        "A_ref",
        "tau",
        "w",
        "P_c",
        "Gamma",
        "gamma",
        "lambda",
        "lambda_",
        "mu",
        "T_eff",
        "T_amb",
        "T_c",
        "xi",
        "G",
        "gravitational_constant",
    }
)


@dataclass(slots=True, init=False)
class AgencityConfig:
    """Runtime software options only.

    Physical/contextual quantities such as ``A_ref``, ``tau``, ``w``, ``P_c``,
    ``Gamma``, ``lambda``, ``mu``, temperatures, ``xi`` and ``G`` are not global
    configuration defaults. They belong to the scientific API that consumes
    them.

    Historical scientific-looking keys are accepted temporarily as deprecated
    compatibility metadata. They never alter a calculation.
    """

    mode: AgencityMode = AgencityMode.CANONICAL
    normalization_method: str = "A_ref"
    backend: str = "numpy"
    prefer_gpu: bool = False
    use_numba: bool = False
    use_jax: bool = False
    compute_signature: bool = True
    compute_multiscale: bool = True
    report_language: str = "en"
    streaming_enabled: bool = False
    batch_parallel: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        mode: AgencityMode | str = AgencityMode.CANONICAL,
        normalization_method: str = "A_ref",
        backend: str = "numpy",
        prefer_gpu: bool = False,
        use_numba: bool = False,
        use_jax: bool = False,
        compute_signature: bool = True,
        compute_multiscale: bool = True,
        report_language: str = "en",
        streaming_enabled: bool = False,
        batch_parallel: bool = False,
        metadata: Dict[str, Any] | None = None,
        **legacy_options,
    ):
        unexpected = sorted(set(legacy_options) - _LEGACY_SCIENTIFIC_KEYS)
        if unexpected:
            names = ", ".join(unexpected)
            raise TypeError(f"unexpected AgencityConfig option(s): {names}")

        self.mode = AgencityMode.from_value(mode)
        self.normalization_method = str(normalization_method)
        self.backend = str(backend)
        self.prefer_gpu = bool(prefer_gpu)
        self.use_numba = bool(use_numba)
        self.use_jax = bool(use_jax)
        self.compute_signature = bool(compute_signature)
        self.compute_multiscale = bool(compute_multiscale)
        self.report_language = str(report_language)
        self.streaming_enabled = bool(streaming_enabled)
        self.batch_parallel = bool(batch_parallel)
        self.metadata = dict(metadata or {})

        if legacy_options:
            warnings.warn(
                "Scientific/physical values in AgencityConfig are deprecated and "
                "ignored by computation. Pass physical parameters explicitly to "
                "the scientific API that owns them.",
                DeprecationWarning,
                stacklevel=2,
            )
            retained = dict(self.metadata.get("legacy_config", {}))
            retained.update(legacy_options)
            self.metadata["legacy_config"] = retained

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["mode"] = self.mode.value
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgencityConfig":
        payload = dict(data or {})
        metadata = payload.pop("metadata", {})
        metadata = dict(metadata) if isinstance(metadata, dict) else {}

        allowed = {
            "mode",
            "normalization_method",
            "backend",
            "prefer_gpu",
            "use_numba",
            "use_jax",
            "compute_signature",
            "compute_multiscale",
            "report_language",
            "streaming_enabled",
            "batch_parallel",
        }
        known = {key: payload.pop(key) for key in list(payload) if key in allowed}
        legacy = {
            key: payload.pop(key)
            for key in list(payload)
            if key in _LEGACY_SCIENTIFIC_KEYS
        }
        if payload:
            extra = dict(metadata.get("extra", {}))
            extra.update(payload)
            metadata["extra"] = extra
        return cls(metadata=metadata, **known, **legacy)

    def with_updates(self, **kwargs) -> "AgencityConfig":
        payload = self.to_dict()
        payload.update(kwargs)
        return AgencityConfig.from_dict(payload)


DEFAULT_CONFIG = AgencityConfig()
