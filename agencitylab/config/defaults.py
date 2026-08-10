from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict

from .modes import AgencityMode


@dataclass(slots=True)
class AgencityConfig:
    """Lightweight runtime configuration.

    Numerical and analysis controls live here, but the canonical physical equations
    do not read ``epsilon``, ``crm_window``, or similar settings as physical constants.
    """

    mode: AgencityMode = AgencityMode.CANONICAL
    normalization_method: str = "A_ref"

    tau_threshold: float = 0.5
    activity_window: int = 1
    crm_window: int = 1

    epsilon: float = 1e-12
    reduced_time_step: float = 1.0

    use_riemann_metric: bool = False
    metric_type: str = "identity"
    agencity_scale: float = 1.0
    temperature: float = 1.0

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

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["mode"] = self.mode.value
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgencityConfig":
        payload = dict(data or {})
        extra = dict(payload.get("metadata", {}))
        if "mode" in payload:
            payload["mode"] = AgencityMode.from_value(payload["mode"])

        allowed = {
            "mode",
            "normalization_method",
            "tau_threshold",
            "activity_window",
            "crm_window",
            "epsilon",
            "reduced_time_step",
            "use_riemann_metric",
            "metric_type",
            "agencity_scale",
            "temperature",
            "backend",
            "prefer_gpu",
            "use_numba",
            "use_jax",
            "compute_signature",
            "compute_multiscale",
            "report_language",
            "streaming_enabled",
            "batch_parallel",
            "metadata",
        }
        known = {}
        for key in list(payload.keys()):
            if key in allowed:
                known[key] = payload.pop(key)
        if payload:
            extra.setdefault("extra", {})
            if not isinstance(extra["extra"], dict):
                extra["extra"] = {}
            extra["extra"].update(payload)

        metadata = known.get("metadata", {})
        merged_metadata = dict(metadata) if isinstance(metadata, dict) else {}
        if extra:
            merged_metadata = {**merged_metadata, **extra}

        return cls(
            mode=known.get("mode", cls.mode),
            normalization_method=known.get("normalization_method", cls.normalization_method),
            tau_threshold=known.get("tau_threshold", cls.tau_threshold),
            activity_window=known.get("activity_window", cls.activity_window),
            crm_window=known.get("crm_window", cls.crm_window),
            epsilon=known.get("epsilon", cls.epsilon),
            reduced_time_step=known.get("reduced_time_step", cls.reduced_time_step),
            use_riemann_metric=known.get("use_riemann_metric", cls.use_riemann_metric),
            metric_type=known.get("metric_type", cls.metric_type),
            agencity_scale=known.get("agencity_scale", cls.agencity_scale),
            temperature=known.get("temperature", cls.temperature),
            backend=known.get("backend", cls.backend),
            prefer_gpu=known.get("prefer_gpu", cls.prefer_gpu),
            use_numba=known.get("use_numba", cls.use_numba),
            use_jax=known.get("use_jax", cls.use_jax),
            compute_signature=known.get("compute_signature", cls.compute_signature),
            compute_multiscale=known.get("compute_multiscale", cls.compute_multiscale),
            report_language=known.get("report_language", cls.report_language),
            streaming_enabled=known.get("streaming_enabled", cls.streaming_enabled),
            batch_parallel=known.get("batch_parallel", cls.batch_parallel),
            metadata=merged_metadata,
        )

    def with_updates(self, **kwargs) -> "AgencityConfig":
        payload = self.to_dict()
        payload.update(kwargs)
        return AgencityConfig.from_dict(payload)


DEFAULT_CONFIG = AgencityConfig()
