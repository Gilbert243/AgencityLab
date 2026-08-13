"""Fluent high-level pipeline for the stable AgencityLab API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agencitylab.models import AgencityResult

from .analyze import analyze_agencity, analyze_multiscale, analyze_signature, textual_analysis
from .compute import compute_agencity
from .validation import prepare_inputs, validate_metadata


@dataclass(slots=True)
class AgencityPipeline:
    """Fluent ``data -> compute -> analyze -> report`` interface.

    The pipeline stores explicit metadata and workflow artifacts. It does not
    maintain a second hidden scientific configuration system.
    """

    metadata: dict[str, Any] = field(default_factory=dict)
    xi: Any = None
    u: Any = None
    result: AgencityResult | None = None
    analysis: dict[str, Any] | None = None
    signature: dict[str, Any] | None = None
    multiscale: Any = None
    report_text: str | None = None

    def from_arrays(self, xi, u) -> "AgencityPipeline":
        self.xi, self.u = prepare_inputs(u=u, xi=xi)
        return self

    def from_signal(self, u, xi=None) -> "AgencityPipeline":
        self.xi, self.u = prepare_inputs(u=u, xi=xi)
        return self

    def reset(self) -> "AgencityPipeline":
        self.result = None
        self.analysis = None
        self.signature = None
        self.multiscale = None
        self.report_text = None
        return self

    def set_metadata(self, **kwargs: Any) -> "AgencityPipeline":
        merged = dict(self.metadata)
        merged.update(kwargs)
        self.metadata = validate_metadata(merged)
        return self

    def set_unit(self, unit: str, *, kind: str | None = None) -> "AgencityPipeline":
        self.metadata["unit"] = str(unit)
        if kind is not None:
            self.metadata["observable_kind"] = str(kind)
        return self

    def set_coordinate_unit(self, unit: str) -> "AgencityPipeline":
        self.metadata["coordinate_unit"] = str(unit)
        return self

    def set_power_unit(self, unit: str) -> "AgencityPipeline":
        self.metadata["power_unit"] = str(unit)
        return self

    def set_reference_amplitude(self, A_ref: float) -> "AgencityPipeline":
        self.metadata["reference_amplitude"] = float(A_ref)
        return self

    def set_characteristic_time(self, tau: float) -> "AgencityPipeline":
        self.metadata["characteristic_time"] = float(tau)
        return self

    def set_memory_window(self, w: float) -> "AgencityPipeline":
        self.metadata["memory_window"] = float(w)
        return self

    def set_characteristic_power(self, P_c: float) -> "AgencityPipeline":
        self.metadata["characteristic_power"] = float(P_c)
        return self

    def set_system_type(self, system_type: str) -> "AgencityPipeline":
        self.metadata["system_type"] = str(system_type)
        return self

    def set_mechanism(self, mechanism: str) -> "AgencityPipeline":
        self.metadata["mechanism"] = str(mechanism)
        return self

    def compute(self, *, verbose: bool = False, **kwargs: Any) -> "AgencityPipeline":
        if self.u is None:
            raise ValueError("No signal loaded")

        compute_kwargs = dict(kwargs)
        if "w" not in compute_kwargs and self.metadata.get("memory_window") is not None:
            compute_kwargs["w"] = self.metadata["memory_window"]

        self.result = compute_agencity(
            u=self.u,
            xi=self.xi,
            metadata=self.metadata,
            verbose=verbose,
            **compute_kwargs,
        )
        return self

    def analyze(self, *, verbose: bool = False) -> "AgencityPipeline":
        if self.result is None:
            self.compute(verbose=verbose)
        assert self.result is not None
        self.signature = analyze_signature(self.result, verbose=verbose)
        self.multiscale = analyze_multiscale(self.result, verbose=verbose)
        self.analysis = analyze_agencity(
            self.result,
            signature=self.signature,
            multiscale=self.multiscale,
            verbose=verbose,
        )
        return self

    def report_dict(self) -> dict[str, Any] | None:
        if self.analysis is None:
            self.analyze()
        return self.analysis

    def report(self, *, refresh: bool = False) -> str:
        if self.result is None:
            self.compute()
        assert self.result is not None
        if self.report_text is None or refresh:
            self.report_text = textual_analysis(self.result)
        return self.report_text

    def run(self, *, verbose: bool = False) -> AgencityResult:
        self.compute(verbose=verbose).analyze(verbose=verbose)
        assert self.result is not None
        return self.result

    @property
    def summary(self) -> dict[str, Any]:
        return self.result.summary() if self.result is not None else {}

    @property
    def b(self):
        return self.result.b if self.result is not None else None

    @property
    def beta(self):
        return self.result.beta if self.result is not None else None

    def inspect(self) -> dict[str, Any]:
        return {
            "has_data": self.u is not None,
            "has_result": self.result is not None,
            "metadata": dict(self.metadata),
        }


def pipeline() -> AgencityPipeline:
    """Return a new fluent pipeline instance."""
    return AgencityPipeline()
