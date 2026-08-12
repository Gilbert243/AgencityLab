"""Diagnostic evaluators for the thermodynamic laws of Agencity.

The Modulus Law and Phase Law are evaluated against supplied data; they never
modify canonical ``b``.  Scientific status: research.  The Phase-Law reference
fit is an empirical reference reported in Volume 2, not a universal constant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping
import warnings

import numpy as np

from agencitylab.models.field_extensions import ParameterProvenance
from agencitylab.scientific_status import ScientificStatus

SCIENTIFIC_STATUS = ScientificStatus.RESEARCH


def _finite_real_array(value, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.bool_
    ):
        raise TypeError(f"{name} must contain real numeric values")
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be real")
    array = np.asarray(array, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _finite_real_scalar(value, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        scalar = float(value)
    except Exception as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _scalarize(value: np.ndarray):
    return float(value) if value.ndim == 0 else value


@dataclass(frozen=True, slots=True)
class PhaseLawFit:
    """Explicit coefficients for evaluating Volume 2 Eq. (18.8)."""

    alpha: float
    beta_fit: float
    r_squared: float | None = None
    reference_kind: str = "user_supplied_fit"
    scientific_status: ScientificStatus | str = ScientificStatus.RESEARCH
    provenance: Mapping[str, ParameterProvenance] = field(default_factory=dict)

    def __post_init__(self) -> None:
        alpha = _finite_real_scalar(self.alpha, name="alpha")
        beta_fit = _finite_real_scalar(self.beta_fit, name="beta_fit")
        r_squared = self.r_squared
        if r_squared is not None:
            r_squared = _finite_real_scalar(r_squared, name="r_squared")
            if not 0.0 <= r_squared <= 1.0:
                raise ValueError("r_squared must lie in [0, 1]")
        status = (
            self.scientific_status
            if isinstance(self.scientific_status, ScientificStatus)
            else ScientificStatus(self.scientific_status)
        )
        if status is not ScientificStatus.RESEARCH:
            raise ValueError("Phase-Law fits in this layer must have status 'research'")
        provenance: dict[str, ParameterProvenance] = {}
        for key, item in dict(self.provenance).items():
            provenance[str(key)] = (
                item
                if isinstance(item, ParameterProvenance)
                else ParameterProvenance.from_dict(item)
            )
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "beta_fit", beta_fit)
        object.__setattr__(self, "r_squared", r_squared)
        object.__setattr__(self, "reference_kind", str(self.reference_kind).strip())
        object.__setattr__(self, "scientific_status", status)
        object.__setattr__(self, "provenance", provenance)

    def to_dict(self) -> dict[str, Any]:
        """Return a lightweight serializable metadata representation."""

        return {
            "alpha": self.alpha,
            "beta_fit": self.beta_fit,
            "r_squared": self.r_squared,
            "reference_kind": self.reference_kind,
            "scientific_status": self.scientific_status.value,
            "provenance": {
                key: item.to_dict() for key, item in self.provenance.items()
            },
        }


def thermal_reference_phase_fit() -> PhaseLawFit:
    """Return the named empirical thermal-system fit reported in Volume 2.

    The reported values are ``alpha ~= 0.82``, ``beta ~= -1.50`` and
    ``R^2 ~= 0.87`` for the thermal systems studied in the manuscript.  They are
    not universal constants and are never used as silent defaults.
    """

    reference = "Volume 2, Chapter 18, Eq. (18.8)"
    provenance = {
        name: ParameterProvenance(
            source="source_document_reference",
            note="empirical thermal-system reference fit; not universal",
            reference=reference,
        )
        for name in ("alpha", "beta_fit", "r_squared")
    }
    return PhaseLawFit(
        alpha=0.82,
        beta_fit=-1.50,
        r_squared=0.87,
        reference_kind="empirical_reference",
        provenance=provenance,
    )


def second_law_residual(dS_ag_dt, dS_therm_dt, total_sigma):
    """Evaluate ``dS_ag/dt + dS_therm/dt - integral(sigma) dV``.

    A zero residual matches Volume 2 Eq. (18.6).  The value is diagnostic and
    is never clipped: negative values from inconsistent data remain visible.
    """

    agencial_rate = _finite_real_array(dS_ag_dt, name="dS_ag_dt")
    thermal_rate = _finite_real_array(dS_therm_dt, name="dS_therm_dt")
    production = _finite_real_array(total_sigma, name="total_sigma")
    try:
        agencial_b, thermal_b, production_b = np.broadcast_arrays(
            agencial_rate,
            thermal_rate,
            production,
        )
    except ValueError as exc:
        raise ValueError("second-law terms are not broadcast-compatible") from exc
    result = np.asarray(agencial_b + thermal_b - production_b, dtype=float)
    return _scalarize(result)


def modulus_law_margin(b, p_diss, t_amb, sdot_int):
    """Return the Volume 2 Eq. (18.7) margin without altering ``b``.

    The evaluated margin is
    ``abs(b) - (P_diss + T_amb * Sdot_int)``.  No term is clipped, so both
    positive and negative internal-entropy rates remain scientifically visible.
    """

    agencity = np.asarray(b)
    if not np.issubdtype(agencity.dtype, np.number) or np.issubdtype(
        agencity.dtype, np.bool_
    ):
        raise TypeError("b must contain real or complex numeric values")
    if not np.all(np.isfinite(agencity)):
        raise ValueError("b must contain only finite values")
    dissipated = _finite_real_array(p_diss, name="p_diss")
    ambient = _finite_real_array(t_amb, name="t_amb")
    entropy_rate = _finite_real_array(sdot_int, name="sdot_int")
    try:
        modulus_b, dissipated_b, ambient_b, entropy_b = np.broadcast_arrays(
            np.abs(agencity),
            dissipated,
            ambient,
            entropy_rate,
        )
    except ValueError as exc:
        raise ValueError("Modulus-Law inputs are not broadcast-compatible") from exc
    result = np.asarray(
        modulus_b - (dissipated_b + ambient_b * entropy_b),
        dtype=float,
    )
    return _scalarize(result)


def modulus_law_satisfied(b, p_diss, t_amb, sdot_int):
    """Return whether the evaluated Modulus-Law margin is non-negative."""

    result = np.asarray(modulus_law_margin(b, p_diss, t_amb, sdot_int) >= 0.0)
    return bool(result) if result.ndim == 0 else result


def _resolve_phase_coefficients(
    *,
    alpha: float | None,
    beta_fit: float | None,
    fit: PhaseLawFit | None,
) -> tuple[float, float]:
    if fit is not None:
        if alpha is not None or beta_fit is not None:
            raise ValueError("provide either fit or explicit alpha/beta_fit, not both")
        if not isinstance(fit, PhaseLawFit):
            raise TypeError("fit must be a PhaseLawFit")
        return fit.alpha, fit.beta_fit
    if alpha is None or beta_fit is None:
        raise ValueError("alpha and beta_fit must be supplied explicitly when fit is absent")
    return (
        _finite_real_scalar(alpha, name="alpha"),
        _finite_real_scalar(beta_fit, name="beta_fit"),
    )


def phase_law_prediction(
    p_diss,
    t_amb,
    sdot_int,
    *,
    alpha: float | None = None,
    beta_fit: float | None = None,
    fit: PhaseLawFit | None = None,
):
    """Evaluate the Phase-Law fitted prediction from Volume 2 Eq. (18.8).

    The source manuscript denotes the imaginary field component by ``O`` in
    this section.  That symbol is *not* the canonical CRM organisation ``O``.
    This API therefore accepts no argument named ``O``.  It evaluates only
    ``alpha * log10(P_diss / (T_amb * |Sdot_int|)) + beta_fit``.

    The ratio must be finite and strictly positive.  In particular,
    ``Sdot_int == 0`` is mathematically undefined and is rejected without EPS.
    """

    alpha_value, beta_value = _resolve_phase_coefficients(
        alpha=alpha,
        beta_fit=beta_fit,
        fit=fit,
    )
    dissipated = _finite_real_array(p_diss, name="p_diss")
    ambient = _finite_real_array(t_amb, name="t_amb")
    entropy_rate = _finite_real_array(sdot_int, name="sdot_int")
    try:
        dissipated_b, ambient_b, entropy_b = np.broadcast_arrays(
            dissipated,
            ambient,
            entropy_rate,
        )
    except ValueError as exc:
        raise ValueError("Phase-Law inputs are not broadcast-compatible") from exc
    if np.any(entropy_b == 0.0):
        raise ValueError("Phase Law is undefined when sdot_int == 0")
    denominator = ambient_b * np.abs(entropy_b)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = dissipated_b / denominator
    if not np.all(np.isfinite(ratio)) or np.any(ratio <= 0.0):
        raise ValueError("Phase-Law ratio must be finite and strictly positive")
    result = np.asarray(alpha_value * np.log10(ratio) + beta_value, dtype=float)
    return _scalarize(result)


def phase_law_residual(
    phase_component,
    p_diss,
    t_amb,
    sdot_int,
    *,
    alpha: float | None = None,
    beta_fit: float | None = None,
    fit: PhaseLawFit | None = None,
):
    """Return supplied imaginary-field component minus Phase-Law prediction."""

    component = _finite_real_array(phase_component, name="phase_component")
    prediction = np.asarray(
        phase_law_prediction(
            p_diss,
            t_amb,
            sdot_int,
            alpha=alpha,
            beta_fit=beta_fit,
            fit=fit,
        )
    )
    try:
        component_b, prediction_b = np.broadcast_arrays(component, prediction)
    except ValueError as exc:
        raise ValueError("phase_component is not compatible with the prediction") from exc
    result = np.asarray(component_b - prediction_b, dtype=float)
    return _scalarize(result)


def phi_imaginary_component(phi) -> np.ndarray:
    """Return ``Im(phi) = |phi| sin(Theta)`` with unambiguous naming.

    This helper exists specifically to avoid confusing the manuscript's
    Chapter-18 symbol ``O`` with canonical CRM organisation ``O``.
    """

    field = np.asarray(phi)
    if not np.issubdtype(field.dtype, np.number) or np.issubdtype(
        field.dtype, np.bool_
    ):
        raise TypeError("phi must contain real or complex numeric values")
    if not np.all(np.isfinite(field)):
        raise ValueError("phi must contain only finite values")
    return np.asarray(np.imag(field), dtype=float)


def second_law_check(entropy_series):
    """Legacy monotonic-series placeholder, not Volume 2 Eq. (18.6)."""

    warnings.warn(
        "second_law_check is a legacy monotonicity heuristic; use "
        "second_law_residual for the Chapter-18 relation",
        DeprecationWarning,
        stacklevel=2,
    )
    return all(x2 >= x1 for x1, x2 in zip(entropy_series, entropy_series[1:]))
