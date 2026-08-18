"""Theory-facing regime signatures and explicit diagnostic classification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np

from .coherence import sigma_theta
from .geometry import curvature
from .transitions import detect_agencity_zeros
from .validity import resolve_analysis_interval


@dataclass(frozen=True, slots=True)
class RegimeCriteria:
    """Contextual thresholds for automatic regime classification.

    The theory supplies qualitative descriptors such as low/high angular
    variance and zero/non-zero curvature, but it does not prescribe universal
    finite-record thresholds. Every numerical field here is therefore a
    diagnostic choice that must be supplied by the caller or experiment.
    """

    sigma_theta_low_max: float
    sigma_theta_high_min: float
    tail_cv_max: float
    unstable_growth_ratio_min: float
    curvature_zero_max: float
    periodicity_min: float
    weak_flow_max: float

    def __post_init__(self) -> None:
        values = asdict(self)
        for name, value in values.items():
            value = float(value)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.sigma_theta_high_min < self.sigma_theta_low_max:
            raise ValueError("sigma_theta_high_min must be >= sigma_theta_low_max")
        if self.periodicity_min > 1.0:
            raise ValueError("periodicity_min must lie in [0, 1]")


def _get(result, name):
    if isinstance(result, Mapping):
        return result[name]
    return getattr(result, name)


def _tail(values: np.ndarray, fraction: float = 0.25) -> np.ndarray:
    start = int(np.floor((1.0 - fraction) * values.size))
    return values[max(0, start) :]


def _periodicity_score(beta: np.ndarray, xi: np.ndarray, tau: float) -> float:
    if beta.size < 4 or np.all(beta == 0.0):
        return float("nan")
    step = float(np.median(np.diff(xi)))
    shift = int(round(float(tau) / step))
    if shift < 1 or 2 * shift >= beta.size:
        return float("nan")
    left = beta[-2 * shift : -shift]
    right = beta[-shift:]
    scale = float(np.sqrt(np.mean(np.abs(left) ** 2)))
    mismatch = float(np.sqrt(np.mean(np.abs(left - right) ** 2)))
    if scale == 0.0:
        return float("nan")
    return float(1.0 / (1.0 + mismatch / scale))


def regime_signature(
    result,
) -> dict[str, float | int | bool | str | None]:
    """Extract a threshold-free signature on the shared valid analysis interval."""

    xi = np.asarray(_get(result, "xi"), dtype=float)
    tau = float(_get(result, "tau"))
    S = np.asarray(_get(result, "S"), dtype=float)
    D = np.asarray(_get(result, "D"), dtype=float)
    J = np.asarray(_get(result, "J"), dtype=float)
    theta = np.asarray(_get(result, "theta"), dtype=float)
    beta = np.asarray(_get(result, "beta"), dtype=complex)
    b = np.asarray(_get(result, "b"), dtype=complex)
    n = xi.size
    if any(arr.ndim != 1 or arr.size != n for arr in (S, D, J, theta, beta, b)):
        raise ValueError("result arrays must be one-dimensional and share xi length")

    exact_null = bool(np.all(b == 0.0) and np.all(S == 0.0))
    interval = resolve_analysis_interval(result, edge_samples=2)
    analysis_mask = interval.mask
    valid_count = int(np.count_nonzero(analysis_mask))

    structural = S > 0.0
    sigma = sigma_theta(theta, xi, tau, valid_mask=structural)
    finite_sigma = np.isfinite(sigma) & analysis_mask
    kappa = curvature(beta, xi)
    finite_kappa = np.isfinite(kappa) & analysis_mask

    mag = np.abs(b)
    mag_valid = mag[analysis_mask]
    D_valid = D[analysis_mask]
    S_valid = S[analysis_mask]
    J_valid = J[analysis_mask]
    beta_valid = beta[analysis_mask]
    xi_valid = xi[analysis_mask]

    tail_mag = _tail(mag_valid)
    tail_J = _tail(J_valid)
    tail_mean = float(np.mean(tail_mag)) if tail_mag.size else float("nan")
    tail_std = float(np.std(tail_mag)) if tail_mag.size else float("nan")
    tail_cv = (
        tail_std / tail_mean
        if tail_mean > 0.0
        else (0.0 if tail_std == 0.0 and tail_mag.size else float("nan"))
    )

    if mag_valid.size:
        quarter = max(1, mag_valid.size // 4)
        early = float(np.mean(mag_valid[:quarter]))
        late = float(np.mean(mag_valid[-quarter:]))
        if early == 0.0:
            growth_ratio = float("inf") if late > 0.0 else 1.0
        else:
            growth_ratio = late / early
    else:
        growth_ratio = float("nan")

    zeros = detect_agencity_zeros(S, J)
    analysis_zero_count = int(np.count_nonzero(analysis_mask[zeros])) if zeros.size else 0
    periodicity = _periodicity_score(beta_valid, xi_valid, tau) if valid_count else float("nan")

    def mean_or_nan(values):
        return float(np.mean(values)) if values.size else float("nan")

    return {
        "n_samples": int(n),
        "analysis_valid_samples": valid_count,
        "analysis_valid_fraction": interval.valid_fraction,
        "analysis_start": interval.start_time,
        "analysis_stop": interval.stop_time,
        "analysis_window": interval.memory_window,
        "analysis_window_source": interval.memory_window_source,
        "exact_null": exact_null,
        "mean_b_real": mean_or_nan(np.real(b[analysis_mask])),
        "mean_b_imag": mean_or_nan(np.imag(b[analysis_mask])),
        "mean_abs_b": mean_or_nan(mag_valid),
        "var_abs_b": float(np.var(mag_valid)) if mag_valid.size else float("nan"),
        "mean_D": mean_or_nan(D_valid),
        "mean_S": mean_or_nan(S_valid),
        "mean_J": mean_or_nan(J_valid),
        "tail_mean_J": mean_or_nan(tail_J),
        "tail_cv_abs_b": float(tail_cv),
        "growth_ratio_abs_b": float(growth_ratio),
        "sigma_theta_mean": mean_or_nan(sigma[finite_sigma]),
        "curvature_mean_abs": mean_or_nan(np.abs(kappa[finite_kappa])),
        "curvature_std": (
            float(np.std(kappa[finite_kappa]))
            if np.any(finite_kappa)
            else float("nan")
        ),
        "tau_periodicity_score": periodicity,
        "tau_periodicity_defined": bool(np.isfinite(periodicity)),
        "zero_density_after_crm_warmup": (
            float(analysis_zero_count / valid_count) if valid_count else float("nan")
        ),
        "status": "threshold-free diagnostic signature",
    }


def _criteria_from(value) -> RegimeCriteria | None:
    if value is None:
        return None
    if isinstance(value, RegimeCriteria):
        return value
    if isinstance(value, Mapping):
        return RegimeCriteria(**value)
    raise TypeError("criteria must be RegimeCriteria, a mapping, or None")


def classify_regime(result_or_signature, *, criteria=None, theta=None, alpha=None, epsilon=None, verbose: bool = False) -> str:
    """Classify a regime using explicit diagnostic criteria.

    With no criteria, only the exact null state is classified; every non-null
    finite record is returned as ``"undetermined"``. This conservative default
    prevents old hard-coded thresholds from masquerading as theory.

    ``theta``, ``alpha`` and ``epsilon`` are accepted for source compatibility
    with the pre-v0.5 API but are not used to redefine the v0.5 classification.
    """
    del theta, alpha, epsilon
    if isinstance(result_or_signature, Mapping) and "mean_abs_b" in result_or_signature:
        sig = dict(result_or_signature)
    elif hasattr(result_or_signature, "beta") and hasattr(result_or_signature, "S"):
        sig = regime_signature(result_or_signature)
    else:
        b = np.asarray(result_or_signature, dtype=complex)
        if b.ndim != 1 or b.size == 0:
            return "unknown"
        if np.all(b == 0.0):
            return "null"
        return "undetermined"

    if bool(sig.get("exact_null", False)):
        return "null"

    cfg = _criteria_from(criteria)
    if cfg is None:
        return "undetermined"

    sigma = float(sig["sigma_theta_mean"])
    kappa = float(sig["curvature_mean_abs"])
    growth = float(sig["growth_ratio_abs_b"])
    tail_cv = float(sig["tail_cv_abs_b"])
    periodicity = float(sig["tau_periodicity_score"])
    mean_abs_b = float(sig["mean_abs_b"])
    tail_J = float(sig["tail_mean_J"])

    low_sigma = np.isfinite(sigma) and sigma <= cfg.sigma_theta_low_max
    high_sigma = np.isfinite(sigma) and sigma >= cfg.sigma_theta_high_min
    flat_geometry = np.isfinite(kappa) and kappa <= cfg.curvature_zero_max
    periodic = np.isfinite(periodicity) and periodicity >= cfg.periodicity_min

    if low_sigma and growth >= cfg.unstable_growth_ratio_min and flat_geometry:
        label = "unstable"
    elif low_sigma and tail_cv <= cfg.tail_cv_max and tail_J < 0.0 and flat_geometry:
        label = "passive_damped"
    elif low_sigma and periodic and np.isfinite(kappa) and kappa > cfg.curvature_zero_max:
        label = "active_oscillating"
    elif high_sigma and mean_abs_b <= cfg.weak_flow_max:
        label = "stochastic"
    elif high_sigma and mean_abs_b > cfg.weak_flow_max:
        label = "chaotic"
    else:
        label = "undetermined"

    if verbose:
        print(f"[regime] label={label}")
        print(f"[regime] criteria={asdict(cfg)}")
    return label


def detect_regime_changes(b, *, window: int = 32, epsilon: float = 1e-12, component: str = "magnitude"):
    """Historical rolling-variance change detector retained as a heuristic.

    It is not used by the theory-facing v0.5 classifier.
    """
    b = np.asarray(b, dtype=complex)
    if b.ndim != 1 or b.size < 2 * window:
        return []
    if component == "magnitude":
        x = np.abs(b)
    elif component == "real":
        x = np.real(b)
    elif component == "imag":
        x = np.imag(b)
    elif component == "phase":
        x = np.unwrap(np.angle(b))
    else:
        raise ValueError("component must be one of: magnitude, real, imag, phase")
    changes = []
    prev = float(np.var(x[:window]))
    for i in range(window, x.size - window):
        current = float(np.var(x[i - window : i + window]))
        if abs(current - prev) > epsilon * max(1.0, abs(prev)):
            changes.append(i)
        prev = current
    return changes
