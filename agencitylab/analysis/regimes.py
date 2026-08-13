"""Theory-facing regime signatures and explicit diagnostic classification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np

from .coherence import sigma_theta
from .geometry import curvature
from .transitions import detect_agencity_zeros


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
    if beta.size < 4:
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
        return 1.0 if mismatch == 0.0 else 0.0
    return float(1.0 / (1.0 + mismatch / scale))


def regime_signature(result) -> dict[str, float | int | bool | str]:
    """Extract a threshold-free signature from a canonical result.

    The signature contains observations only. It is intentionally separate
    from automatic classification so that scientific data are not silently
    forced through arbitrary universal thresholds.
    """
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

    valid = S > 0.0
    sigma = sigma_theta(theta, xi, tau, valid_mask=valid)
    finite_sigma = np.isfinite(sigma)
    kappa = curvature(beta, xi)
    finite_kappa = np.isfinite(kappa)
    mag = np.abs(b)
    tail_mag = _tail(mag)
    tail_J = _tail(J)
    tail_mean = float(np.mean(tail_mag)) if tail_mag.size else float("nan")
    tail_std = float(np.std(tail_mag)) if tail_mag.size else float("nan")
    tail_cv = tail_std / tail_mean if tail_mean > 0.0 else (0.0 if tail_std == 0.0 else float("inf"))

    quarter = max(1, n // 4)
    early = float(np.mean(mag[:quarter]))
    late = float(np.mean(mag[-quarter:]))
    if early == 0.0:
        growth_ratio = float("inf") if late > 0.0 else 1.0
    else:
        growth_ratio = late / early

    warm = xi >= xi[0] + 2.0 * tau
    zeros = detect_agencity_zeros(S, J)
    warm_count = int(np.count_nonzero(warm))
    warm_zero_count = int(np.count_nonzero(warm[zeros])) if zeros.size else 0

    return {
        "n_samples": int(n),
        "exact_null": bool(np.all(b == 0.0) and np.all(S == 0.0)),
        "mean_b_real": float(np.mean(np.real(b))),
        "mean_b_imag": float(np.mean(np.imag(b))),
        "mean_abs_b": float(np.mean(mag)),
        "var_abs_b": float(np.var(mag)),
        "mean_D": float(np.mean(D)),
        "mean_S": float(np.mean(S)),
        "mean_J": float(np.mean(J)),
        "tail_mean_J": float(np.mean(tail_J)) if tail_J.size else float("nan"),
        "tail_cv_abs_b": float(tail_cv),
        "growth_ratio_abs_b": float(growth_ratio),
        "sigma_theta_mean": float(np.mean(sigma[finite_sigma])) if np.any(finite_sigma) else float("nan"),
        "curvature_mean_abs": float(np.mean(np.abs(kappa[finite_kappa]))) if np.any(finite_kappa) else float("nan"),
        "curvature_std": float(np.std(kappa[finite_kappa])) if np.any(finite_kappa) else float("nan"),
        "tau_periodicity_score": _periodicity_score(beta, xi, tau),
        "zero_density_after_crm_warmup": float(warm_zero_count / warm_count) if warm_count else float("nan"),
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
