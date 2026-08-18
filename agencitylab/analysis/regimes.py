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
    variance, convergence, periodicity, weak structure, and irregular
    structure, but it does not prescribe universal finite-record thresholds.
    Every numerical field here is therefore a diagnostic choice supplied by
    the caller or experiment.

    ``weak_flow_max`` is retained only for source compatibility with v0.5. It
    no longer separates stochastic and chaotic regimes because absolute flow
    magnitude is not the theoretical distinction between them.
    """

    sigma_theta_low_max: float
    sigma_theta_high_min: float
    tail_cv_max: float
    unstable_growth_ratio_min: float
    curvature_zero_max: float
    periodicity_min: float
    weak_flow_max: float | None = None
    weak_structure_max: float | None = None
    weak_beta_variance_max: float | None = None
    structure_variability_min: float | None = None

    def __post_init__(self) -> None:
        values = asdict(self)
        for name, value in values.items():
            if value is None:
                continue
            numeric = float(value)
            if not np.isfinite(numeric) or numeric < 0.0:
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


def _coefficient_of_variation(values: np.ndarray) -> float:
    if not values.size:
        return float("nan")
    mean = float(np.mean(values))
    std = float(np.std(values))
    if mean > 0.0:
        return std / mean
    return 0.0 if std == 0.0 else float("nan")


def _growth_ratio(values: np.ndarray) -> float:
    if not values.size:
        return float("nan")
    quarter = max(1, values.size // 4)
    early = float(np.mean(values[:quarter]))
    late = float(np.mean(values[-quarter:]))
    if early == 0.0:
        return float("inf") if late > 0.0 else 1.0
    return late / early


def _complex_variance(values: np.ndarray) -> float:
    if not values.size:
        return float("nan")
    centre = np.mean(values)
    return float(np.mean(np.abs(values - centre) ** 2))


def _relative_fixed_point_rms(values: np.ndarray) -> float:
    """Return tail scatter relative to its complex fixed-point candidate."""
    if not values.size:
        return float("nan")
    centre = complex(np.mean(values))
    scatter = float(np.sqrt(np.mean(np.abs(values - centre) ** 2)))
    scale = abs(centre)
    if scale > 0.0:
        return scatter / scale
    return 0.0 if scatter == 0.0 else float("inf")


def _periodicity_diagnostic(
    beta: np.ndarray,
    xi: np.ndarray,
) -> tuple[float, float | None]:
    """Estimate finite-record beta periodicity without identifying tau with T.

    This is a numerical diagnostic, not a canonical equation. The strongest
    non-zero Fourier component proposes a candidate period, then two complete
    terminal cycles are compared directly. A constant/fixed-point record and
    records with fewer than two candidate cycles are left undefined.
    """
    if beta.ndim != 1 or xi.ndim != 1 or beta.size != xi.size or beta.size < 8:
        return float("nan"), None
    if not np.all(np.isfinite(xi)) or np.any(np.diff(xi) <= 0.0):
        return float("nan"), None
    finite_beta = np.isfinite(np.real(beta)) & np.isfinite(np.imag(beta))
    if not np.all(finite_beta):
        return float("nan"), None

    diffs = np.diff(xi)
    step = float(np.median(diffs))
    spacing_error = float(np.max(np.abs(diffs - step)))
    numerical_spacing_tol = (
        64.0
        * np.finfo(float).eps
        * max(1.0, abs(step), float(np.max(np.abs(xi))))
    )
    if spacing_error > numerical_spacing_tol:
        return float("nan"), None

    centred = beta - np.mean(beta)
    scale = float(np.sqrt(np.mean(np.abs(centred) ** 2)))
    if scale == 0.0:
        return float("nan"), None

    n = beta.size
    spectrum = np.fft.fft(centred)
    max_k = n // 2
    if max_k < 1:
        return float("nan"), None
    candidate_k = np.arange(1, max_k + 1)
    paired_power = np.abs(spectrum[candidate_k]) ** 2
    negative_k = (-candidate_k) % n
    paired_power += np.abs(spectrum[negative_k]) ** 2
    if not np.any(paired_power > 0.0):
        return float("nan"), None

    dominant_k = int(candidate_k[int(np.argmax(paired_power))])
    candidate_period = n * step / dominant_k
    shift = int(round(candidate_period / step))
    if shift < 2 or 2 * shift > n:
        return float("nan"), None

    left = beta[-2 * shift : -shift]
    right = beta[-shift:]
    mismatch = float(np.sqrt(np.mean(np.abs(left - right) ** 2)))
    score = float(1.0 / (1.0 + mismatch / scale))
    return score, float(shift * step)


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

    exact_null = bool(
        np.all(D == 0.0)
        and np.all(S == 0.0)
        and np.all(J == 0.0)
        and np.all(beta == 0.0)
        and np.all(b == 0.0)
    )
    interval = resolve_analysis_interval(result, edge_samples=2)
    analysis_mask = interval.mask
    valid_count = int(np.count_nonzero(analysis_mask))

    structural = S > 0.0
    sigma = sigma_theta(theta, xi, tau, valid_mask=structural)
    finite_sigma = np.isfinite(sigma) & analysis_mask
    kappa = curvature(beta, xi)
    finite_kappa = np.isfinite(kappa) & analysis_mask

    mag_b = np.abs(b)
    mag_beta = np.abs(beta)
    mag_b_valid = mag_b[analysis_mask]
    mag_beta_valid = mag_beta[analysis_mask]
    D_valid = D[analysis_mask]
    S_valid = S[analysis_mask]
    J_valid = J[analysis_mask]
    beta_valid = beta[analysis_mask]
    xi_valid = xi[analysis_mask]

    tail_mag_b = _tail(mag_b_valid)
    tail_mag_beta = _tail(mag_beta_valid)
    tail_beta = _tail(beta_valid)
    tail_J = _tail(J_valid)
    finite_kappa_values = kappa[finite_kappa]
    tail_kappa = _tail(finite_kappa_values)

    zeros = detect_agencity_zeros(S, J)
    analysis_zero_count = int(np.count_nonzero(analysis_mask[zeros])) if zeros.size else 0
    periodicity, estimated_period = (
        _periodicity_diagnostic(beta_valid, xi_valid)
        if valid_count
        else (float("nan"), None)
    )

    def mean_or_nan(values):
        return float(np.mean(values)) if values.size else float("nan")

    tail_beta_mean = complex(np.mean(tail_beta)) if tail_beta.size else complex(np.nan, np.nan)

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
        "mean_abs_b": mean_or_nan(mag_b_valid),
        "var_abs_b": float(np.var(mag_b_valid)) if mag_b_valid.size else float("nan"),
        "tail_cv_abs_b": _coefficient_of_variation(tail_mag_b),
        "growth_ratio_abs_b": _growth_ratio(mag_b_valid),
        "mean_abs_beta": mean_or_nan(mag_beta_valid),
        "variance_beta": _complex_variance(beta_valid),
        "tail_mean_abs_beta": mean_or_nan(tail_mag_beta),
        "tail_beta_mean_real": float(np.real(tail_beta_mean)),
        "tail_beta_mean_imag": float(np.imag(tail_beta_mean)),
        "tail_beta_mean_abs": float(abs(tail_beta_mean)),
        "tail_beta_relative_rms": _relative_fixed_point_rms(tail_beta),
        "tail_cv_abs_beta": _coefficient_of_variation(tail_mag_beta),
        "growth_ratio_abs_beta": _growth_ratio(mag_beta_valid),
        "mean_D": mean_or_nan(D_valid),
        "mean_S": mean_or_nan(S_valid),
        "std_S": float(np.std(S_valid)) if S_valid.size else float("nan"),
        "mean_J": mean_or_nan(J_valid),
        "tail_mean_J": mean_or_nan(tail_J),
        "sigma_theta_mean": mean_or_nan(sigma[finite_sigma]),
        "curvature_mean_abs": mean_or_nan(np.abs(kappa[finite_kappa])),
        "tail_curvature_mean_abs": mean_or_nan(np.abs(tail_kappa)),
        "curvature_std": (
            float(np.std(kappa[finite_kappa]))
            if np.any(finite_kappa)
            else float("nan")
        ),
        "periodicity_score": periodicity,
        "periodicity_defined": bool(np.isfinite(periodicity)),
        "estimated_period": estimated_period,
        # Compatibility aliases. Their names are historical; the score is no
        # longer evaluated at lag tau and tau is not identified with a period.
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


def _metric(signature: Mapping, name: str) -> float:
    value = signature.get(name, float("nan"))
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def classify_regime(
    result_or_signature,
    *,
    criteria=None,
    theta=None,
    alpha=None,
    epsilon=None,
    verbose: bool = False,
) -> str:
    """Classify a regime using explicit diagnostic criteria.

    With no criteria, only the exact null state is classified; every non-null
    finite record is returned as ``"undetermined"``. A raw ``b`` array never
    proves the null regime because ``b = 0`` alone does not establish
    ``D = S = beta = 0``.

    Non-null classification uses intrinsic evidence from ``beta``, ``J``,
    ``Theta`` and ``S``. Absolute ``b`` magnitude is retained in the signature
    as a flux observation but does not define the intrinsic regime.

    ``theta``, ``alpha`` and ``epsilon`` are accepted for source compatibility
    with the pre-v0.5 API but are not used to redefine the classification.
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
        return "undetermined"

    if bool(sig.get("exact_null", False)):
        return "null"

    cfg = _criteria_from(criteria)
    if cfg is None:
        return "undetermined"

    sigma = _metric(sig, "sigma_theta_mean")
    tail_kappa = _metric(sig, "tail_curvature_mean_abs")
    growth = _metric(sig, "growth_ratio_abs_beta")
    fixed_point_rms = _metric(sig, "tail_beta_relative_rms")
    periodicity = _metric(sig, "periodicity_score")
    beta_variance = _metric(sig, "variance_beta")
    mean_abs_beta = _metric(sig, "mean_abs_beta")
    tail_beta_abs = _metric(sig, "tail_beta_mean_abs")
    mean_S = _metric(sig, "mean_S")
    std_S = _metric(sig, "std_S")
    tail_J = _metric(sig, "tail_mean_J")

    low_sigma = np.isfinite(sigma) and sigma <= cfg.sigma_theta_low_max
    high_sigma = np.isfinite(sigma) and sigma >= cfg.sigma_theta_high_min
    flat_geometry = np.isfinite(tail_kappa) and tail_kappa <= cfg.curvature_zero_max
    fixed_point = np.isfinite(fixed_point_rms) and fixed_point_rms <= cfg.tail_cv_max
    periodic = np.isfinite(periodicity) and periodicity >= cfg.periodicity_min
    aperiodic = np.isfinite(periodicity) and periodicity < cfg.periodicity_min

    if (
        low_sigma
        and np.isfinite(growth)
        and growth >= cfg.unstable_growth_ratio_min
        and tail_J > 0.0
        and flat_geometry
    ):
        label = "unstable"
    elif low_sigma and fixed_point and tail_J < 0.0 and tail_beta_abs > 0.0:
        label = "passive_damped"
    elif (
        low_sigma
        and periodic
        and not fixed_point
        and mean_abs_beta > 0.0
        and beta_variance > 0.0
        and (not np.isfinite(growth) or growth < cfg.unstable_growth_ratio_min)
    ):
        # Curvature remains reported evidence, but is not a hard gate here:
        # the theory's Van der Pol discussion can degenerate to an almost-real
        # segment even while beta is demonstrably periodic and closed.
        label = "active_oscillating"
    elif (
        high_sigma
        and cfg.weak_structure_max is not None
        and cfg.weak_beta_variance_max is not None
        and np.isfinite(mean_S)
        and mean_S <= cfg.weak_structure_max
        and np.isfinite(beta_variance)
        and beta_variance <= cfg.weak_beta_variance_max
    ):
        label = "stochastic"
    elif (
        high_sigma
        and cfg.weak_structure_max is not None
        and cfg.weak_beta_variance_max is not None
        and cfg.structure_variability_min is not None
        and np.isfinite(mean_S)
        and mean_S > cfg.weak_structure_max
        and np.isfinite(std_S)
        and std_S >= cfg.structure_variability_min
        and np.isfinite(beta_variance)
        and beta_variance > cfg.weak_beta_variance_max
        and aperiodic
    ):
        label = "chaotic"
    else:
        label = "undetermined"

    if verbose:
        print(f"[regime] label={label}")
        print(f"[regime] criteria={asdict(cfg)}")
    return label


def detect_regime_changes(
    b,
    *,
    window: int = 32,
    epsilon: float = 1e-12,
    component: str = "magnitude",
):
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
