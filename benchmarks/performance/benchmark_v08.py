"""Reproducible before/after benchmark for AgencityLab v0.8.

The pre-v0.8 direct-window CRM and direct local ``Sigma_Theta`` loop are retained
locally as engineering references. Timing values are observations, never CI
pass/fail thresholds. Scientific agreement is reported independently through
maximum absolute differences and finite-domain equality.

Examples
--------
CI-sized run::

    python benchmarks/performance/benchmark_v08.py --quick

Fuller local run with JSON output::

    python benchmarks/performance/benchmark_v08.py --output benchmark-v08.json
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Callable

import numpy as np

from agencitylab import compute_agencity
from agencitylab.api import (
    AgencityStream,
    StreamNotReadyError,
    analyze_agencity,
    compute_agencity_spectrum,
    compute_multivariate_agencity,
    run_batch,
)
from agencitylab.analysis.coherence import sigma_theta
from agencitylab.core.activation import activation, reduced_coordinate
from agencitylab.core.activity import activity
from agencitylab.core.agencity import agencity
from agencitylab.core.beta import compute_beta
from agencitylab.core.crm import _window_to_samples, causal_moving_correlation
from agencitylab.core.intensity import compute_intensities
from agencitylab.core.normalization import normalize_signal


def _legacy_pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pre-v0.8 direct centred Pearson coefficient."""
    a0 = a - np.mean(a)
    b0 = b - np.mean(b)
    ss_a = float(np.dot(a0, a0))
    ss_b = float(np.dot(b0, b0))
    if ss_a == 0.0 or ss_b == 0.0:
        return 0.0
    return float(np.clip(np.dot(a0, b0) / np.sqrt(ss_a * ss_b), -1.0, 1.0))


def legacy_crm(signal, *, axis, window, other=None):
    """Pre-v0.8 O(N*w) CRM, used only as a benchmark reference."""
    x = np.asarray(signal, dtype=float).ravel()
    y = x if other is None else np.asarray(other, dtype=float).ravel()
    width = _window_to_samples(window, axis)
    if x.size < 2 * width:
        raise ValueError("signal too short for two CRM windows")

    out = np.zeros(x.size, dtype=float)
    for end in range(2 * width - 1, x.size):
        recent = x[end - width + 1 : end + 1]
        previous = y[end - 2 * width + 1 : end - width + 1]
        out[end] = _legacy_pearson(recent, previous)
    return out


def legacy_sigma_theta(theta, xi, tau, *, valid_mask=None):
    """Pre-v0.8 O(N*w_tau) local-unwrapped angular variance."""
    theta = np.asarray(theta, dtype=float)
    xi = np.asarray(xi, dtype=float)
    valid = (
        np.ones(theta.size, dtype=bool)
        if valid_mask is None
        else np.asarray(valid_mask, dtype=bool)
    )
    out = np.full(theta.size, np.nan, dtype=float)
    for index, time_value in enumerate(xi):
        if time_value - xi[0] < tau:
            continue
        left = int(np.searchsorted(xi, time_value - tau, side="left"))
        indices = np.arange(left, index + 1)
        if indices.size < 2 or not np.all(valid[indices]):
            continue
        out[index] = float(np.var(np.unwrap(theta[indices])))
    return out


def _legacy_pipeline(xi, u, *, A_ref, tau, w, P_c):
    """Canonical pipeline with only the historical CRM substituted."""
    u_star, _ = normalize_signal(u, A_ref=A_ref, method="canonical")
    t_star = reduced_coordinate(xi, tau)
    X_star = activation(u_star, axis=t_star)
    A_star = activity(X_star, axis=t_star)
    M = legacy_crm(u_star, axis=xi, window=w)
    O = legacy_crm(u_star, axis=xi, window=w, other=X_star)
    D, S = compute_intensities(X_star, A_star, M, O)
    J, U, beta = compute_beta(D, S, M, O)
    b = agencity(beta, P_c)
    return {
        "u_star": u_star,
        "X_star": X_star,
        "A_star": A_star,
        "M": M,
        "O": O,
        "D": D,
        "S": S,
        "J": J,
        "U": U,
        "theta": np.angle(U),
        "beta": beta,
        "b": b,
    }


def _signal(size: int):
    xi = np.arange(size, dtype=float)
    u = (
        np.sin(0.017 * xi)
        + 0.20 * np.cos(0.031 * xi)
        + 0.05 * np.sin(0.003 * xi * xi / max(size, 1))
    )
    return xi, u


def _median_seconds(function: Callable[[], Any], repeats: int):
    durations = []
    value = None
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        value = function()
        durations.append(time.perf_counter() - start)
    return float(statistics.median(durations)), value


def _peak_mebibytes(function: Callable[[], Any]) -> float:
    gc.collect()
    tracemalloc.start()
    try:
        function()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return float(peak / (1024.0 * 1024.0))


def _max_difference(reference: Any, optimized: Any, fields: tuple[str, ...]) -> float:
    differences = []
    for field in fields:
        left = reference[field]
        right = getattr(optimized, field) if hasattr(optimized, field) else optimized[field]
        differences.append(float(np.max(np.abs(np.asarray(left) - np.asarray(right)))))
    return max(differences, default=0.0)


def _nan_domain_difference(reference, optimized):
    reference = np.asarray(reference, dtype=float)
    optimized = np.asarray(optimized, dtype=float)
    finite_reference = np.isfinite(reference)
    finite_optimized = np.isfinite(optimized)
    domain_equal = bool(np.array_equal(finite_reference, finite_optimized))
    common = finite_reference & finite_optimized
    difference = (
        float(np.max(np.abs(reference[common] - optimized[common])))
        if np.any(common)
        else 0.0
    )
    return domain_equal, difference


def _profile_stages(xi, u, *, A_ref, tau, w):
    """Profile the optimized canonical stages without changing their execution."""
    stages = {}

    start = time.perf_counter()
    u_star, _ = normalize_signal(u, A_ref=A_ref, method="canonical")
    stages["normalization_ms"] = 1000.0 * (time.perf_counter() - start)

    start = time.perf_counter()
    t_star = reduced_coordinate(xi, tau)
    X_star = activation(u_star, axis=t_star)
    A_star = activity(X_star, axis=t_star)
    stages["derivatives_X_A_ms"] = 1000.0 * (time.perf_counter() - start)

    start = time.perf_counter()
    M = causal_moving_correlation(u_star, tau, axis=xi, window=w)
    stages["memory_M_ms"] = 1000.0 * (time.perf_counter() - start)

    start = time.perf_counter()
    O = causal_moving_correlation(u_star, tau, axis=xi, window=w, other=X_star)
    stages["organization_O_ms"] = 1000.0 * (time.perf_counter() - start)

    start = time.perf_counter()
    D, S = compute_intensities(X_star, A_star, M, O)
    J, _, beta = compute_beta(D, S, M, O)
    agencity(beta, 2.5)
    stages["D_S_J_beta_b_ms"] = 1000.0 * (time.perf_counter() - start)
    return stages


def run_benchmark(*, quick: bool = False, repeats: int | None = None):
    """Run the suite and return a JSON-serializable report."""
    sizes = [4096, 16384, 65536] if quick else [4096, 32768, 131072]
    repeats = repeats or (2 if quick else 3)
    A_ref, tau, w, P_c = 1.5, 64.0, 64.0, 2.5
    fields = (
        "u_star",
        "X_star",
        "A_star",
        "M",
        "O",
        "D",
        "S",
        "J",
        "theta",
        "beta",
        "b",
    )

    report = {
        "schema_version": "0.8",
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "mode": "quick" if quick else "full",
        },
        "parameters": {
            "sizes": sizes,
            "repeats": repeats,
            "A_ref": A_ref,
            "tau": tau,
            "w": w,
            "P_c": P_c,
        },
        "crm": [],
        "pipeline": [],
        "sigma_theta": [],
    }

    for size in sizes:
        xi, u = _signal(size)
        before_crm = lambda: legacy_crm(u, axis=xi, window=w)
        after_crm = lambda: causal_moving_correlation(u, tau, axis=xi, window=w)
        before_seconds, before_value = _median_seconds(before_crm, repeats)
        after_seconds, after_value = _median_seconds(after_crm, repeats)
        report["crm"].append(
            {
                "n": size,
                "legacy_seconds": before_seconds,
                "optimized_seconds": after_seconds,
                "speedup": before_seconds / after_seconds,
                "legacy_peak_mib": _peak_mebibytes(before_crm),
                "optimized_peak_mib": _peak_mebibytes(after_crm),
                "max_abs_difference": float(np.max(np.abs(before_value - after_value))),
            }
        )

        before_pipeline = lambda: _legacy_pipeline(
            xi, u, A_ref=A_ref, tau=tau, w=w, P_c=P_c
        )
        after_pipeline = lambda: compute_agencity(
            u=u,
            xi=xi,
            A_ref=A_ref,
            tau=tau,
            w=w,
            P_c=P_c,
        )
        before_seconds, before_value = _median_seconds(before_pipeline, repeats)
        after_seconds, after_value = _median_seconds(after_pipeline, repeats)
        report["pipeline"].append(
            {
                "n": size,
                "legacy_seconds": before_seconds,
                "optimized_seconds": after_seconds,
                "speedup": before_seconds / after_seconds,
                "legacy_peak_mib": _peak_mebibytes(before_pipeline),
                "optimized_peak_mib": _peak_mebibytes(after_pipeline),
                "max_abs_difference": _max_difference(before_value, after_value, fields),
            }
        )

        theta = np.asarray(after_value.theta, dtype=float)
        structural = np.asarray(after_value.S, dtype=float) > 0.0
        before_sigma = lambda: legacy_sigma_theta(
            theta,
            xi,
            tau,
            valid_mask=structural,
        )
        after_sigma = lambda: sigma_theta(
            theta,
            xi,
            tau,
            valid_mask=structural,
        )
        before_seconds, before_value = _median_seconds(before_sigma, repeats)
        after_seconds, after_value = _median_seconds(after_sigma, repeats)
        domain_equal, difference = _nan_domain_difference(before_value, after_value)
        report["sigma_theta"].append(
            {
                "n": size,
                "legacy_seconds": before_seconds,
                "optimized_seconds": after_seconds,
                "speedup": before_seconds / after_seconds,
                "legacy_peak_mib": _peak_mebibytes(before_sigma),
                "optimized_peak_mib": _peak_mebibytes(after_sigma),
                "finite_domain_equal": domain_equal,
                "max_abs_difference": difference,
            }
        )

    profile_size = sizes[-1]
    profile_xi, profile_u = _signal(profile_size)
    report["optimized_stage_profile"] = {
        "n": profile_size,
        **_profile_stages(
            profile_xi,
            profile_u,
            A_ref=A_ref,
            tau=tau,
            w=w,
        ),
    }

    workload_size = sizes[0] if quick else sizes[1]
    xi, u = _signal(workload_size)
    result = compute_agencity(
        u=u,
        xi=xi,
        A_ref=A_ref,
        tau=tau,
        w=w,
        P_c=P_c,
    )
    analysis = lambda: analyze_agencity(result)
    analysis_seconds, _ = _median_seconds(analysis, repeats)

    scales = np.asarray([32.0, 48.0, 64.0, 96.0])
    multiscale = lambda: compute_agencity_spectrum(
        u,
        xi,
        scales,
        A_ref=A_ref,
        P_c=P_c,
        windows=scales,
    )
    multiscale_seconds, _ = _median_seconds(multiscale, repeats)

    multivariate_u = np.column_stack(
        [
            u,
            0.8 * np.sin(0.013 * xi + 0.3),
            0.6 * np.cos(0.021 * xi - 0.2),
        ]
    )
    multivariate = lambda: compute_multivariate_agencity(
        multivariate_u,
        xi,
        A_ref=np.asarray([1.5, 1.2, 1.0]),
        tau=np.asarray([64.0, 48.0, 96.0]),
        w=np.asarray([64.0, 48.0, 96.0]),
        P_c=np.asarray([2.5, 3.0, 1.5]),
    )
    multivariate_seconds, _ = _median_seconds(multivariate, repeats)

    batch_items = [
        {
            "xi": xi,
            "u": u * (1.0 + 0.02 * index),
            "A_ref": A_ref,
            "tau": tau,
            "w": w,
            "P_c": P_c + index,
        }
        for index in range(4)
    ]
    batch_serial = lambda: run_batch(batch_items, analyze=False)
    batch_parallel = lambda: run_batch(
        batch_items,
        analyze=False,
        parallel=True,
        executor="thread",
        max_workers=2,
    )
    batch_serial_seconds, serial_values = _median_seconds(batch_serial, repeats)
    batch_parallel_seconds, parallel_values = _median_seconds(batch_parallel, repeats)
    batch_parallel_difference = max(
        float(np.max(np.abs(left.b - right.b)))
        for left, right in zip(serial_values, parallel_values)
    )

    def stream():
        processor = AgencityStream(
            analyze=False,
            A_ref=A_ref,
            tau=tau,
            w=w,
            P_c=P_c,
        )
        final = None
        for indices in np.array_split(np.arange(workload_size), 4):
            try:
                final = processor.update(u[indices], xi[indices])
            except StreamNotReadyError:
                continue
        return final

    streaming_seconds, stream_result = _median_seconds(stream, repeats)
    stream_difference = _max_difference(
        {field: getattr(result, field) for field in fields},
        stream_result,
        fields,
    )

    report["representative_workloads"] = {
        "n": workload_size,
        "analysis_seconds": analysis_seconds,
        "analysis_peak_mib": _peak_mebibytes(analysis),
        "multiscale_4_scales_seconds": multiscale_seconds,
        "multiscale_4_scales_peak_mib": _peak_mebibytes(multiscale),
        "multivariate_3_components_seconds": multivariate_seconds,
        "multivariate_3_components_peak_mib": _peak_mebibytes(multivariate),
        "batch_4_serial_seconds": batch_serial_seconds,
        "batch_4_serial_peak_mib": _peak_mebibytes(batch_serial),
        "batch_4_threaded_seconds": batch_parallel_seconds,
        "batch_threaded_max_abs_difference": batch_parallel_difference,
        "stream_4_chunks_seconds": streaming_seconds,
        "stream_4_chunks_peak_mib": _peak_mebibytes(stream),
        "stream_final_max_abs_difference": stream_difference,
    }
    return report


def _print_report(report) -> None:
    print("AgencityLab v0.8 engineering benchmark")
    print(json.dumps(report["environment"], sort_keys=True))
    for group in ("crm", "pipeline", "sigma_theta"):
        print(f"\n{group.upper()}")
        for row in report[group]:
            domain = (
                f"  finite_domain_equal={row['finite_domain_equal']}"
                if "finite_domain_equal" in row
                else ""
            )
            print(
                f"n={row['n']:>7d}  before={row['legacy_seconds']:.6f}s  "
                f"after={row['optimized_seconds']:.6f}s  "
                f"speedup={row['speedup']:.2f}x  "
                f"max_abs_diff={row['max_abs_difference']:.3e}  "
                f"peak={row['legacy_peak_mib']:.2f}->{row['optimized_peak_mib']:.2f} MiB"
                f"{domain}"
            )
    print("\nOPTIMIZED_STAGE_PROFILE")
    print(json.dumps(report["optimized_stage_profile"], sort_keys=True))
    print("\nREPRESENTATIVE_WORKLOADS")
    print(json.dumps(report["representative_workloads"], sort_keys=True))
    print("\nBENCHMARK_JSON")
    print(json.dumps(report, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="use CI-sized workloads")
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.repeats is not None and args.repeats < 1:
        parser.error("--repeats must be >= 1")

    report = run_benchmark(quick=args.quick, repeats=args.repeats)
    _print_report(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
