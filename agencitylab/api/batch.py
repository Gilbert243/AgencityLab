"""
Batch utilities for AgencityLab.

Designed for research workflows:
- process many signals
- optional parallel execution
- optional analysis
- batch-level summaries and comparisons
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Optional, Tuple, Literal

import numpy as np

from .compute import compute_agencity
from .analyze import analyze_agencity
from .validation import validate_batch_items


ExecutorKind = Literal["thread", "process"]


def _extract_item(item):
    """
    Normalize a batch item.

    Accepted forms:
        - raw signal array
        - (xi, u) tuple
        - {"xi": ..., "u": ...}
        - {"data": ...}
    """
    if isinstance(item, dict):
        xi = item.get("xi", None)
        u = item.get("u", item.get("data", None))
        return xi, u

    if isinstance(item, (tuple, list)) and len(item) == 2:
        return item[0], item[1]

    return None, item


def _compute_one(args):
    """
    Top-level worker for multiprocessing/threading.
    """
    xi, u, preset, config, metadata, verbose, kwargs = args
    return compute_agencity(
        data=u,
        xi=xi,
        preset=preset,
        config=config,
        metadata=metadata,
        verbose=verbose,
        **kwargs,
    )


def run_batch(
    items: Iterable[Any],
    *,
    preset: str = "default",
    analyze: bool = False,
    parallel: bool = False,
    executor: ExecutorKind = "thread",
    max_workers: Optional[int] = None,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    **kwargs,
):
    """
    Run Agencity on multiple signals.

    Parameters
    ----------
    items:
        Iterable of signals, (xi, u) pairs, or dictionaries.
    preset:
        Compute preset.
    analyze:
        If True, return analyses alongside results.
    parallel:
        If True, use a worker pool.
    executor:
        "thread" or "process".
    max_workers:
        Worker pool size.
    config:
        Optional compute config passed to every run.
    metadata:
        Optional metadata passed to every run.

    Returns
    -------
    list[AgencityResult]
    or
    {"results": [...], "analyses": [...]}
    """
    items = validate_batch_items(items)
    config = dict(config or {})
    metadata = dict(metadata or {})

    payloads = []
    for item in items:
        xi, u = _extract_item(item)
        payloads.append((xi, u, preset, config, metadata, verbose, dict(kwargs)))

    results = []

    if parallel and len(payloads) > 1:
        if executor == "process":
            pool_cls = ProcessPoolExecutor
        else:
            pool_cls = ThreadPoolExecutor

        with pool_cls(max_workers=max_workers) as pool:
            futures = {pool.submit(_compute_one, p): i for i, p in enumerate(payloads)}
            ordered = [None] * len(payloads)

            for fut in as_completed(futures):
                idx = futures[fut]
                ordered[idx] = fut.result()

            results = ordered
    else:
        for i, p in enumerate(payloads):
            if verbose:
                print(f"[batch] computing item {i + 1}/{len(payloads)}")
            results.append(_compute_one(p))

    if not analyze:
        return results

    analyses = []
    for i, result in enumerate(results):
        if verbose:
            print(f"[batch] analyzing item {i + 1}/{len(results)}")
        analyses.append(analyze_agencity(result, verbose=verbose))

    return {"results": results, "analyses": analyses}


def analyze_batch(results: Iterable[Any], *, verbose: bool = False):
    """
    Analyze a list of AgencityResult objects.
    """
    results = list(results)
    analyses = []

    for i, result in enumerate(results):
        if verbose:
            print(f"[batch] analyzing result {i + 1}/{len(results)}")
        analyses.append(analyze_agencity(result, verbose=verbose))

    return analyses


def summarize_batch(results: Iterable[Any]) -> Dict[str, Any]:
    """
    Summarize a batch of results.
    """
    results = list(results)
    if not results:
        return {}

    taus = np.array([float(r.tau) for r in results], dtype=float)
    b_means = np.array([float(r.b_mean) for r in results], dtype=float)
    beta_means = np.array([float(r.beta_mean) for r in results], dtype=float)
    regimes = [analyze_agencity(r).get("regime", "unknown") for r in results]

    return {
        "n": len(results),
        "tau_mean": float(np.mean(taus)),
        "tau_std": float(np.std(taus)),
        "b_mean_mean": float(np.mean(b_means)),
        "b_mean_std": float(np.std(b_means)),
        "beta_mean_mean": float(np.mean(beta_means)),
        "beta_mean_std": float(np.std(beta_means)),
        "regime_counts": dict(Counter(regimes)),
    }


def compare_batch(results: Iterable[Any]) -> Dict[str, Any]:
    """
    Alias for summarize_batch.
    """
    return summarize_batch(results)