"""Batch utilities for the stable AgencityLab computational API."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, Literal, Optional

import numpy as np

from agencitylab.exceptions import AgencityValidationError, BatchItemError
from agencitylab.models import ExperimentMetadata

from .analyze import analyze_agencity
from .compute import compute_agencity
from .validation import validate_batch_items, validate_metadata

ExecutorKind = Literal["thread", "process"]


def _extract_item(item, *, index: int):
    """Return ``(xi, u, item_options)`` for one supported batch item."""
    if isinstance(item, dict):
        payload = dict(item)
        if "u" in payload and "data" in payload:
            raise BatchItemError(
                f"batch item {index}: provide only one of 'u' or 'data'"
            )
        xi = payload.pop("xi", None)
        u = payload.pop("u", payload.pop("data", None))
        if u is None:
            raise BatchItemError(f"batch item {index}: missing 'u' or 'data'")
        return xi, u, payload

    # Pre-v0.3 accepted both tuple and list pairs. Preserve that public input
    # contract; a two-sample raw scalar signal is invalid anyway because the
    # canonical API requires at least three samples.
    if isinstance(item, (tuple, list)) and len(item) == 2:
        return item[0], item[1], {}

    return None, item, {}


def _merge_metadata(global_metadata, local_metadata):
    merged = validate_metadata(global_metadata)
    if local_metadata is None:
        return merged

    if isinstance(local_metadata, ExperimentMetadata):
        local_payload = local_metadata.to_dict()
    elif isinstance(local_metadata, dict):
        # Preserve only keys the item actually supplied. Validating the local
        # dictionary first would materialize empty defaults and accidentally
        # overwrite batch-wide physical metadata.
        local_payload = dict(local_metadata)
    else:
        raise AgencityValidationError(
            "item metadata must be a dictionary, ExperimentMetadata, or None"
        )

    merged.update(local_payload)
    return validate_metadata(merged)


def _compute_one(payload):
    index, xi, u, preset, config, metadata, verbose, kwargs = payload
    try:
        return compute_agencity(
            u=u,
            xi=xi,
            preset=preset,
            config=config,
            metadata=metadata,
            verbose=verbose,
            **kwargs,
        )
    except Exception as exc:
        if isinstance(exc, BatchItemError):
            raise
        raise BatchItemError(f"batch item {index} failed: {exc}") from exc


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
    metadata: Optional[Dict[str, Any] | ExperimentMetadata] = None,
    **kwargs,
):
    """Compute Agencity for multiple scalar signals.

    Supported items are raw signals, ``(xi, u)`` tuple/list pairs, or dictionaries.
    Item dictionaries may additionally contain any ``compute_agencity`` keyword,
    ``metadata``, ``config``, or ``preset``. Per-item values override batch-wide
    values without mutating the supplied dictionaries.

    Results preserve input order in both serial and parallel execution. Failures
    are re-raised as :class:`BatchItemError` with the failing zero-based item
    index.
    """
    materialized = validate_batch_items(items)
    if executor not in {"thread", "process"}:
        raise AgencityValidationError("executor must be 'thread' or 'process'")
    if max_workers is not None and (not isinstance(max_workers, int) or max_workers < 1):
        raise AgencityValidationError("max_workers must be a positive integer or None")

    global_config = dict(config or {})
    global_metadata = validate_metadata(metadata)
    global_kwargs = dict(kwargs)

    payloads = []
    for index, item in enumerate(materialized):
        xi, u, local = _extract_item(item, index=index)
        item_preset = local.pop("preset", preset)
        item_config = dict(global_config)
        local_config = local.pop("config", None)
        if local_config is not None:
            if not isinstance(local_config, dict):
                raise BatchItemError(f"batch item {index}: config must be a dictionary")
            item_config.update(local_config)

        try:
            item_metadata = _merge_metadata(global_metadata, local.pop("metadata", None))
        except AgencityValidationError as exc:
            raise BatchItemError(f"batch item {index}: {exc}") from exc

        item_kwargs = dict(global_kwargs)
        item_kwargs.update(local)
        payloads.append(
            (
                index,
                xi,
                u,
                item_preset,
                item_config,
                item_metadata,
                verbose,
                item_kwargs,
            )
        )

    if not parallel or len(payloads) == 1:
        results = [_compute_one(payload) for payload in payloads]
    else:
        pool_cls = ProcessPoolExecutor if executor == "process" else ThreadPoolExecutor
        ordered = [None] * len(payloads)
        with pool_cls(max_workers=max_workers) as pool:
            futures = {pool.submit(_compute_one, payload): payload[0] for payload in payloads}
            for future in as_completed(futures):
                index = futures[future]
                try:
                    ordered[index] = future.result()
                except Exception as exc:
                    if isinstance(exc, BatchItemError):
                        raise
                    raise BatchItemError(f"batch item {index} failed: {exc}") from exc
        results = ordered

    if not analyze:
        return results

    analyses = []
    for index, result in enumerate(results):
        if verbose:
            print(f"[batch] analyzing item {index + 1}/{len(results)}")
        analysis = analyze_agencity(result, verbose=verbose)
        result.attach_analysis(analysis)
        analyses.append(analysis)

    return {"results": results, "analyses": analyses}


def analyze_batch(results: Iterable[Any], *, verbose: bool = False):
    """Analyze a materialized sequence of AgencityResult objects."""
    analyses = []
    for index, result in enumerate(list(results)):
        if verbose:
            print(f"[batch] analyzing result {index + 1}")
        analysis = analyze_agencity(result, verbose=verbose)
        result.attach_analysis(analysis)
        analyses.append(analysis)
    return analyses


def summarize_batch(results: Iterable[Any]) -> Dict[str, Any]:
    """Return descriptive batch statistics without changing canonical results."""
    results = list(results)
    if not results:
        return {}

    taus = np.asarray([result.tau for result in results], dtype=float)
    b_means = np.asarray([result.b_mean for result in results], dtype=float)
    beta_means = np.asarray([result.beta_mean for result in results], dtype=float)

    regimes = []
    for result in results:
        if result.analysis:
            regimes.append(result.analysis.get("regime", "unknown"))
        else:
            regimes.append("unknown")

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
    """Compatibility alias for :func:`summarize_batch`."""
    return summarize_batch(results)