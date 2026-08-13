"""Batch utilities for the stable AgencityLab computational API."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any, Iterable, Literal

import numpy as np

from agencitylab.exceptions import AgencityValidationError, BatchItemError
from agencitylab.models import ExperimentMetadata

from .analyze import analyze_agencity
from .compute import compute_agencity
from .validation import validate_batch_items, validate_metadata

ExecutorKind = Literal["thread", "process"]


def _extract_item(item: Any, *, index: int) -> tuple[Any, Any, dict[str, Any]]:
    """Return ``(xi, u, item_options)`` for one supported batch item."""
    if isinstance(item, dict):
        payload = dict(item)
        xi = payload.pop("xi", None)
        if "u" not in payload:
            raise BatchItemError(f"batch item {index}: missing 'u'")
        u = payload.pop("u")
        return xi, u, payload

    if isinstance(item, tuple) and len(item) == 2:
        return item[0], item[1], {}

    return None, item, {}


def _merge_metadata(global_metadata: Any, local_metadata: Any) -> dict[str, Any]:
    merged = validate_metadata(global_metadata)
    if local_metadata is None:
        return merged

    if isinstance(local_metadata, ExperimentMetadata):
        local_payload = local_metadata.to_dict()
    elif isinstance(local_metadata, dict):
        local_payload = dict(local_metadata)
    else:
        raise AgencityValidationError(
            "item metadata must be a dictionary, ExperimentMetadata, or None"
        )

    merged.update(local_payload)
    return validate_metadata(merged)


def _compute_one(payload: tuple[Any, ...]):
    index, xi, u, metadata, verbose, kwargs = payload
    try:
        return compute_agencity(
            u=u,
            xi=xi,
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
    analyze: bool = False,
    parallel: bool = False,
    executor: ExecutorKind = "thread",
    max_workers: int | None = None,
    verbose: bool = False,
    metadata: dict[str, Any] | ExperimentMetadata | None = None,
    **compute_kwargs: Any,
):
    """Compute Agencity for multiple scalar signals.

    Supported items are raw signals, ``(xi, u)`` tuples, or dictionaries with a
    required ``u`` key. Item dictionaries may contain any explicit
    :func:`compute_agencity` keyword plus optional ``metadata``. Per-item values
    override batch-wide compute arguments without mutating caller dictionaries.

    Results preserve input order in both serial and parallel execution. Failures
    are re-raised as :class:`BatchItemError` with the zero-based item index.

    When ``analyze=True``, diagnostics are returned separately from canonical
    results as ``{"results": ..., "analyses": ...}``.
    """
    materialized = validate_batch_items(items)
    if executor not in {"thread", "process"}:
        raise AgencityValidationError("executor must be 'thread' or 'process'")
    if max_workers is not None and (not isinstance(max_workers, int) or max_workers < 1):
        raise AgencityValidationError("max_workers must be a positive integer or None")

    global_metadata = validate_metadata(metadata)
    global_kwargs = dict(compute_kwargs)

    payloads = []
    for index, item in enumerate(materialized):
        xi, u, local = _extract_item(item, index=index)
        try:
            item_metadata = _merge_metadata(global_metadata, local.pop("metadata", None))
        except AgencityValidationError as exc:
            raise BatchItemError(f"batch item {index}: {exc}") from exc

        item_kwargs = dict(global_kwargs)
        item_kwargs.update(local)
        payloads.append((index, xi, u, item_metadata, verbose, item_kwargs))

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

    analyses = [analyze_agencity(result, verbose=verbose) for result in results]
    return {"results": results, "analyses": analyses}


def analyze_batch(results: Iterable[Any], *, verbose: bool = False) -> list[dict[str, Any]]:
    """Analyze results without mutating their canonical data model."""
    return [analyze_agencity(result, verbose=verbose) for result in list(results)]


def summarize_batch(results: Iterable[Any]) -> dict[str, Any]:
    """Return descriptive batch statistics without changing canonical results."""
    materialized = list(results)
    if not materialized:
        return {}

    taus = np.asarray([result.tau for result in materialized], dtype=float)
    b_means = np.asarray([result.b_mean for result in materialized], dtype=float)
    beta_means = np.asarray([result.beta_mean for result in materialized], dtype=float)

    return {
        "n": len(materialized),
        "tau_mean": float(np.mean(taus)),
        "tau_std": float(np.std(taus)),
        "b_mean_mean": float(np.mean(b_means)),
        "b_mean_std": float(np.std(b_means)),
        "beta_mean_mean": float(np.mean(beta_means)),
        "beta_mean_std": float(np.std(beta_means)),
    }
