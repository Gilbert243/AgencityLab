"""
Caching helpers for AgencityLab pipeline execution.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from typing import Any, Dict


class PipelineCache:
    """Very small in-memory cache for pipeline artifacts."""
    def __init__(self):
        self._store: Dict[str, Any] = {}

    @staticmethod
    def _normalize_key(value: Any) -> str:
        """Convert a Python object into a stable cache key."""
        if is_dataclass(value):
            value = asdict(value)
        try:
            payload = json.dumps(value, sort_keys=True, default=str)
        except TypeError:
            payload = repr(value)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def make_key(self, *items: Any) -> str:
        """Build a compound key from several objects."""
        payload = [self._normalize_key(item) for item in items]
        return hashlib.sha256("|".join(payload).encode("utf-8")).hexdigest()

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a cached item."""
        return self._store.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Store a cached item."""
        self._store[key] = value

    def clear(self) -> None:
        """Clear the cache."""
        self._store.clear()
