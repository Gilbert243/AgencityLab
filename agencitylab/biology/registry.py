"""Explicit registry for named biological reference bundles.

The registry ships with no biological defaults. Entries are supplied by users or
future source-backed reference datasets and remain explicitly versioned.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import (
    BiologicalMemoryReference,
    BiologicalReference,
    BiologicalTimeReference,
)


@dataclass(frozen=True, slots=True)
class BiologicalReferenceBundle:
    """One versioned reference with optional calibrated ``tau`` and ``w``."""

    reference: BiologicalReference
    time_reference: BiologicalTimeReference | None = None
    memory_reference: BiologicalMemoryReference | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if self.time_reference is not None and self.time_reference.reference != self.reference:
            raise ValueError("time_reference must belong to the registered biological reference")
        if self.memory_reference is not None and self.memory_reference.reference != self.reference:
            raise ValueError("memory_reference must belong to the registered biological reference")
        object.__setattr__(self, "note", str(self.note).strip())


class BiologicalReferenceRegistry:
    """In-memory registry of explicit, versioned biological reference bundles."""

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], BiologicalReferenceBundle] = {}

    def register(
        self,
        bundle: BiologicalReferenceBundle,
        *,
        replace: bool = False,
    ) -> None:
        """Register a named reference bundle without inventing fallback values."""
        identifier = bundle.reference.identifier
        if not identifier:
            raise ValueError("registered biological references require an identifier")
        key = (identifier, bundle.reference.version)
        if key in self._entries and not replace:
            raise ValueError(f"biological reference {identifier!r} version {key[1]!r} exists")
        self._entries[key] = bundle

    def get(self, identifier: str, *, version: str) -> BiologicalReferenceBundle:
        """Return an exact reference version; no latest-version fallback is inferred."""
        key = (str(identifier).strip(), str(version).strip())
        try:
            return self._entries[key]
        except KeyError as exc:
            raise KeyError(
                f"unknown biological reference {key[0]!r} version {key[1]!r}"
            ) from exc

    def available(self) -> tuple[tuple[str, str], ...]:
        """Return registered ``(identifier, version)`` pairs in stable order."""
        return tuple(sorted(self._entries))
