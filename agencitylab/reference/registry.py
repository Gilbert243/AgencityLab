"""Dataset-registry models and validation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from importlib import resources
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlparse

_NAME = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_FORMAT = re.compile(r"^[a-z0-9][a-z0-9.+-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_STATUSES = {"reference_data", "experimental_data", "benchmark_data"}
_REQUIRED = {
    "name",
    "version",
    "description",
    "scientific_status",
    "regime",
    "format",
    "source",
    "sha256",
    "metadata",
}


class DatasetRegistryError(ValueError):
    """Raised when a reference dataset registry is invalid."""


@dataclass(frozen=True, slots=True)
class DatasetRecord:
    """One validated dataset-registry entry."""

    name: str
    version: str
    description: str
    scientific_status: str
    regime: str
    format: str
    source: str
    sha256: str
    metadata: dict[str, Any]
    kind: str
    builtin_resource: str | None = None
    path: str | None = None
    url: str | None = None
    size: int | None = None
    license: str | None = None
    citation: str | None = None

    @property
    def identifier(self) -> str:
        """Return the deterministic ``name@version`` identifier."""

        return f"{self.name}@{self.version}"

    @property
    def filename(self) -> str:
        """Return the validated external or embedded filename."""

        location = self.builtin_resource or self.path
        if location is not None:
            return PurePosixPath(location).name
        if self.url is not None:
            return PurePosixPath(urlparse(self.url).path).name
        raise DatasetRegistryError(f"dataset {self.identifier} has no file location")

    def to_dict(self) -> dict[str, Any]:
        """Return detached public registry metadata."""

        result = {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "scientific_status": self.scientific_status,
            "regime": self.regime,
            "format": self.format,
            "source": self.source,
            "sha256": self.sha256,
            "metadata": dict(self.metadata),
            "kind": self.kind,
        }
        for key in ("builtin_resource", "path", "url", "size", "license", "citation"):
            value = getattr(self, key)
            if value is not None:
                result[key] = value
        return result


def _safe_relative_path(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise DatasetRegistryError(f"{field} must be a non-empty POSIX relative path")
    if "\\" in value:
        raise DatasetRegistryError(f"{field} must not contain backslashes")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise DatasetRegistryError(f"{field} is not a safe relative path: {value!r}")
    return value


def _text(entry: dict[str, Any], field: str) -> str:
    value = entry.get(field)
    if not isinstance(value, str) or not value.strip():
        raise DatasetRegistryError(f"{field} must be a non-empty string")
    return value.strip()


def _validate_entry(raw: Any, index: int) -> DatasetRecord:
    if not isinstance(raw, dict):
        raise DatasetRegistryError(f"datasets[{index}] must be an object")
    missing = sorted(_REQUIRED - raw.keys())
    if missing:
        raise DatasetRegistryError(f"datasets[{index}] missing fields: {', '.join(missing)}")

    name = _text(raw, "name")
    version = _text(raw, "version")
    data_format = _text(raw, "format").lower().lstrip(".")
    if not _NAME.fullmatch(name):
        raise DatasetRegistryError(f"invalid dataset name: {name!r}")
    if not _NAME.fullmatch(version):
        raise DatasetRegistryError(f"invalid dataset version: {version!r}")
    if not _FORMAT.fullmatch(data_format):
        raise DatasetRegistryError(f"invalid dataset format: {data_format!r}")

    status = _text(raw, "scientific_status")
    if status not in _STATUSES:
        raise DatasetRegistryError(
            f"scientific_status must be one of {sorted(_STATUSES)}; got {status!r}"
        )
    checksum = _text(raw, "sha256").lower()
    if not _SHA256.fullmatch(checksum):
        raise DatasetRegistryError("sha256 must contain exactly 64 lowercase hexadecimal digits")
    metadata = raw.get("metadata")
    if not isinstance(metadata, dict):
        raise DatasetRegistryError("metadata must be a JSON object")

    kind = _text(raw, "kind")
    if kind not in {"builtin", "remote"}:
        raise DatasetRegistryError("kind must be 'builtin' or 'remote'")
    builtin_resource = raw.get("builtin_resource")
    path = raw.get("path")
    url = raw.get("url")
    if kind == "builtin":
        builtin_resource = _safe_relative_path(builtin_resource, "builtin_resource")
        if path is not None or url is not None:
            raise DatasetRegistryError("builtin entries cannot define path or url")
    else:
        if (path is None) == (url is None):
            raise DatasetRegistryError("remote entries must define exactly one of path or url")
        if path is not None:
            path = _safe_relative_path(path, "path")
        if url is not None:
            if not isinstance(url, str) or urlparse(url).scheme != "https" or not urlparse(url).netloc:
                raise DatasetRegistryError("url must be an absolute HTTPS URL")

    size = raw.get("size")
    if size is not None:
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise DatasetRegistryError("size must be a non-negative integer")

    return DatasetRecord(
        name=name,
        version=version,
        description=_text(raw, "description"),
        scientific_status=status,
        regime=_text(raw, "regime"),
        format=data_format,
        source=_text(raw, "source"),
        sha256=checksum,
        metadata=dict(metadata),
        kind=kind,
        builtin_resource=builtin_resource,
        path=path,
        url=url,
        size=size,
        license=raw.get("license"),
        citation=raw.get("citation"),
    )


def validate_manifest(payload: Any) -> tuple[DatasetRecord, ...]:
    """Validate a registry payload and return immutable records."""

    if not isinstance(payload, dict):
        raise DatasetRegistryError("registry root must be a JSON object")
    if payload.get("schema_version") != "1.0":
        raise DatasetRegistryError("registry schema_version must be '1.0'")
    datasets = payload.get("datasets")
    if not isinstance(datasets, list):
        raise DatasetRegistryError("registry datasets must be a list")
    records = tuple(_validate_entry(item, index) for index, item in enumerate(datasets))
    identifiers = [record.identifier for record in records]
    if len(identifiers) != len(set(identifiers)):
        raise DatasetRegistryError("dataset name/version pairs must be unique")
    return records


def load_local_registry() -> tuple[DatasetRecord, ...]:
    """Load the registry snapshot distributed with AgencityLab."""

    resource = resources.files("agencitylab.reference").joinpath("data/registry.json")
    with resource.open("r", encoding="utf-8") as stream:
        return validate_manifest(json.load(stream))


def resolve_record(
    records: tuple[DatasetRecord, ...], name: str, version: str | None = None
) -> DatasetRecord:
    """Resolve a dataset name and optional version deterministically."""

    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise DatasetRegistryError(f"invalid dataset name: {name!r}")
    candidates = [record for record in records if record.name == name]
    if version is not None:
        candidates = [record for record in candidates if record.version == version]
    if not candidates:
        suffix = "" if version is None else f" version {version!r}"
        raise KeyError(f"dataset {name!r}{suffix} is not registered")
    if version is not None or len(candidates) == 1:
        return candidates[0]
    return max(candidates, key=lambda item: _version_key(item.version))


def _version_key(version: str) -> tuple[str, ...]:
    parts = re.split(r"([0-9]+)", version)
    return tuple(("1" + part.zfill(16)) if part.isdigit() else ("0" + part) for part in parts)


__all__ = [
    "DatasetRecord",
    "DatasetRegistryError",
    "load_local_registry",
    "resolve_record",
    "validate_manifest",
]
