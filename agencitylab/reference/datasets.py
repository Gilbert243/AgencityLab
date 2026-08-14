"""Discovery, safe loading, and explicit download of reference datasets."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from importlib import resources
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

import numpy as np

from agencitylab.models import AgencityDataset, AgencitySignal, ExperimentMetadata

from ._download import (
    ChecksumMismatchError,
    DatasetDownloadError,
    download_file,
    sha256_file,
    validate_filename,
    validate_url,
)
from .registry import (
    DatasetRecord,
    DatasetRegistryError,
    load_local_registry,
    resolve_record,
    validate_manifest,
)

_OFFICIAL_REPOSITORY = "Gilbert243/AgencityLab"
_REGISTRY_PATH = "reference_datasets/registry.json"
_MAX_REGISTRY_BYTES = 5 * 1024 * 1024


def cache_dir() -> Path:
    """Return the user cache location without creating it."""

    if sys.platform == "win32":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Caches"
    else:
        root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return root / "agencitylab" / "reference"


def _local_records(kind: str | None = None) -> tuple[DatasetRecord, ...]:
    records = load_local_registry()
    if kind is None:
        return records
    return tuple(record for record in records if record.kind == kind)


def _names(records: tuple[DatasetRecord, ...]) -> tuple[str, ...]:
    return tuple(sorted({record.name for record in records}))


def available() -> tuple[str, ...]:
    """List dataset names in the deterministic packaged registry snapshot."""

    return _names(_local_records())


def available_builtin() -> tuple[str, ...]:
    """List datasets embedded in the installed AgencityLab distribution."""

    return _names(_local_records("builtin"))


def available_remote(
    *, refresh: bool = False, ref: str = "main", timeout: float = 10.0
) -> tuple[str, ...]:
    """List downloadable datasets.

    By default this uses the packaged, offline registry snapshot.  Set
    ``refresh=True`` to explicitly consult the official GitHub registry.
    """

    records = remote_registry(ref=ref, timeout=timeout) if refresh else _local_records("remote")
    return _names(records)


def registry(
    *, refresh_remote: bool = False, ref: str = "main", timeout: float = 10.0
) -> list[dict[str, Any]]:
    """Return detached metadata for the local registry and optional remote refresh."""

    records = list(_local_records())
    if refresh_remote:
        refreshed = remote_registry(ref=ref, timeout=timeout)
        records = [record for record in records if record.kind == "builtin"] + list(refreshed)
    return [record.to_dict() for record in records]


def _official_raw_url(path: str, ref: str) -> str:
    if not isinstance(ref, str) or not ref.strip() or any(char.isspace() for char in ref):
        raise ValueError("ref must be a non-empty Git ref without whitespace")
    safe_path = "/".join(quote(part, safe="") for part in PurePosixPath(path).parts)
    safe_ref = quote(ref, safe="")
    return f"https://raw.githubusercontent.com/{_OFFICIAL_REPOSITORY}/{safe_ref}/{safe_path}"


def _read_registry_url(url: str, timeout: float) -> tuple[DatasetRecord, ...]:
    request = Request(validate_url(url), headers={"User-Agent": "AgencityLab-reference/1"})
    timeout_value = float(timeout)
    if timeout_value <= 0.0:
        raise ValueError("timeout must be strictly positive")
    try:
        with urlopen(request, timeout=timeout_value) as response:
            content_length = response.headers.get("Content-Length")
            if content_length is not None and int(content_length) > _MAX_REGISTRY_BYTES:
                raise DatasetRegistryError("remote registry exceeds the 5 MiB safety limit")
            payload = response.read(_MAX_REGISTRY_BYTES + 1)
    except HTTPError as exc:
        raise DatasetDownloadError(f"HTTP {exc.code} while fetching dataset registry") from exc
    except (URLError, OSError) as exc:
        raise DatasetDownloadError(f"failed to fetch dataset registry: {exc}") from exc
    if len(payload) > _MAX_REGISTRY_BYTES:
        raise DatasetRegistryError("remote registry exceeds the 5 MiB safety limit")
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DatasetRegistryError("remote registry is not valid UTF-8 JSON") from exc
    records = validate_manifest(document)
    if any(record.kind != "remote" for record in records):
        raise DatasetRegistryError("official remote registry may contain only remote datasets")
    return records


def remote_registry(*, ref: str = "main", timeout: float = 10.0) -> tuple[DatasetRecord, ...]:
    """Explicitly fetch and validate the official registry at a Git ref."""

    return _read_registry_url(_official_raw_url(_REGISTRY_PATH, ref), timeout)


def info(
    name: str,
    *,
    version: str | None = None,
    refresh_remote: bool = False,
    ref: str = "main",
    timeout: float = 10.0,
) -> dict[str, Any]:
    """Return metadata for one registered dataset."""

    records = _local_records()
    if refresh_remote:
        records = _local_records("builtin") + remote_registry(ref=ref, timeout=timeout)
    return resolve_record(records, name, version).to_dict()


def _builtin_record(name: str, version: str | None) -> DatasetRecord:
    record = resolve_record(_local_records("builtin"), name, version)
    if record.builtin_resource is None:
        raise DatasetRegistryError(f"builtin dataset {record.identifier} has no resource")
    return record


def _dataset_from_csv(path: Path, record: DatasetRecord) -> AgencityDataset | np.ndarray:
    table = np.genfromtxt(path, delimiter=",", names=True, encoding="utf-8")
    signals = record.metadata.get("signals")
    coordinate = record.metadata.get("coordinate_column")
    if not isinstance(signals, list) or not isinstance(coordinate, str):
        return table
    if table.dtype.names is None or coordinate not in table.dtype.names:
        raise DatasetRegistryError(f"coordinate column {coordinate!r} is absent from {record.filename}")

    items: list[AgencitySignal] = []
    for specification in signals:
        if not isinstance(specification, dict) or not isinstance(specification.get("column"), str):
            raise DatasetRegistryError("dataset signal specifications must define a column")
        column = specification["column"]
        if column not in table.dtype.names:
            raise DatasetRegistryError(f"signal column {column!r} is absent from {record.filename}")
        signal_metadata = ExperimentMetadata.from_dict(specification.get("metadata", {}))
        items.append(
            AgencitySignal(
                xi=np.asarray(table[coordinate], dtype=float),
                u=np.asarray(table[column], dtype=float),
                metadata=signal_metadata,
            )
        )
    dataset_metadata = ExperimentMetadata.from_dict(
        {
            "title": record.name,
            "description": record.description,
            "source": record.source,
            "domain": record.metadata.get("domain", "dynamical systems"),
            "tags": ["reference", "embedded", record.regime],
            "created_at": "",
            "extra": {
                "dataset_name": record.name,
                "dataset_version": record.version,
                "scientific_status": record.scientific_status,
                "sha256": record.sha256,
            },
        }
    )
    return AgencityDataset(items=items, metadata=dataset_metadata)


def load_path(path: str | Path, *, format: str | None = None) -> Any:
    """Safely parse a local simple-data file without executing its contents."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    data_format = (format or source.suffix.lstrip(".")).lower()
    if data_format == "npy":
        return np.load(source, allow_pickle=False)
    if data_format == "npz":
        with np.load(source, allow_pickle=False) as archive:
            return {name: archive[name].copy() for name in archive.files}
    if data_format == "csv":
        return np.genfromtxt(source, delimiter=",", names=True, encoding="utf-8")
    if data_format == "json":
        with source.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if isinstance(payload, dict) and "items" in payload:
            return AgencityDataset.from_dict(payload)
        return payload
    if data_format == "txt":
        return source.read_text(encoding="utf-8")
    raise ValueError(f"no safe loader is registered for format {data_format!r}")


def load(name: str, *, version: str | None = None) -> Any:
    """Load an embedded dataset offline and verify its packaged checksum."""

    record = _builtin_record(name, version)
    resource_path = record.builtin_resource
    if resource_path is None:
        raise DatasetRegistryError(f"builtin dataset {record.identifier} has no resource")
    resource = resources.files("agencitylab.reference").joinpath(resource_path)
    if not resource.is_file():
        raise FileNotFoundError(f"packaged dataset resource is missing: {record.builtin_resource}")
    with resources.as_file(resource) as path:
        if sha256_file(path) != record.sha256:
            raise ChecksumMismatchError(f"packaged dataset checksum mismatch: {record.identifier}")
        if record.format == "csv":
            return _dataset_from_csv(path, record)
        return load_path(path, format=record.format)


def _remote_records(refresh_registry: bool, ref: str, timeout: float) -> tuple[DatasetRecord, ...]:
    return remote_registry(ref=ref, timeout=timeout) if refresh_registry else _local_records("remote")


def _record_url(record: DatasetRecord, ref: str) -> str:
    if record.url is not None:
        return record.url
    if record.path is None:
        raise DatasetRegistryError(f"remote dataset {record.identifier} has no path or URL")
    return _official_raw_url(record.path, ref)


def _destination(record: DatasetRecord, destination: str | Path | None) -> Path:
    if destination is not None:
        return Path(destination).expanduser()
    return cache_dir() / record.name / record.version


def _write_provenance(path: Path, record: DatasetRecord, source_url: str) -> None:
    payload = {
        "name": record.name,
        "version": record.version,
        "source_url": source_url,
        "sha256": record.sha256,
        "local_path": str(path.resolve()),
    }
    sidecar = path.with_name(path.name + ".metadata.json")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{sidecar.name}.",
            suffix=".part",
            dir=sidecar.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        temporary_path.replace(sidecar)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def download(
    name: str,
    *,
    version: str | None = None,
    destination: str | Path | None = None,
    ref: str = "main",
    timeout: float = 30.0,
    force: bool = False,
    refresh_registry: bool = False,
) -> Path:
    """Explicitly download and verify an official registered dataset."""

    record = resolve_record(_remote_records(refresh_registry, ref, timeout), name, version)
    source_url = _record_url(record, ref)
    path = download_file(
        source_url,
        destination=_destination(record, destination),
        filename=record.filename,
        expected_sha256=record.sha256,
        timeout=timeout,
        force=force,
    )
    _write_provenance(path, record, source_url)
    return path


def download_url(
    url: str,
    *,
    destination: str | Path | None = None,
    filename: str | None = None,
    expected_sha256: str | None = None,
    timeout: float = 30.0,
    force: bool = False,
) -> Path:
    """Explicitly download an arbitrary user-supplied data URL.

    No parser is invoked.  Supplying ``expected_sha256`` is recommended when a
    trusted digest is available.
    """

    source_url = validate_url(url)
    inferred = PurePosixPath(urlparse(source_url).path).name
    safe_name = validate_filename(filename or inferred)
    target_directory = cache_dir() / "urls" if destination is None else Path(destination)
    return download_file(
        source_url,
        destination=target_directory,
        filename=safe_name,
        expected_sha256=expected_sha256,
        timeout=timeout,
        force=force,
    )


def local_path(
    name: str,
    *,
    version: str | None = None,
    destination: str | Path | None = None,
) -> Path:
    """Return the expected local path for a packaged-registry remote dataset."""

    record = resolve_record(_local_records("remote"), name, version)
    return _destination(record, destination) / record.filename


def is_downloaded(
    name: str,
    *,
    version: str | None = None,
    destination: str | Path | None = None,
) -> bool:
    """Return whether a cached file exists and matches its official checksum."""

    record = resolve_record(_local_records("remote"), name, version)
    path = _destination(record, destination) / record.filename
    return path.is_file() and sha256_file(path) == record.sha256


def remove(
    name: str,
    *,
    version: str | None = None,
    destination: str | Path | None = None,
) -> bool:
    """Remove one downloaded dataset and its provenance sidecar."""

    path = local_path(name, version=version, destination=destination)
    existed = path.is_file()
    path.unlink(missing_ok=True)
    path.with_name(path.name + ".metadata.json").unlink(missing_ok=True)
    return existed


__all__ = [
    "ChecksumMismatchError",
    "DatasetDownloadError",
    "DatasetRegistryError",
    "available",
    "available_builtin",
    "available_remote",
    "cache_dir",
    "download",
    "download_url",
    "info",
    "is_downloaded",
    "load",
    "load_path",
    "local_path",
    "registry",
    "remote_registry",
    "remove",
]
