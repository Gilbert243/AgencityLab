from __future__ import annotations

import json
from pathlib import Path

import pytest

from agencitylab.reference._download import sha256_file
from agencitylab.reference.registry import DatasetRegistryError, validate_manifest


ROOT = Path(__file__).resolve().parents[2]


def _manifest():
    return json.loads((ROOT / "agencitylab/reference/data/registry.json").read_text())


def test_packaged_registry_schema_and_official_files_are_valid():
    records = validate_manifest(_manifest())
    assert len({record.identifier for record in records}) == len(records)
    for record in records:
        if record.kind == "builtin":
            path = ROOT / "agencitylab/reference" / str(record.builtin_resource)
        else:
            path = ROOT / str(record.path)
        assert path.is_file(), record.identifier
        assert path.stat().st_size == record.size
        assert sha256_file(path) == record.sha256
        assert record.description
        assert record.source
        assert record.metadata


def test_remote_registry_contains_only_repository_backed_files():
    payload = json.loads((ROOT / "reference_datasets/registry.json").read_text())
    records = validate_manifest(payload)
    assert records and all(record.kind == "remote" for record in records)
    for record in records:
        assert record.path is not None
        assert (ROOT / record.path).is_file()
        assert sha256_file(ROOT / record.path) == record.sha256


@pytest.mark.parametrize(
    "mutate",
    (
        lambda payload: payload.update(schema_version="2"),
        lambda payload: payload["datasets"].append(dict(payload["datasets"][0])),
        lambda payload: payload["datasets"][0].update(sha256="invalid"),
        lambda payload: payload["datasets"][0].update(scientific_status="proof"),
        lambda payload: payload["datasets"][0].update(metadata="invalid"),
        lambda payload: payload["datasets"][1].update(path="../escape.csv"),
    ),
)
def test_invalid_registry_entries_are_rejected(mutate):
    payload = _manifest()
    mutate(payload)
    with pytest.raises(DatasetRegistryError):
        validate_manifest(payload)
