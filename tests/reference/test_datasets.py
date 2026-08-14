from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
from urllib.error import HTTPError, URLError

import numpy as np
import pytest

from agencitylab.models import AgencityDataset
from agencitylab.reference import datasets
from agencitylab.reference import _download


ROOT = Path(__file__).resolve().parents[2]


class FakeResponse:
    def __init__(self, payload: bytes, *, url: str, content_length: int | None = None):
        self._stream = BytesIO(payload)
        self._url = url
        length = len(payload) if content_length is None else content_length
        self.headers = {"Content-Length": str(length)}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)

    def geturl(self) -> str:
        return self._url


def _response(payload: bytes, url: str = "https://example.test/data.csv"):
    return lambda _request, timeout: FakeResponse(payload, url=url)


def test_builtin_discovery_and_offline_load(monkeypatch):
    monkeypatch.setattr(datasets, "urlopen", lambda *_a, **_k: pytest.fail("unexpected network"))

    assert datasets.available_builtin() == ("canonical_regimes_v1",)
    assert "canonical_regimes_v1" in datasets.available()
    loaded = datasets.load("canonical_regimes_v1")

    assert isinstance(loaded, AgencityDataset)
    assert len(loaded) == 3
    assert loaded.metadata.extra["dataset_version"] == "1.0.0"
    assert {signal.metadata.system_type for signal in loaded} == {
        "rest state",
        "harmonic oscillator",
        "passive damped oscillator",
    }
    assert all(np.all(np.isfinite(signal.u)) for signal in loaded)


def test_packaged_registry_and_builtin_resource_exist():
    import importlib.resources as resources

    package = resources.files("agencitylab.reference")
    assert package.joinpath("data/registry.json").is_file()
    assert package.joinpath("data/canonical_regimes_v1.csv").is_file()


def test_available_remote_is_offline_by_default(monkeypatch):
    monkeypatch.setattr(datasets, "urlopen", lambda *_a, **_k: pytest.fail("unexpected network"))
    assert datasets.available_remote() == ("lorenz_reference_v1",)


def test_remote_registry_listing_uses_explicit_network(monkeypatch):
    payload = (ROOT / "reference_datasets" / "registry.json").read_bytes()
    url = "https://raw.githubusercontent.com/Gilbert243/AgencityLab/main/reference_datasets/registry.json"
    monkeypatch.setattr(datasets, "urlopen", _response(payload, url))

    assert datasets.available_remote(refresh=True, ref="main") == ("lorenz_reference_v1",)
    records = datasets.remote_registry(ref="main")
    assert records[0].sha256 == "3e347f6373b2fa69325e1d03b49c64a0866e0149b954ba2ac2e4d75674bf5e40"


def test_official_download_verifies_checksum_and_reuses_cache(monkeypatch, tmp_path):
    payload = (ROOT / "reference_datasets" / "chaotic" / "lorenz_reference_v1.csv").read_bytes()
    url = "https://raw.githubusercontent.com/Gilbert243/AgencityLab/main/reference_datasets/chaotic/lorenz_reference_v1.csv"
    monkeypatch.setattr(_download, "urlopen", _response(payload, url))

    path = datasets.download("lorenz_reference_v1", destination=tmp_path)
    assert path.read_bytes() == payload
    provenance = json.loads(path.with_name(path.name + ".metadata.json").read_text())
    assert provenance["name"] == "lorenz_reference_v1"
    assert provenance["version"] == "1.0.0"
    assert provenance["sha256"] == datasets.info("lorenz_reference_v1")["sha256"]

    monkeypatch.setattr(_download, "urlopen", lambda *_a, **_k: pytest.fail("unexpected redownload"))
    assert datasets.download("lorenz_reference_v1", destination=tmp_path) == path
    assert datasets.is_downloaded("lorenz_reference_v1", destination=tmp_path)
    assert datasets.local_path("lorenz_reference_v1", destination=tmp_path) == path


def test_explicit_url_download_with_checksum_and_destination(monkeypatch, tmp_path):
    payload = b"xi,u\n0,1\n"
    digest = _download.hashlib.sha256(payload).hexdigest()
    url = "https://example.test/reference.csv"
    monkeypatch.setattr(_download, "urlopen", _response(payload, url))

    path = datasets.download_url(url, destination=tmp_path, expected_sha256=digest)
    assert path == tmp_path / "reference.csv"
    assert path.read_bytes() == payload


def test_checksum_mismatch_refuses_file_and_removes_partial(monkeypatch, tmp_path):
    payload = b"wrong bytes"
    url = "https://example.test/reference.csv"
    monkeypatch.setattr(_download, "urlopen", _response(payload, url))

    with pytest.raises(datasets.ChecksumMismatchError, match="SHA-256 mismatch"):
        datasets.download_url(
            url,
            destination=tmp_path,
            expected_sha256="0" * 64,
        )
    assert list(tmp_path.iterdir()) == []


def test_partial_download_is_rejected(monkeypatch, tmp_path):
    payload = b"short"
    url = "https://example.test/reference.csv"

    def partial(_request, timeout):
        return FakeResponse(payload, url=url, content_length=len(payload) + 10)

    monkeypatch.setattr(_download, "urlopen", partial)
    with pytest.raises(datasets.DatasetDownloadError, match="partial download"):
        datasets.download_url(url, destination=tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "failure",
    (
        HTTPError("https://example.test/missing.csv", 404, "not found", {}, None),
        URLError("timed out"),
    ),
)
def test_http_and_timeout_errors_are_comprehensible(monkeypatch, tmp_path, failure):
    def fail(_request, timeout):
        raise failure

    monkeypatch.setattr(_download, "urlopen", fail)
    with pytest.raises(datasets.DatasetDownloadError):
        datasets.download_url("https://example.test/missing.csv", destination=tmp_path)


@pytest.mark.parametrize("filename", ("../escape.csv", "subdir/file.csv", "..", "a\\b.csv"))
def test_external_filenames_cannot_traverse_directories(filename, tmp_path):
    with pytest.raises(ValueError, match="filename"):
        datasets.download_url(
            "https://example.test/data.csv",
            destination=tmp_path,
            filename=filename,
        )


def test_remove_deletes_only_selected_cached_dataset(monkeypatch, tmp_path):
    payload = (ROOT / "reference_datasets" / "chaotic" / "lorenz_reference_v1.csv").read_bytes()
    url = "https://raw.githubusercontent.com/Gilbert243/AgencityLab/main/reference_datasets/chaotic/lorenz_reference_v1.csv"
    monkeypatch.setattr(_download, "urlopen", _response(payload, url))
    path = datasets.download("lorenz_reference_v1", destination=tmp_path)
    unrelated = tmp_path / "keep.txt"
    unrelated.write_text("keep", encoding="utf-8")

    assert datasets.remove("lorenz_reference_v1", destination=tmp_path) is True
    assert not path.exists()
    assert unrelated.read_text(encoding="utf-8") == "keep"


def test_safe_simple_loaders_do_not_enable_pickle(tmp_path):
    array_path = tmp_path / "values.npy"
    np.save(array_path, np.arange(4.0))
    np.testing.assert_array_equal(datasets.load_path(array_path), np.arange(4.0))

    unknown = tmp_path / "payload.pkl"
    unknown.write_bytes(b"not executed")
    with pytest.raises(ValueError, match="no safe loader"):
        datasets.load_path(unknown)
