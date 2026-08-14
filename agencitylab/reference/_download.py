"""Small, dependency-free download primitives for reference datasets."""

from __future__ import annotations

import hashlib
import re
import tempfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

_CHECKSUM = re.compile(r"^[0-9a-f]{64}$")
_FILENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")


class DatasetDownloadError(RuntimeError):
    """Raised when a requested dataset cannot be downloaded safely."""


class ChecksumMismatchError(DatasetDownloadError):
    """Raised when downloaded bytes do not match the expected SHA-256."""


def sha256_file(path: str | Path) -> str:
    """Return the lowercase SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_filename(filename: str) -> str:
    """Validate a single portable filename with no directory components."""

    if not isinstance(filename, str) or not _FILENAME.fullmatch(filename):
        raise ValueError(f"invalid download filename: {filename!r}")
    if filename in {".", ".."} or Path(filename).name != filename:
        raise ValueError(f"invalid download filename: {filename!r}")
    return filename


def validate_url(url: str) -> str:
    """Require an absolute HTTP(S) data URL without embedded credentials."""

    if not isinstance(url, str):
        raise ValueError("url must be a string")
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("url must be an absolute HTTP or HTTPS URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("url must not contain embedded credentials")
    return url


def _expected_digest(value: str | None) -> str | None:
    if value is None:
        return None
    digest = value.lower()
    if not _CHECKSUM.fullmatch(digest):
        raise ValueError("expected_sha256 must contain 64 hexadecimal digits")
    return digest


def download_file(
    url: str,
    *,
    destination: str | Path,
    filename: str,
    expected_sha256: str | None,
    timeout: float,
    force: bool,
) -> Path:
    """Download one file atomically and optionally verify its SHA-256."""

    source_url = validate_url(url)
    safe_name = validate_filename(filename)
    expected = _expected_digest(expected_sha256)
    timeout_value = float(timeout)
    if timeout_value <= 0.0:
        raise ValueError("timeout must be strictly positive")

    directory = Path(destination).expanduser()
    target = directory / safe_name
    if target.exists() and not force:
        if not target.is_file():
            raise DatasetDownloadError(f"download target is not a file: {target}")
        if expected is None or sha256_file(target) == expected:
            return target

    directory.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{safe_name}.", suffix=".part", dir=directory, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            digest = hashlib.sha256()
            received = 0
            request = Request(source_url, headers={"User-Agent": "AgencityLab-reference/1"})
            try:
                with urlopen(request, timeout=timeout_value) as response:
                    final_url = response.geturl()
                    validate_url(final_url)
                    content_length = response.headers.get("Content-Length")
                    expected_length = int(content_length) if content_length is not None else None
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        temporary.write(chunk)
                        digest.update(chunk)
                        received += len(chunk)
            except HTTPError as exc:
                raise DatasetDownloadError(
                    f"HTTP {exc.code} while downloading {source_url}"
                ) from exc
            except (URLError, OSError) as exc:
                raise DatasetDownloadError(f"failed to download {source_url}: {exc}") from exc

        if expected_length is not None and received != expected_length:
            raise DatasetDownloadError(
                f"partial download: expected {expected_length} bytes, received {received}"
            )
        actual = digest.hexdigest()
        if expected is not None and actual != expected:
            raise ChecksumMismatchError(
                f"SHA-256 mismatch for {safe_name}: expected {expected}, received {actual}"
            )
        temporary_path.replace(target)
        temporary_path = None
        return target
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


__all__ = [
    "ChecksumMismatchError",
    "DatasetDownloadError",
    "download_file",
    "sha256_file",
    "validate_filename",
    "validate_url",
]
