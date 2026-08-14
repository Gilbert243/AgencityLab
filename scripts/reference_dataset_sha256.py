"""Print a file's SHA-256 and size for a reference-dataset registry entry."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=Path)
    args = parser.parse_args()
    if not args.file.is_file():
        parser.error(f"not a file: {args.file}")
    print(f"sha256={sha256(args.file)}")
    print(f"size={args.file.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
