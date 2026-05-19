#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "datasets" / "registry.json"
DEFAULT_DATA_ROOT = ROOT / "data" / "raw"


def load_registry(path: Path = DEFAULT_REGISTRY) -> dict[str, dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        registry = json.load(handle)
    if not isinstance(registry, dict) or not registry:
        raise ValueError("dataset registry must be a non-empty object")
    return registry


def dataset_names(path: Path = DEFAULT_REGISTRY) -> list[str]:
    return sorted(load_registry(path))


def download_dataset(
    name: str,
    *,
    registry_path: Path = DEFAULT_REGISTRY,
    data_root: Path = DEFAULT_DATA_ROOT,
    force: bool = False,
    extract: bool = False,
) -> dict[str, str]:
    registry = load_registry(registry_path)
    if name not in registry:
        raise KeyError(f"unknown dataset '{name}'. Available datasets: {', '.join(sorted(registry))}")

    entry = registry[name]
    target = _target_path(data_root, entry)
    target.parent.mkdir(parents=True, exist_ok=True)

    status = "exists"
    if force or not target.exists():
        status = "downloaded"
        _download(entry["source_url"], target)

    _verify_checksum(target, entry.get("sha256"))

    extracted_to = ""
    if extract:
        extracted_to = str(_extract_dataset(target, data_root, entry))

    return {
        "name": name,
        "status": status,
        "path": str(target),
        "extracted_to": extracted_to,
    }


def _target_path(data_root: Path, entry: dict[str, Any]) -> Path:
    local_path = Path(str(entry["local_path"]))
    if local_path.is_absolute() or ".." in local_path.parts:
        raise ValueError("dataset local_path must be a safe relative path")
    return data_root / local_path


def _download(url: str, target: Path) -> None:
    temporary = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url) as response, temporary.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    temporary.replace(target)


def _verify_checksum(path: Path, expected_sha256: str | None) -> None:
    if not expected_sha256:
        return
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"sha256 mismatch for {path}: expected {expected_sha256}, got {actual}")


def _extract_dataset(target: Path, data_root: Path, entry: dict[str, Any]) -> Path:
    if target.suffix.lower() != ".zip":
        raise ValueError("only .zip datasets can be extracted")
    extract_dir = entry.get("extract_dir")
    if not extract_dir:
        raise ValueError("dataset does not define extract_dir")
    destination = data_root / str(extract_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target) as archive:
        _safe_extract(archive, destination)
    return destination


def _safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if root != target and root not in target.parents:
            raise ValueError(f"refusing to extract unsafe zip member: {member.filename}")
    archive.extractall(destination)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download Jano example datasets locally.")
    parser.add_argument("dataset", nargs="?", help="Dataset key from datasets/registry.json.")
    parser.add_argument("--list", action="store_true", help="List available dataset keys.")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY), help="Path to registry JSON.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="Local data root.")
    parser.add_argument("--force", action="store_true", help="Re-download even if the file exists.")
    parser.add_argument("--extract", action="store_true", help="Extract zip datasets after download.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    registry_path = Path(args.registry)

    if args.list:
        for name in dataset_names(registry_path):
            print(name)
        return 0

    if not args.dataset:
        parser.error("dataset is required unless --list is used")

    result = download_dataset(
        args.dataset,
        registry_path=registry_path,
        data_root=Path(args.data_root),
        force=args.force,
        extract=args.extract,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
