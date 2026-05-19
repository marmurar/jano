from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from scripts.download_dataset import dataset_names, download_dataset, load_registry


def test_dataset_registry_uses_safe_local_paths() -> None:
    registry = load_registry()

    assert {"bike_sharing_hourly", "bts_airline_2024_01", "nyc_tlc_yellow_2024_01"} <= set(
        registry
    )
    for entry in registry.values():
        local_path = Path(entry["local_path"])
        assert not local_path.is_absolute()
        assert ".." not in local_path.parts
        assert not str(local_path).startswith("data/")
        assert entry["source_url"].startswith("https://")


def test_download_dataset_copies_file_to_local_data_root(tmp_path) -> None:
    source = tmp_path / "source.csv"
    source.write_text("timestamp,target\n2024-01-01,1\n", encoding="utf-8")
    registry_path = _write_registry(
        tmp_path,
        {
            "tiny": {
                "source_url": source.as_uri(),
                "local_path": "tiny/source.csv",
                "sha256": None,
            }
        },
    )

    result = download_dataset("tiny", registry_path=registry_path, data_root=tmp_path / "raw")
    second = download_dataset("tiny", registry_path=registry_path, data_root=tmp_path / "raw")

    assert result["status"] == "downloaded"
    assert second["status"] == "exists"
    assert Path(result["path"]).read_text(encoding="utf-8").startswith("timestamp")


def test_download_dataset_extracts_zip_safely(tmp_path) -> None:
    source = tmp_path / "source.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("folder/example.csv", "x\n1\n")
    registry_path = _write_registry(
        tmp_path,
        {
            "zip_data": {
                "source_url": source.as_uri(),
                "local_path": "zip_data/source.zip",
                "extract_dir": "zip_data/extracted",
                "sha256": None,
            }
        },
    )

    result = download_dataset(
        "zip_data",
        registry_path=registry_path,
        data_root=tmp_path / "raw",
        extract=True,
    )

    extracted = Path(result["extracted_to"])
    assert (extracted / "folder" / "example.csv").exists()


def test_download_dataset_rejects_unknown_or_unsafe_entries(tmp_path) -> None:
    registry_path = _write_registry(
        tmp_path,
        {
            "unsafe": {
                "source_url": "file:///tmp/source.csv",
                "local_path": "../outside.csv",
                "sha256": None,
            }
        },
    )

    with pytest.raises(KeyError, match="unknown dataset"):
        download_dataset("missing", registry_path=registry_path, data_root=tmp_path / "raw")
    with pytest.raises(ValueError, match="safe relative path"):
        download_dataset("unsafe", registry_path=registry_path, data_root=tmp_path / "raw")


def test_dataset_names_are_sorted(tmp_path) -> None:
    registry_path = _write_registry(
        tmp_path,
        {
            "z": {"source_url": "file:///tmp/z", "local_path": "z.csv"},
            "a": {"source_url": "file:///tmp/a", "local_path": "a.csv"},
        },
    )

    assert dataset_names(registry_path) == ["a", "z"]


def _write_registry(tmp_path, registry) -> Path:
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(registry), encoding="utf-8")
    return path
