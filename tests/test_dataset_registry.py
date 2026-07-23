from pathlib import Path

import pytest
import yaml

from football_tracking.benchmarking.dataset_registry import (
    DatasetRegistryError,
    audit_dataset_registry,
)


def test_dataset_registry_reports_ready_and_missing(tmp_path: Path) -> None:
    ready = tmp_path / "ready"
    ready.mkdir()
    registry = tmp_path / "sources.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "sources": [
                    {
                        "id": "ready",
                        "domain": "traffic",
                        "benchmark_scope": "mot",
                        "access": "local",
                        "annotation_format": "mot",
                        "local_requirements": [str(ready)],
                    },
                    {
                        "id": "portal",
                        "release_required": False,
                        "domain": "wildlife",
                        "benchmark_scope": "mot",
                        "access": "account_required",
                        "annotation_format": "json",
                        "local_requirements": [str(tmp_path / "missing")],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    result = audit_dataset_registry(registry)

    assert result["ready_count"] == 1
    assert result["blocked_count"] == 1
    assert result["required_source_count"] == 1
    assert result["required_ready_count"] == 1
    assert result["optional_source_count"] == 1
    assert result["sources"][0]["status"] == "ready"
    assert result["sources"][1]["status"] == "download_requires_account"


def test_dataset_registry_rejects_non_boolean_release_requirement(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "sources.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "sources": [
                    {
                        "id": "invalid",
                        "release_required": "yes",
                        "domain": "traffic",
                        "benchmark_scope": "mot",
                        "access": "local",
                        "annotation_format": "mot",
                        "local_requirements": [str(tmp_path)],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(DatasetRegistryError, match="release_required"):
        audit_dataset_registry(registry)


def test_dataset_registry_rejects_duplicate_ids(tmp_path: Path) -> None:
    source = {
        "id": "same",
        "domain": "traffic",
        "benchmark_scope": "mot",
        "access": "local",
        "annotation_format": "mot",
        "local_requirements": [str(tmp_path)],
    }
    registry = tmp_path / "sources.yaml"
    registry.write_text(
        yaml.safe_dump({"schema_version": 1, "sources": [source, source]}),
        encoding="utf-8",
    )

    with pytest.raises(DatasetRegistryError, match="duplicated"):
        audit_dataset_registry(registry)
