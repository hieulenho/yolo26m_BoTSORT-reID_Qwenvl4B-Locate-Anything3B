"""Registry and readiness checks for official multi-domain benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from football_tracking.benchmarking.semantic_annotation import audit_annotation_package
from football_tracking.paths import get_project_root


class DatasetRegistryError(RuntimeError):
    """Raised when a benchmark source registry is invalid."""


def audit_dataset_registry(registry_path: str | Path) -> dict[str, Any]:
    """Validate a source registry and report local benchmark readiness."""

    path = Path(registry_path).resolve()
    if not path.is_file():
        raise DatasetRegistryError(f"Dataset registry does not exist: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise DatasetRegistryError("Dataset registry must use schema_version: 1.")
    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        raise DatasetRegistryError("Dataset registry must contain non-empty sources.")

    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            raise DatasetRegistryError(f"sources[{index}] must be a mapping.")
        source_id = str(source.get("id", "")).strip()
        if not source_id or source_id in seen:
            raise DatasetRegistryError(f"Missing or duplicated source id: {source_id!r}")
        seen.add(source_id)
        for key in ("domain", "benchmark_scope", "access", "annotation_format"):
            if not str(source.get(key, "")).strip():
                raise DatasetRegistryError(f"Source '{source_id}' is missing '{key}'.")
        requirements = source.get("local_requirements", [])
        if not isinstance(requirements, list) or not requirements:
            raise DatasetRegistryError(
                f"Source '{source_id}' must define local_requirements."
            )
        checked = [_resolve_local_path(value, path.parent) for value in requirements]
        missing = [str(candidate) for candidate in checked if not candidate.exists()]
        access = str(source["access"])
        release_required = source.get("release_required", True)
        if not isinstance(release_required, bool):
            raise DatasetRegistryError(
                f"Source '{source_id}' release_required must be true or false."
            )
        review_audit = None
        if access == "local_human_review" and not missing:
            review_audit = audit_annotation_package(checked[0])
        ready = not missing and (
            review_audit is None or bool(review_audit["ready_to_finalize"])
        )
        rows.append(
            {
                "id": source_id,
                "domain": str(source["domain"]),
                "benchmark_scope": str(source["benchmark_scope"]),
                "annotation_format": str(source["annotation_format"]),
                "access": access,
                "release_required": release_required,
                "ready": ready,
                "status": (
                    "ready"
                    if ready
                    else (
                        "human_review_required"
                        if review_audit is not None
                        else _missing_status(access)
                    )
                ),
                "official_url": source.get("official_url"),
                "download_url": source.get("download_url"),
                "usage_note": str(source.get("usage_note", "")),
                "metrics": list(source.get("metrics", [])),
                "local_requirements": [str(candidate) for candidate in checked],
                "missing": missing,
                "review_audit": review_audit,
            }
        )
    required_rows = [row for row in rows if row["release_required"]]
    optional_rows = [row for row in rows if not row["release_required"]]
    return {
        "registry": str(path),
        "source_count": len(rows),
        "ready_count": sum(row["ready"] for row in rows),
        "blocked_count": sum(not row["ready"] for row in rows),
        "required_source_count": len(required_rows),
        "required_ready_count": sum(row["ready"] for row in required_rows),
        "required_blocked_count": sum(not row["ready"] for row in required_rows),
        "optional_source_count": len(optional_rows),
        "optional_ready_count": sum(row["ready"] for row in optional_rows),
        "sources": rows,
    }


def _resolve_local_path(value: Any, registry_dir: Path) -> Path:
    text = str(value).strip()
    if not text:
        raise DatasetRegistryError("local_requirements entries must not be empty.")
    candidate = Path(text)
    if candidate.is_absolute():
        return candidate.resolve()
    del registry_dir
    return (get_project_root() / candidate).resolve()


def _missing_status(access: str) -> str:
    return {
        "account_required": "download_requires_account",
        "permission_sensitive": "permission_and_download_required",
        "manual_download": "manual_download_required",
        "local_human_review": "human_review_required",
    }.get(access, "local_data_missing")
