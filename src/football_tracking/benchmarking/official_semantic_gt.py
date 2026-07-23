"""Audit semantic ground truth derived from official benchmark annotations."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml


class OfficialSemanticGtError(RuntimeError):
    """Raised when official semantic GT provenance is malformed."""


def audit_official_semantic_gt(
    manifests: list[str | Path],
    *,
    minimum_domains: int = 2,
    minimum_tracks: int = 20,
) -> dict[str, Any]:
    if minimum_domains < 1 or minimum_tracks < 1:
        raise OfficialSemanticGtError("Minimum domains and tracks must be positive.")
    if not manifests:
        raise OfficialSemanticGtError("At least one semantic GT manifest is required.")

    rows = [_audit_manifest(Path(value).resolve()) for value in manifests]
    domains = sorted(
        {
            domain
            for row in rows
            for domain in row["domains"]
        }
    )
    track_count = sum(int(row["track_count"]) for row in rows)
    issues = [
        issue
        for row in rows
        for issue in row["issues"]
    ]
    if len(domains) < minimum_domains:
        issues.append(
            f"official semantic GT covers {len(domains)}/{minimum_domains} domains"
        )
    if track_count < minimum_tracks:
        issues.append(
            f"official semantic GT contains {track_count}/{minimum_tracks} tracks"
        )
    return {
        "schema_version": 1,
        "status": "ready" if not issues else "blocked",
        "ground_truth_kind": "official_dataset_annotations",
        "manifest_count": len(rows),
        "domain_count": len(domains),
        "domains": domains,
        "track_count": track_count,
        "minimum_domains": minimum_domains,
        "minimum_tracks": minimum_tracks,
        "issues": issues,
        "manifests": rows,
    }


def _audit_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return _manifest_row(path, issues=["manifest does not exist"])
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return _manifest_row(path, issues=["manifest root is not a mapping"])

    source = str(payload.get("ground_truth_source", "")).strip()
    issues: list[str] = []
    if not source.lower().startswith("official"):
        issues.append("ground_truth_source is not official")
    if payload.get("require_review_metadata") is not False:
        issues.append("official manifest must set require_review_metadata: false")
    samples = payload.get("samples", [])
    if not isinstance(samples, list) or not samples:
        issues.append("manifest has no samples")
        samples = []

    domains: set[str] = set()
    track_count = 0
    for sample in samples:
        if not isinstance(sample, dict):
            issues.append("sample is not a mapping")
            continue
        ground_truth = sample.get("ground_truth", {})
        if not isinstance(ground_truth, dict):
            issues.append("sample ground_truth is not a mapping")
            continue
        domain = str(ground_truth.get("domain", "")).strip()
        if not domain:
            issues.append("sample domain is missing")
        else:
            domains.add(domain)
        tracks = ground_truth.get("tracks", [])
        if not isinstance(tracks, list) or not tracks:
            issues.append("sample contains no semantic tracks")
            continue
        for track in tracks:
            if not isinstance(track, dict):
                issues.append("semantic track is not a mapping")
                continue
            track_id = int(track.get("track_id", 0))
            label = str(track.get("class_label", "")).strip()
            if track_id < 1 or not label:
                issues.append("semantic track requires positive track_id and class_label")
                continue
            track_count += 1
    return _manifest_row(
        path,
        source=source,
        domains=sorted(domains),
        sample_count=len(samples),
        track_count=track_count,
        issues=issues,
    )


def _manifest_row(
    path: Path,
    *,
    source: str = "",
    domains: list[str] | None = None,
    sample_count: int = 0,
    track_count: int = 0,
    issues: list[str],
) -> dict[str, Any]:
    digest = (
        hashlib.sha256(path.read_bytes()).hexdigest()
        if path.is_file()
        else None
    )
    return {
        "path": str(path),
        "sha256": digest,
        "ground_truth_source": source,
        "domains": domains or [],
        "sample_count": sample_count,
        "track_count": track_count,
        "ready": not issues,
        "issues": issues,
    }


__all__ = [
    "OfficialSemanticGtError",
    "audit_official_semantic_gt",
]
