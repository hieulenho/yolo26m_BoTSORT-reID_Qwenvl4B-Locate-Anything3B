"""Human-review package for cross-domain track semantic ground truth."""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from football_tracking.detection.serialization import file_sha256
from football_tracking.vlm.tracking_context import MotTrackRow, read_mot_tracks


class SemanticAnnotationError(RuntimeError):
    """Raised when an annotation package is incomplete or inconsistent."""


def audit_annotation_package(package_dir: str | Path) -> dict[str, Any]:
    """Report human-review progress without accepting model proposals as GT."""

    root = Path(package_dir).resolve()
    csv_path = root / "track_annotations.csv"
    review_path = root / "ground_truth_review.yaml"
    issues: list[str] = []
    rows: list[dict[str, str]] = []
    review: dict[str, Any] = {}

    if csv_path.is_file():
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        issues.append(f"missing track annotation CSV: {csv_path}")
    if review_path.is_file():
        loaded = yaml.safe_load(review_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            review = loaded
        else:
            issues.append("ground_truth_review.yaml must be a mapping")
    else:
        issues.append(f"missing review metadata: {review_path}")

    reviewed_rows = [
        row for row in rows if str(row.get("review_status", "")).strip().lower() == "reviewed"
    ]
    def is_ignored(row: dict[str, str]) -> bool:
        return str(row.get("ignore", "")).strip().lower() in {"1", "true", "yes"}

    ignored_rows = [row for row in rows if is_ignored(row)]
    accepted_rows = [row for row in reviewed_rows if not is_ignored(row)]
    labeled_rows = [row for row in accepted_rows if str(row.get("class_label", "")).strip()]
    attributed_rows = [row for row in reviewed_rows if str(row.get("annotator", "")).strip()]

    review_metadata = review.get("review") if isinstance(review, dict) else None
    metadata_reviewed = (
        isinstance(review_metadata, dict)
        and str(review_metadata.get("status", "")).strip().lower() == "reviewed"
        and all(
            str(review_metadata.get(key, "")).strip()
            for key in ("annotator", "reviewed_at", "method")
        )
    )
    domain_reviewed = bool(str(review.get("domain", "")).strip())
    objects_reviewed = isinstance(review.get("objects"), list) and bool(review.get("objects"))

    if reviewed_rows and len(attributed_rows) != len(reviewed_rows):
        issues.append("one or more reviewed tracks have no annotator")
    if len(labeled_rows) != len(accepted_rows):
        issues.append("one or more reviewed, non-ignored tracks have no class_label")
    if not metadata_reviewed:
        issues.append("sample review metadata is incomplete or still draft")
    if not domain_reviewed:
        issues.append("domain has not been reviewed")
    if not objects_reviewed:
        issues.append("object vocabulary has not been reviewed")

    total = len(rows)
    ready = (
        total > 0
        and len(reviewed_rows) == total
        and len(labeled_rows) == len(accepted_rows)
        and metadata_reviewed
        and domain_reviewed
        and objects_reviewed
    )
    return {
        "package_dir": str(root),
        "ready_to_finalize": ready,
        "track_count": total,
        "reviewed_track_count": len(reviewed_rows),
        "labeled_track_count": len(labeled_rows),
        "ignored_track_count": len(ignored_rows),
        "remaining_track_count": max(total - len(reviewed_rows), 0),
        "review_percent": round(100.0 * len(reviewed_rows) / total, 3) if total else 0.0,
        "metadata_reviewed": metadata_reviewed,
        "domain_reviewed": domain_reviewed,
        "objects_reviewed": objects_reviewed,
        "issues": issues,
    }


def prepare_annotation_package(
    *,
    sample_id: str,
    source_video: str | Path,
    tracks_path: str | Path,
    discovery_path: str | Path,
    route_path: str | Path,
    semantics_path: str | Path,
    run_report_path: str | Path,
    output_dir: str | Path,
    crops_per_track: int = 3,
    max_tracks: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    if crops_per_track < 1:
        raise SemanticAnnotationError("crops_per_track must be positive.")
    sample = str(sample_id).strip()
    if not sample:
        raise SemanticAnnotationError("sample_id must not be empty.")
    paths = {
        "source_video": Path(source_video).resolve(),
        "tracks": Path(tracks_path).resolve(),
        "discovery": Path(discovery_path).resolve(),
        "route": Path(route_path).resolve(),
        "semantics": Path(semantics_path).resolve(),
        "run_report": Path(run_report_path).resolve(),
    }
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.is_file()]
    if missing:
        raise SemanticAnnotationError("Missing annotation inputs: " + "; ".join(missing))
    root = Path(output_dir).resolve()
    outputs = {
        "track_csv": root / "track_annotations.csv",
        "review_yaml": root / "ground_truth_review.yaml",
        "draft_manifest": root / "manifest.draft.yaml",
        "package_json": root / "annotation_package.json",
        "readme": root / "README.md",
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not overwrite:
        raise SemanticAnnotationError(f"Annotation output exists: {existing[0]}")
    root.mkdir(parents=True, exist_ok=True)
    sheets_dir = root / "contact_sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)

    discovery = _read_json(paths["discovery"])
    route = _read_json(paths["route"])
    rows_by_track: dict[int, list[MotTrackRow]] = defaultdict(list)
    for row in read_mot_tracks(paths["tracks"]):
        rows_by_track[row.track_id].append(row)
    ranked_tracks = sorted(
        rows_by_track,
        key=lambda track_id: (-len(rows_by_track[track_id]), track_id),
    )
    if max_tracks is not None and max_tracks > 0:
        ranked_tracks = ranked_tracks[:max_tracks]
    sheets = _write_contact_sheets(
        paths["source_video"],
        rows_by_track,
        ranked_tracks,
        sheets_dir,
        crops_per_track=crops_per_track,
    )
    _write_track_csv(outputs["track_csv"], sample, rows_by_track, ranked_tracks, sheets)

    proposed_objects = [
        {
            "canonical_name": str(row.get("canonical_name", "")),
            "action": str(row.get("action", "detect")),
        }
        for row in discovery.get("objects", [])
        if isinstance(row, dict) and str(row.get("canonical_name", "")).strip()
    ]
    review = {
        "sample_id": sample,
        "domain": "",
        "detector_route": "",
        "objects": proposed_objects,
        "review": {
            "status": "draft",
            "annotator": "",
            "reviewed_at": "",
            "method": "manual_video_and_contact_sheet",
            "notes": "Review domain, complete object vocabulary, then label every CSV row.",
        },
        "proposals_not_ground_truth": {
            "domain": _domain_name(discovery),
            "detector_route": str(route.get("route_name", "")),
        },
    }
    outputs["review_yaml"].write_text(
        yaml.safe_dump(review, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    draft = _manifest_payload(sample, paths, review, tracks=[])
    draft["require_review_metadata"] = True
    outputs["draft_manifest"].write_text(
        yaml.safe_dump(draft, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    package = {
        "schema_version": 1,
        "status": "manual_review_required",
        "created_at": datetime.now(UTC).isoformat(),
        "sample_id": sample,
        "track_count": len(ranked_tracks),
        "observation_count": sum(len(rows_by_track[value]) for value in ranked_tracks),
        "inputs": {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in paths.items()
        },
        "outputs": {name: str(path) for name, path in outputs.items()},
    }
    outputs["package_json"].write_text(
        json.dumps(package, indent=2),
        encoding="utf-8",
    )
    outputs["readme"].write_text(_annotation_readme(sample), encoding="utf-8")
    return package


def finalize_annotation_package(
    *,
    package_dir: str | Path,
    output_manifest: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    root = Path(package_dir).resolve()
    review_path = root / "ground_truth_review.yaml"
    csv_path = root / "track_annotations.csv"
    package_path = root / "annotation_package.json"
    for path in (review_path, csv_path, package_path):
        if not path.is_file():
            raise SemanticAnnotationError(f"Annotation package file is missing: {path}")
    review = yaml.safe_load(review_path.read_text(encoding="utf-8"))
    if not isinstance(review, dict):
        raise SemanticAnnotationError("ground_truth_review.yaml must be a mapping.")
    _require_reviewed(review.get("review"), "ground_truth_review.review")
    domain = str(review.get("domain", "")).strip()
    objects = review.get("objects")
    if not domain:
        raise SemanticAnnotationError("ground_truth_review.domain must be assigned manually.")
    if not isinstance(objects, list) or not objects:
        raise SemanticAnnotationError("ground_truth_review.objects must be reviewed and non-empty.")
    track_labels = _read_reviewed_tracks(csv_path)
    package = _read_json(package_path)
    destination = (
        Path(output_manifest).resolve()
        if output_manifest is not None
        else root / "manifest.reviewed.yaml"
    )
    if destination.exists() and not overwrite:
        raise SemanticAnnotationError(f"Reviewed manifest already exists: {destination}")
    inputs = package.get("inputs", {})
    artifacts = {
        name: _portable_reference(Path(str(value["path"])), destination.parent)
        for name, value in inputs.items()
        if name in {"discovery", "route", "semantics", "run_report"}
        and isinstance(value, dict)
        and value.get("path")
    }
    manifest = {
        "schema_version": 1,
        "require_review_metadata": True,
        "samples": [
            {
                "sample_id": str(review.get("sample_id", package.get("sample_id", ""))),
                "artifacts": artifacts,
                "ground_truth": {
                    "domain": domain,
                    "detector_route": str(review.get("detector_route", "")).strip(),
                    "objects": objects,
                    "tracks": track_labels,
                    "review": review["review"],
                },
            }
        ],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    return {
        "status": "ok",
        "manifest": str(destination),
        "track_count": len(track_labels),
        "manifest_sha256": file_sha256(destination),
    }


def merge_reviewed_manifests(
    *,
    manifest_paths: list[str | Path],
    output_manifest: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Combine independently reviewed samples into one evaluation manifest."""
    if not manifest_paths:
        raise SemanticAnnotationError("At least one reviewed manifest is required.")
    destination = Path(output_manifest).resolve()
    if destination.exists() and not overwrite:
        raise SemanticAnnotationError(f"Merged manifest already exists: {destination}")

    samples: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    sources: list[dict[str, str]] = []
    for raw_path in manifest_paths:
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise SemanticAnnotationError(f"Reviewed manifest does not exist: {path}")
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list):
            raise SemanticAnnotationError(f"Invalid semantic manifest: {path}")
        if payload.get("require_review_metadata") is not True:
            raise SemanticAnnotationError(
                f"Reviewed manifest must require review metadata: {path}"
            )
        for index, sample in enumerate(payload["samples"]):
            if not isinstance(sample, dict):
                raise SemanticAnnotationError(f"Invalid sample {index} in {path}")
            sample_id = str(sample.get("sample_id", "")).strip()
            if not sample_id:
                raise SemanticAnnotationError(f"Sample {index} has no sample_id in {path}")
            if sample_id in sample_ids:
                raise SemanticAnnotationError(f"Duplicate sample_id: {sample_id}")
            ground_truth = sample.get("ground_truth")
            if not isinstance(ground_truth, dict):
                raise SemanticAnnotationError(f"Sample {sample_id} has no ground_truth.")
            _require_reviewed(
                ground_truth.get("review"),
                f"sample[{sample_id}].ground_truth.review",
            )
            tracks = ground_truth.get("tracks")
            if not isinstance(tracks, list) or not tracks:
                raise SemanticAnnotationError(
                    f"Sample {sample_id} has no reviewed track labels."
                )
            sample_copy = deepcopy(sample)
            artifacts = sample_copy.get("artifacts", {})
            if isinstance(artifacts, dict):
                sample_copy["artifacts"] = {
                    name: _portable_reference(
                        _resolve_reference(str(reference), path.parent),
                        destination.parent,
                    )
                    for name, reference in artifacts.items()
                }
            sample_ids.add(sample_id)
            samples.append(sample_copy)
        sources.append(
            {
                "path": _portable_reference(path, destination.parent),
                "sha256": file_sha256(path),
            }
        )

    payload = {
        "schema_version": 1,
        "require_review_metadata": True,
        "source_manifests": sources,
        "samples": samples,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    return {
        "status": "ok",
        "manifest": str(destination),
        "sample_count": len(samples),
        "track_count": sum(
            len(sample["ground_truth"]["tracks"]) for sample in samples
        ),
        "manifest_sha256": file_sha256(destination),
    }


def validate_review_metadata(value: Any, section: str) -> None:
    _require_reviewed(value, section)


def _require_reviewed(value: Any, section: str) -> None:
    if not isinstance(value, dict):
        raise SemanticAnnotationError(f"{section} must be a mapping.")
    if str(value.get("status", "")).strip().lower() != "reviewed":
        raise SemanticAnnotationError(f"{section}.status must equal 'reviewed'.")
    for key in ("annotator", "reviewed_at", "method"):
        if not str(value.get(key, "")).strip():
            raise SemanticAnnotationError(f"{section}.{key} must not be empty.")


def _write_contact_sheets(
    video_path: Path,
    rows_by_track: dict[int, list[MotTrackRow]],
    track_ids: list[int],
    output_dir: Path,
    *,
    crops_per_track: int,
) -> dict[int, Path]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SemanticAnnotationError(f"Could not open annotation video: {video_path}")
    output: dict[int, Path] = {}
    try:
        for track_id in track_ids:
            selected = _temporal_rows(rows_by_track[track_id], crops_per_track)
            tiles: list[np.ndarray] = []
            for row in selected:
                capture.set(cv2.CAP_PROP_POS_FRAMES, row.frame_index - 1)
                ok, frame = capture.read()
                if not ok:
                    continue
                crop = _crop(frame, row)
                if crop is None:
                    continue
                tile = np.full((300, 256, 3), 245, dtype=np.uint8)
                resized = _fit(crop, 256, 260)
                x = (256 - resized.shape[1]) // 2
                y = (260 - resized.shape[0]) // 2
                tile[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
                cv2.putText(
                    tile,
                    f"track {track_id} | frame {row.frame_index}",
                    (8, 286),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.46,
                    (20, 20, 20),
                    1,
                    cv2.LINE_AA,
                )
                tiles.append(tile)
            if not tiles:
                continue
            sheet = cv2.hconcat(tiles)
            path = output_dir / f"track_{track_id:06d}.jpg"
            if not cv2.imwrite(str(path), sheet):
                raise SemanticAnnotationError(f"Could not write contact sheet: {path}")
            output[track_id] = path
    finally:
        capture.release()
    return output


def _temporal_rows(rows: list[MotTrackRow], count: int) -> list[MotTrackRow]:
    ordered = sorted(rows, key=lambda row: row.frame_index)
    positions = np.linspace(0, len(ordered) - 1, min(count, len(ordered)), dtype=int)
    return [ordered[int(index)] for index in positions]


def _crop(frame: np.ndarray, row: MotTrackRow, padding: float = 0.15) -> np.ndarray | None:
    height, width = frame.shape[:2]
    x1 = max(int(row.x - row.width * padding), 0)
    y1 = max(int(row.y - row.height * padding), 0)
    x2 = min(int(row.x + row.width * (1 + padding)), width)
    y2 = min(int(row.y + row.height * (1 + padding)), height)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def _fit(image: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    scale = min(max_width / image.shape[1], max_height / image.shape[0])
    size = (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def _write_track_csv(
    path: Path,
    sample_id: str,
    rows_by_track: dict[int, list[MotTrackRow]],
    track_ids: list[int],
    sheets: dict[int, Path],
) -> None:
    fields = [
        "sample_id",
        "track_id",
        "first_frame",
        "last_frame",
        "observation_count",
        "contact_sheet",
        "class_label",
        "fine_label",
        "ignore",
        "review_status",
        "annotator",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for track_id in track_ids:
            rows = rows_by_track[track_id]
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "track_id": track_id,
                    "first_frame": min(row.frame_index for row in rows),
                    "last_frame": max(row.frame_index for row in rows),
                    "observation_count": len(rows),
                    "contact_sheet": str(sheets.get(track_id, "")),
                    "class_label": "",
                    "fine_label": "",
                    "ignore": "false",
                    "review_status": "draft",
                    "annotator": "",
                    "notes": "",
                }
            )


def _read_reviewed_tracks(path: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[int] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, row in enumerate(csv.DictReader(handle), start=2):
            track_id = int(row.get("track_id", "0"))
            if track_id < 1 or track_id in seen:
                raise SemanticAnnotationError(
                    f"Invalid/duplicate track_id at CSV line {line_number}."
                )
            seen.add(track_id)
            ignore = str(row.get("ignore", "false")).strip().lower() in {"1", "true", "yes"}
            if str(row.get("review_status", "")).strip().lower() != "reviewed":
                raise SemanticAnnotationError(f"Track {track_id} is not reviewed.")
            if not str(row.get("annotator", "")).strip():
                raise SemanticAnnotationError(f"Track {track_id} has no annotator.")
            class_label = str(row.get("class_label", "")).strip()
            if not ignore and not class_label:
                raise SemanticAnnotationError(f"Track {track_id} has no class_label.")
            value: dict[str, Any] = {
                "track_id": track_id,
                "class_label": class_label or "unknown",
            }
            fine_label = str(row.get("fine_label", "")).strip()
            if fine_label:
                value["fine_label"] = fine_label
            if ignore:
                value["ignore"] = True
            output.append(value)
    if not output:
        raise SemanticAnnotationError("No reviewed tracks were found.")
    return output


def _manifest_payload(
    sample_id: str,
    paths: dict[str, Path],
    review: dict[str, Any],
    tracks: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "samples": [
            {
                "sample_id": sample_id,
                "artifacts": {
                    name: str(paths[name])
                    for name in ("discovery", "route", "semantics", "run_report")
                },
                "ground_truth": {
                    "domain": review.get("domain", ""),
                    "detector_route": review.get("detector_route", ""),
                    "objects": review.get("objects", []),
                    "tracks": tracks,
                    "review": review.get("review", {}),
                },
            }
        ],
    }


def _domain_name(discovery: dict[str, Any]) -> str:
    value = discovery.get("domain", "")
    return str(value.get("name", "")) if isinstance(value, dict) else str(value)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SemanticAnnotationError(f"Expected a JSON object: {path}")
    return value


def _resolve_reference(value: str, base_dir: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def _portable_reference(path: Path, base_dir: Path) -> str:
    try:
        return os.path.relpath(path.resolve(), base_dir.resolve())
    except ValueError:
        return str(path.resolve())


def _annotation_readme(sample_id: str) -> str:
    return f"""# Semantic GT review: {sample_id}

1. Watch the source video and inspect every image in `contact_sheets/`.
2. Fill `class_label`, optional `fine_label`, `review_status=reviewed`, and `annotator`
   for every row in `track_annotations.csv`. Use `ignore=true` only when no human can judge it.
3. Review `domain`, `detector_route`, and the complete object list in
   `ground_truth_review.yaml`; then set its review status and provenance.
4. Run the finalize command. Draft/model proposals cannot be evaluated as GT.
"""


__all__ = [
    "SemanticAnnotationError",
    "finalize_annotation_package",
    "merge_reviewed_manifests",
    "prepare_annotation_package",
    "validate_review_metadata",
]
