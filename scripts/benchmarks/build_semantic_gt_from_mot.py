"""Derive track-level semantic GT by matching predictions to official MOT GT."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--categories", type=Path, required=True)
    parser.add_argument("--selection-semantics", type=Path, required=True)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--route", type=Path, required=True)
    parser.add_argument("--semantics", type=Path, required=True)
    parser.add_argument("--run-report", type=Path)
    parser.add_argument("--tracking-metadata", type=Path)
    parser.add_argument("--qwen-answer", type=Path)
    parser.add_argument("--locate-result", type=Path)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--detector-route", default="")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        payload = build_semantic_gt_from_mot(
            prediction=args.prediction,
            ground_truth=args.ground_truth,
            categories=args.categories,
            selection_semantics=args.selection_semantics,
            discovery=args.discovery,
            route=args.route,
            semantics=args.semantics,
            run_report=args.run_report,
            tracking_metadata=args.tracking_metadata,
            qwen_answer=args.qwen_answer,
            locate_result=args.locate_result,
            sample_id=args.sample_id,
            domain=args.domain,
            detector_route=args.detector_route,
            iou_threshold=args.iou_threshold,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 2
    print(json.dumps(payload, indent=2))
    return 0


def build_semantic_gt_from_mot(
    *,
    prediction: Path,
    ground_truth: Path,
    categories: Path,
    selection_semantics: Path,
    discovery: Path,
    route: Path,
    semantics: Path,
    run_report: Path | None,
    tracking_metadata: Path | None,
    qwen_answer: Path | None,
    locate_result: Path | None,
    sample_id: str,
    domain: str,
    detector_route: str,
    iou_threshold: float,
    output_dir: Path,
    overwrite: bool,
) -> dict[str, Any]:
    if not 0 < iou_threshold <= 1:
        raise ValueError("IoU threshold must be in (0, 1].")
    required = [
        prediction,
        ground_truth,
        categories,
        selection_semantics,
        discovery,
        route,
        semantics,
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"Required input does not exist: {missing[0]}")
    output = output_dir.resolve()
    manifest_path = output / "semantic_gt_manifest.yaml"
    mapping_path = output / "track_gt_mapping.json"
    if (manifest_path.exists() or mapping_path.exists()) and not overwrite:
        raise ValueError(f"Output exists and overwrite=false: {output}")
    category_map = _categories(categories)
    selected_payload = _json(selection_semantics)
    selected_ids = sorted(
        {
            int(row["track_id"])
            for row in selected_payload.get("tracks", [])
            if isinstance(row, dict) and int(row.get("track_id", 0)) > 0
        }
    )
    if not selected_ids:
        raise ValueError("Selection semantics contains no track IDs.")
    predicted = _mot_by_frame(prediction)
    gt = _mot_by_frame(ground_truth)
    evidence = _match_tracks(predicted, gt, selected_ids, category_map, iou_threshold)

    gt_tracks = [
        {
            "track_id": row["predicted_track_id"],
            "class_label": row["class_label"],
            "fine_label": row["fine_label"],
            "official_gt_track_ids": row["matched_gt_track_ids"],
            "matched_observation_count": row["matched_observation_count"],
            "predicted_observation_count": row["predicted_observation_count"],
        }
        for row in evidence
    ]
    known_classes = sorted(
        {row["class_label"] for row in evidence if row["class_label"] != "unknown"}
    )
    artifacts: dict[str, str] = {
        "discovery": str(discovery.resolve()),
        "route": str(route.resolve()),
        "semantics": str(semantics.resolve()),
    }
    for key, path in (
        ("run_report", run_report),
        ("tracking_metadata", tracking_metadata),
        ("qwen_answer", qwen_answer),
        ("locate_result", locate_result),
    ):
        if path is not None and path.is_file():
            artifacts[key] = str(path.resolve())
    manifest = {
        "schema_version": 1,
        "require_review_metadata": False,
        "ground_truth_source": "official MOT boxes matched by framewise IoU",
        "samples": [
            {
                "sample_id": sample_id,
                "artifacts": artifacts,
                "ground_truth": {
                    "domain": domain,
                    "detector_route": detector_route,
                    "objects": [
                        {"canonical_name": label, "action": "track"}
                        for label in known_classes
                    ],
                    "tracks": gt_tracks,
                },
            }
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    mapping = {
        "schema_version": 1,
        "prediction": str(prediction.resolve()),
        "ground_truth": str(ground_truth.resolve()),
        "categories": str(categories.resolve()),
        "selection_semantics": str(selection_semantics.resolve()),
        "iou_threshold": iou_threshold,
        "selected_track_count": len(selected_ids),
        "known_track_count": sum(row["class_label"] != "unknown" for row in evidence),
        "unknown_track_count": sum(row["class_label"] == "unknown" for row in evidence),
        "tracks": evidence,
    }
    mapping_path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    return {
        "status": "ok",
        "manifest": str(manifest_path),
        "mapping": str(mapping_path),
        "selected_track_count": len(selected_ids),
        "known_track_count": mapping["known_track_count"],
        "unknown_track_count": mapping["unknown_track_count"],
    }


def _match_tracks(
    predicted: dict[int, list[dict[str, Any]]],
    ground_truth: dict[int, list[dict[str, Any]]],
    selected_ids: list[int],
    categories: dict[int, dict[str, str]],
    threshold: float,
) -> list[dict[str, Any]]:
    observations: Counter[int] = Counter()
    matches: defaultdict[int, list[tuple[int, int, float]]] = defaultdict(list)
    class_scores: defaultdict[int, Counter[int]] = defaultdict(Counter)
    for frame, prediction_rows in predicted.items():
        gt_rows = ground_truth.get(frame, [])
        for prediction in prediction_rows:
            track_id = int(prediction["track_id"])
            if track_id not in selected_ids:
                continue
            observations[track_id] += 1
            candidates = [(_iou(prediction, row), row) for row in gt_rows]
            if not candidates:
                continue
            iou, best = max(candidates, key=lambda item: item[0])
            if iou < threshold:
                continue
            gt_track_id = int(best["track_id"])
            class_id = int(best["class_id"])
            matches[track_id].append((gt_track_id, class_id, iou))
            class_scores[track_id][class_id] += iou
    output = []
    for track_id in selected_ids:
        scores = class_scores[track_id]
        class_id = scores.most_common(1)[0][0] if scores else None
        category = (
            categories.get(class_id, {"class_label": "unknown", "fine_label": ""})
            if class_id is not None
            else {"class_label": "unknown", "fine_label": ""}
        )
        label = category["class_label"]
        output.append(
            {
                "predicted_track_id": track_id,
                "class_id": class_id,
                "class_label": label,
                "fine_label": category["fine_label"],
                "predicted_observation_count": observations[track_id],
                "matched_observation_count": len(matches[track_id]),
                "match_fraction": round(
                    len(matches[track_id]) / max(observations[track_id], 1), 6
                ),
                "mean_matched_iou": round(
                    sum(row[2] for row in matches[track_id])
                    / max(len(matches[track_id]), 1),
                    6,
                ),
                "matched_gt_track_ids": sorted({row[0] for row in matches[track_id]}),
                "class_scores": {str(key): round(value, 6) for key, value in scores.items()},
            }
        )
    return output


def _mot_by_frame(path: Path) -> dict[int, list[dict[str, Any]]]:
    frames: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        columns = raw_line.split(",")
        if len(columns) < 6:
            raise ValueError(f"Malformed MOT row at {path}:{line_number}")
        frame = int(float(columns[0]))
        frames[frame].append(
            {
                "track_id": int(float(columns[1])),
                "x": float(columns[2]),
                "y": float(columns[3]),
                "width": float(columns[4]),
                "height": float(columns[5]),
                "class_id": int(float(columns[7])) if len(columns) > 7 else 1,
            }
        )
    return dict(frames)


def _iou(left: dict[str, Any], right: dict[str, Any]) -> float:
    x1 = max(left["x"], right["x"])
    y1 = max(left["y"], right["y"])
    x2 = min(left["x"] + left["width"], right["x"] + right["width"])
    y2 = min(left["y"] + left["height"], right["y"] + right["height"])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = left["width"] * left["height"] + right["width"] * right["height"] - intersection
    return intersection / union if union > 0 else 0.0


def _categories(path: Path) -> dict[int, dict[str, str]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Categories must be a YAML/JSON mapping.")
    categories: dict[int, dict[str, str]] = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            class_label = str(value.get("class_label", "")).strip().lower()
            fine_label = str(value.get("fine_label", "")).strip().lower()
            if not class_label:
                raise ValueError(f"Category {key} requires class_label.")
        else:
            class_label = str(value).strip().lower()
            fine_label = ""
        categories[int(key)] = {
            "class_label": class_label or "unknown",
            "fine_label": fine_label,
        }
    return categories


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be a mapping: {path}")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
