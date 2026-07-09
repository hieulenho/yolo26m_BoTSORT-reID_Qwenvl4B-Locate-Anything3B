"""Build a render-ready prediction manifest for one semantic video pipeline."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PIPELINE_NAMES = {
    "A": "YOLO26m + BoT-SORT ReID + Qwen3-VL 4B",
    "B": "YOLO26m + BoT-SORT ReID + LocateAnything 3B",
    "C": "YOLO26m + BoT-SORT ReID + LocateAnything 3B + Qwen3-VL 4B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a standard track_predictions JSON for video rendering. "
            "It prefers a saved prediction manifest, can fall back to reviewed "
            "annotation CSV labels for demo rendering, and can add a selected "
            "LocateAnything query target."
        ),
    )
    parser.add_argument("--pipeline", choices=sorted(PIPELINE_NAMES), required=True)
    parser.add_argument("--sequence-name", required=True)
    parser.add_argument("--query", default="")
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preferred-manifest", type=Path)
    parser.add_argument("--annotation-csv", type=Path)
    parser.add_argument("--locate-final-resolution", type=Path)
    parser.add_argument("--qwen-answer", type=Path)
    parser.add_argument("--completion-manifest", type=Path)
    parser.add_argument("--use-annotation-labels", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"Output exists and overwrite=false: {args.output}")
    if not args.tracks.is_file():
        raise SystemExit(f"Tracks file does not exist: {args.tracks}")

    source_notes: list[str] = []
    track_predictions: list[dict[str, Any]] = []

    if args.preferred_manifest and args.preferred_manifest.is_file():
        manifest = _read_json(args.preferred_manifest)
        track_predictions = [
            dict(item)
            for item in manifest.get("track_predictions", [])
            if str(item.get("sequence_name")) == args.sequence_name
        ]
        for item in track_predictions:
            metadata = dict(item.get("metadata") or {})
            metadata.setdefault("source_type", "preferred_render_manifest")
            item["metadata"] = metadata
        source_notes.append(f"preferred_manifest:{args.preferred_manifest}")

    if not track_predictions and args.use_annotation_labels:
        if args.annotation_csv and args.annotation_csv.is_file():
            track_predictions = _predictions_from_annotation_csv(
                args.annotation_csv,
                sequence_name=args.sequence_name,
            )
            source_notes.append(f"annotation_csv:{args.annotation_csv}")
        elif args.annotation_csv:
            source_notes.append(f"annotation_csv_missing:{args.annotation_csv}")

    locate_predictions = _predictions_from_locate_resolution(
        path=args.locate_final_resolution,
        sequence_name=args.sequence_name,
        query=args.query,
    )
    if locate_predictions:
        track_predictions = _merge_by_track_id(track_predictions, locate_predictions)
        source_notes.append(f"locate_final_resolution:{args.locate_final_resolution}")

    qwen_predictions = _predictions_from_qwen_answer(
        path=args.qwen_answer,
        sequence_name=args.sequence_name,
    )
    if qwen_predictions:
        track_predictions = _merge_by_track_id(track_predictions, qwen_predictions)
        source_notes.append(f"qwen_structured_answer:{args.qwen_answer}")

    completion_predictions = _predictions_from_completion_manifest(
        path=args.completion_manifest,
        sequence_name=args.sequence_name,
    )
    if completion_predictions:
        track_predictions = _merge_missing_by_track_id(
            track_predictions,
            completion_predictions,
        )
        source_notes.append(f"coverage_completion:{args.completion_manifest}")

    track_count = _count_unique_tracks(args.tracks)
    payload = {
        "variant_id": _variant_id(args.pipeline, args.sequence_name),
        "variant_name": f"Pipeline {args.pipeline} - {PIPELINE_NAMES[args.pipeline]}",
        "benchmark_name": f"{args.sequence_name}_semantic_video_render",
        "pipeline_type": _pipeline_type(args.pipeline),
        "created_at": datetime.now(UTC).isoformat(),
        "track_predictions": sorted(
            track_predictions,
            key=lambda item: int(item.get("track_id", 0)),
        ),
        "query_predictions": [],
        "metadata": {
            "purpose": "video_rendering",
            "pipeline": args.pipeline,
            "pipeline_name": PIPELINE_NAMES[args.pipeline],
            "query": args.query,
            "tracks": str(args.tracks),
            "unique_track_count_in_mot": track_count,
            "render_label_source_notes": source_notes,
            "qwen_answer": str(args.qwen_answer) if args.qwen_answer else None,
            "warning": (
                "This file is a render manifest. Check render_label_source_notes before "
                "using it as a model-accuracy claim."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(args.output),
                "pipeline": args.pipeline,
                "track_predictions": len(track_predictions),
                "unique_track_count_in_mot": track_count,
                "label_sources": source_notes,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON file: {path}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"JSON file must contain an object: {path}")
    return data


def _predictions_from_annotation_csv(path: Path, *, sequence_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("sequence_name")) != sequence_name:
                continue
            rows.append(
                {
                    "sequence_name": sequence_name,
                    "track_id": int(float(str(row["track_id"]))),
                    "status": "resolved",
                    "team_label": str(row.get("team_label") or "unknown"),
                    "role_label": str(row.get("role_label") or "unknown"),
                    "confidence": None,
                    "evidence_frames": _evidence_frames(row),
                    "metadata": {
                        "source": str(path),
                        "source_type": "reviewed_annotation_csv",
                        "notes": row.get("notes"),
                        "contact_sheet": row.get("contact_sheet"),
                        "not_model_claim": True,
                    },
                }
            )
    return rows


def _evidence_frames(row: dict[str, str]) -> list[int]:
    try:
        start = int(float(row.get("start_frame") or 0))
        end = int(float(row.get("end_frame") or start))
    except ValueError:
        return []
    if start <= 0:
        return []
    if end <= start:
        return [start]
    mid = round((start + end) / 2)
    return sorted({start, mid, end})


def _predictions_from_locate_resolution(
    *,
    path: Path | None,
    sequence_name: str,
    query: str,
) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    final = _read_json(path)
    if str(final.get("status") or "").lower() not in {"resolved", "ok", "success"}:
        return []
    selected_ids = final.get("selected_track_ids") or []
    if not selected_ids and final.get("selected_track_id") is not None:
        selected_ids = [final["selected_track_id"]]
    if not selected_ids:
        return []
    team_label, role_label = _labels_from_query(query)
    predictions: list[dict[str, Any]] = []
    for track_id in selected_ids:
        predictions.append(
            {
                "sequence_name": sequence_name,
                "track_id": int(track_id),
                "status": "resolved",
                "team_label": team_label,
                "role_label": role_label,
                "confidence": final.get("score_margin"),
                "evidence_frames": [],
                "metadata": {
                    "source": str(path),
                    "source_type": "locateanything_query_resolution",
                    "query": query,
                    "resolution_status": final.get("status"),
                    "decision_reason": final.get("decision_reason"),
                },
            }
        )
    return predictions


def _predictions_from_qwen_answer(
    *,
    path: Path | None,
    sequence_name: str,
) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    payload = _read_json(path)
    answer = payload.get("answer")
    if isinstance(answer, dict):
        parsed = answer
    elif isinstance(answer, str):
        parsed = _parse_embedded_json(answer)
    else:
        return []
    if not parsed:
        return []

    predictions: list[dict[str, Any]] = []
    for row in parsed.get("track_predictions", []):
        if not isinstance(row, dict) or row.get("track_id") is None:
            continue
        predictions.append(
            {
                "sequence_name": sequence_name,
                "track_id": int(row["track_id"]),
                "status": "resolved",
                "team_label": str(row.get("team_label") or "unknown"),
                "role_label": str(row.get("role_label") or "unknown"),
                "confidence": row.get("confidence"),
                "evidence_frames": row.get("evidence_frames") or [],
                "metadata": {
                    "source": str(path),
                    "source_type": "qwen_structured_prediction",
                    "evidence": row.get("evidence"),
                    "not_model_claim": False,
                },
            }
        )
    return predictions


def _parse_embedded_json(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        first_newline = stripped.find("\n")
        last_fence = stripped.rfind("```")
        if first_newline >= 0 and last_fence > first_newline:
            stripped = stripped[first_newline + 1 : last_fence].strip()
    candidates = [stripped]
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        candidates.append(stripped[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _predictions_from_completion_manifest(
    *,
    path: Path | None,
    sequence_name: str,
) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    payload = _read_json(path)
    return [
        dict(item)
        for item in payload.get("track_predictions", [])
        if str(item.get("sequence_name")) == sequence_name
    ]


def _labels_from_query(query: str) -> tuple[str, str]:
    text = query.lower()
    role = "goalkeeper" if "goalkeeper" in text or "keeper" in text else "player"
    if "light blue" in text or "mci" in text or "man city" in text:
        return ("light_blue", role)
    if "dark blue" in text or "chelsea" in text or "che" in text:
        return ("dark_blue", role)
    if "yellow" in text:
        return ("yellow_kit", role)
    if "dark kit" in text or "black" in text:
        return ("dark_kit", role)
    if "green" in text:
        return ("goalkeeper_green" if role == "goalkeeper" else "green_kit", role)
    if "red" in text:
        return ("goalkeeper_red" if role == "goalkeeper" else "red_kit", role)
    if "orange" in text:
        return ("goalkeeper_orange" if role == "goalkeeper" else "orange_kit", role)
    return ("unknown", role)


def _merge_by_track_id(
    base: list[dict[str, Any]],
    override: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged = {int(item["track_id"]): dict(item) for item in base}
    for item in override:
        merged[int(item["track_id"])] = dict(item)
    return list(merged.values())


def _merge_missing_by_track_id(
    base: list[dict[str, Any]],
    fallback: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged = {int(item["track_id"]): dict(item) for item in base}
    for item in fallback:
        merged.setdefault(int(item["track_id"]), dict(item))
    return list(merged.values())


def _count_unique_tracks(path: Path) -> int:
    ids: set[int] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 2:
                continue
            try:
                ids.add(int(float(row[1])))
            except ValueError:
                continue
    return len(ids)


def _variant_id(pipeline: str, sequence_name: str) -> str:
    return f"pipeline_{pipeline.lower()}_{sequence_name}_render"


def _pipeline_type(pipeline: str) -> str:
    return {
        "A": "yolo_botsort_qwen",
        "B": "yolo_botsort_locateanything",
        "C": "yolo_botsort_locateanything_qwen",
    }[pipeline]


if __name__ == "__main__":
    main()
