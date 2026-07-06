"""Load saved semantic target predictions for language-query evaluation."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from football_tracking.locate_tracking.artifacts.mot_reader import read_mot_track_file
from football_tracking.locate_tracking.artifacts.mot_schemas import MotTrackObservation
from football_tracking.locate_tracking.benchmark.manifest import load_prediction_manifest
from football_tracking.locate_tracking.benchmark.schemas import (
    LanguagePrediction,
    LanguagePredictionManifest,
)
from football_tracking.locate_tracking.identity.segment_store import load_semantic_target


class LanguagePredictionLoadError(RuntimeError):
    """Raised when prediction artifacts cannot be loaded."""


def prediction_index(
    manifest: LanguagePredictionManifest,
) -> dict[tuple[str, str], LanguagePrediction]:
    return {
        (prediction.sequence_name, prediction.query_id): prediction
        for prediction in manifest.predictions
    }


def load_prediction_manifest_index(
    path: str | Path,
) -> tuple[LanguagePredictionManifest, dict[tuple[str, str], LanguagePrediction]]:
    manifest = load_prediction_manifest(path)
    return manifest, prediction_index(manifest)


def predicted_observations_for_query(
    prediction: LanguagePrediction | None,
) -> dict[int, tuple[MotTrackObservation, ...]]:
    if prediction is None or prediction.status != "resolved":
        return {}
    if prediction.semantic_target_path is None or prediction.tracks_path is None:
        return {}
    target = load_semantic_target(prediction.semantic_target_path)
    rows = read_mot_track_file(prediction.tracks_path).observations
    by_frame: dict[int, list[MotTrackObservation]] = defaultdict(list)
    for segment in target.segments:
        if segment.status not in {"confirmed", "probation"}:
            continue
        end_frame = segment.end_frame if segment.end_frame is not None else 10**18
        for row in rows:
            if row.track_id != segment.raw_track_id:
                continue
            if segment.start_frame <= row.frame_index <= end_frame:
                by_frame[row.frame_index].append(row)
    return {frame: tuple(items) for frame, items in by_frame.items()}


def raw_id_transitions_for_prediction(prediction: LanguagePrediction | None) -> int:
    if prediction is None or prediction.semantic_target_path is None:
        return 0
    target = load_semantic_target(prediction.semantic_target_path)
    raw_ids = [
        segment.raw_track_id
        for segment in target.segments
        if segment.status in {"confirmed", "probation"}
    ]
    return sum(1 for left, right in zip(raw_ids, raw_ids[1:], strict=False) if left != right)


def load_reacquisition_payload(prediction: LanguagePrediction | None) -> dict[str, Any] | None:
    if prediction is None or prediction.reacquisition_result_path is None:
        return None
    path = prediction.reacquisition_result_path
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None

