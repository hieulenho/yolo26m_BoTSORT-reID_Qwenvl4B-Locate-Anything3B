"""Complete render labels for every track using reviewed color prototypes."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

from football_tracking.team_benchmark.visual_color_classifier import (
    TrackColorClassifier,
    TrackColorSample,
    build_samples_for_all_tracks,
    load_mot_tracks,
)


def build_track_label_completion(
    *,
    sequence_name: str,
    source_video: str | Path,
    tracks_path: str | Path,
    annotation_csv: str | Path | None,
    samples_per_track: int = 7,
) -> dict[str, Any]:
    """Build render-only labels with explicit provenance and full ID coverage."""
    source_video = Path(source_video)
    tracks_path = Path(tracks_path)
    annotation_path = Path(annotation_csv) if annotation_csv else None
    tracks = load_mot_tracks(tracks_path)
    all_samples = build_samples_for_all_tracks(
        sequence_name=sequence_name,
        source_video=source_video,
        tracks_path=tracks_path,
        samples_per_track=samples_per_track,
        min_observations=1,
    )
    samples_by_id = {sample.track_id: sample for sample in all_samples}
    reviewed_labels = _load_reviewed_labels(annotation_path, sequence_name)
    observation_counts = {
        track_id: len(detections) for track_id, detections in tracks.items()
    }
    seed_samples = _seed_samples(
        samples_by_id,
        reviewed_labels,
        observation_counts=observation_counts,
        min_observations=20,
    )
    classifier = TrackColorClassifier.fit(seed_samples) if seed_samples else None

    predictions: list[dict[str, Any]] = []
    for track_id, detections in sorted(tracks.items()):
        predictions.append(
            _complete_one_track(
                sequence_name=sequence_name,
                track_id=track_id,
                sample=samples_by_id.get(track_id),
                reviewed=reviewed_labels.get(track_id),
                classifier=classifier,
                observation_count=len(detections),
            )
        )

    team_counts = Counter(row["team_label"] for row in predictions)
    role_counts = Counter(row["role_label"] for row in predictions)
    return {
        "variant_id": f"{sequence_name}_visual_label_completion",
        "variant_name": "Render-only visual label coverage completion",
        "benchmark_name": f"{sequence_name}_semantic_video_render",
        "pipeline_type": "visual_color_coverage_completion",
        "track_predictions": predictions,
        "query_predictions": [],
        "metadata": {
            "purpose": "render_label_coverage_completion",
            "sequence_name": sequence_name,
            "source_video": str(source_video),
            "tracks": str(tracks_path),
            "annotation_csv": str(annotation_path) if annotation_path else None,
            "samples_per_track": samples_per_track,
            "unique_track_count": len(tracks),
            "prediction_count": len(predictions),
            "reviewed_seed_count": len(seed_samples),
            "reviewed_label_count": len(reviewed_labels),
            "prototype_min_observations": 20,
            "team_label_counts": dict(sorted(team_counts.items())),
            "role_label_counts": dict(sorted(role_counts.items())),
            "not_model_benchmark_claim": True,
            "warning": (
                "These labels guarantee readable render coverage. They are calibrated "
                "from reviewed color prototypes and must not be reported as Qwen or "
                "LocateAnything benchmark predictions."
            ),
        },
    }


def _load_reviewed_labels(
    path: Path | None,
    sequence_name: str,
) -> dict[int, dict[str, str]]:
    if path is None or not path.is_file():
        return {}
    labels: dict[int, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("sequence_name")) != sequence_name:
                continue
            track_id = int(float(str(row["track_id"])))
            labels[track_id] = {
                "team_label": str(row.get("team_label") or "unknown"),
                "role_label": str(row.get("role_label") or "unknown"),
            }
    return labels


def _seed_samples(
    samples_by_id: dict[int, TrackColorSample],
    reviewed_labels: dict[int, dict[str, str]],
    *,
    observation_counts: dict[int, int],
    min_observations: int,
) -> list[TrackColorSample]:
    seeds: list[TrackColorSample] = []
    for track_id, labels in reviewed_labels.items():
        if observation_counts.get(track_id, 0) < min_observations:
            continue
        sample = samples_by_id.get(track_id)
        if sample is None or sample.feature is None:
            continue
        seeds.append(
            replace(
                sample,
                team_label=labels["team_label"],
                role_label=labels["role_label"],
            )
        )
    return seeds


def _complete_one_track(
    *,
    sequence_name: str,
    track_id: int,
    sample: TrackColorSample | None,
    reviewed: dict[str, str] | None,
    classifier: TrackColorClassifier | None,
    observation_count: int,
) -> dict[str, Any]:
    if reviewed is not None:
        return {
            "sequence_name": sequence_name,
            "track_id": track_id,
            "status": "resolved",
            "team_label": reviewed["team_label"],
            "role_label": reviewed["role_label"],
            "confidence": 1.0,
            "evidence_frames": list(sample.evidence_frames) if sample else [],
            "metadata": {
                "source_type": "reviewed_label_coverage_seed",
                "observation_count": observation_count,
                "not_model_claim": True,
            },
        }
    if observation_count < 6:
        return _unknown_prediction(
            sequence_name=sequence_name,
            track_id=track_id,
            observation_count=observation_count,
            reason="too_few_observations_for_reliable_visual_label",
        )
    if sample is None or sample.feature is None or classifier is None:
        return _unknown_prediction(
            sequence_name=sequence_name,
            track_id=track_id,
            observation_count=observation_count,
            reason="no_usable_visual_feature_or_color_prototype",
        )

    predicted = classifier.predict(
        sequence_name=sequence_name,
        track_id=track_id,
        role_label=None,
        feature=sample.feature,
        evidence_frames=sample.evidence_frames,
        crop_count=sample.crop_count,
    )
    team_label = str(predicted.team_label or "unknown")
    return {
        "sequence_name": sequence_name,
        "track_id": track_id,
        "status": "resolved",
        "team_label": team_label,
        "role_label": _role_for_team(team_label),
        "confidence": predicted.confidence,
        "evidence_frames": list(predicted.evidence_frames),
        "metadata": {
            "source_type": "visual_color_coverage_completion",
            "classifier": "reviewed_color_prototype",
            "crop_count": predicted.crop_count,
            "observation_count": observation_count,
            "distances": predicted.distances,
            "reviewed_seed_track": False,
            "not_model_claim": True,
        },
    }


def _unknown_prediction(
    *,
    sequence_name: str,
    track_id: int,
    observation_count: int,
    reason: str,
) -> dict[str, Any]:
    return {
        "sequence_name": sequence_name,
        "track_id": track_id,
        "status": "resolved",
        "team_label": "unknown",
        "role_label": "unknown",
        "confidence": None,
        "evidence_frames": [],
        "metadata": {
            "source_type": "coverage_unknown_fallback",
            "observation_count": observation_count,
            "reason": reason,
            "not_model_claim": True,
        },
    }


def _role_for_team(team_label: str) -> str:
    normalized = team_label.lower()
    if normalized in {"referee", "referee_black", "official", "official_black"}:
        return "referee"
    if normalized.startswith("goalkeeper"):
        return "goalkeeper"
    if normalized == "unknown":
        return "unknown"
    return "player"
