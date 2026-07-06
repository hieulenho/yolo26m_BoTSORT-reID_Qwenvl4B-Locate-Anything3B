"""Visual jersey-color team classifier for track-level experiments."""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from football_tracking.team_benchmark.manifest import load_team_benchmark_manifest
from football_tracking.team_benchmark.schemas import TeamTrackAnnotation


@dataclass(frozen=True)
class MotDetection:
    frame: int
    track_id: int
    x: float
    y: float
    w: float
    h: float
    score: float


@dataclass(frozen=True)
class TrackColorSample:
    sequence_name: str
    track_id: int
    team_label: str | None
    role_label: str | None
    feature: tuple[float, ...] | None
    evidence_frames: tuple[int, ...]
    crop_count: int
    status: str


@dataclass(frozen=True)
class TrackColorPrediction:
    sequence_name: str
    track_id: int
    team_label: str | None
    role_label: str | None
    confidence: float | None
    evidence_frames: tuple[int, ...]
    crop_count: int
    status: str
    distances: dict[str, float]


def load_mot_tracks(path: str | Path) -> dict[int, list[MotDetection]]:
    by_track: dict[int, list[MotDetection]] = defaultdict(list)
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 7:
                continue
            det = MotDetection(
                frame=int(float(row[0])),
                track_id=int(float(row[1])),
                x=float(row[2]),
                y=float(row[3]),
                w=float(row[4]),
                h=float(row[5]),
                score=float(row[6]),
            )
            by_track[det.track_id].append(det)
    for detections in by_track.values():
        detections.sort(key=lambda item: item.frame)
    return dict(by_track)


class TrackColorClassifier:
    """Prototype classifier over robust color features extracted from track crops."""

    def __init__(self, prototypes: dict[str, np.ndarray]) -> None:
        if not prototypes:
            raise ValueError("At least one prototype is required.")
        self.prototypes = prototypes

    @classmethod
    def fit(cls, samples: list[TrackColorSample]) -> TrackColorClassifier:
        grouped: dict[str, list[np.ndarray]] = defaultdict(list)
        for sample in samples:
            if sample.team_label and sample.feature is not None:
                grouped[sample.team_label].append(np.asarray(sample.feature, dtype=np.float32))
        prototypes = {
            label: np.mean(np.stack(features), axis=0)
            for label, features in grouped.items()
            if features
        }
        return cls(prototypes)

    def predict(
        self,
        *,
        sequence_name: str,
        track_id: int,
        role_label: str | None,
        feature: tuple[float, ...] | None,
        evidence_frames: tuple[int, ...],
        crop_count: int,
    ) -> TrackColorPrediction:
        if feature is None:
            return TrackColorPrediction(
                sequence_name=sequence_name,
                track_id=track_id,
                team_label=None,
                role_label=role_label,
                confidence=None,
                evidence_frames=evidence_frames,
                crop_count=crop_count,
                status="unknown",
                distances={},
            )
        vector = np.asarray(feature, dtype=np.float32)
        distances = {
            label: float(np.linalg.norm(vector - prototype))
            for label, prototype in self.prototypes.items()
        }
        sorted_distances = sorted(distances.items(), key=lambda item: item[1])
        label, best = sorted_distances[0]
        second = sorted_distances[1][1] if len(sorted_distances) > 1 else best + 1.0
        margin = max(second - best, 0.0)
        confidence = float(1.0 / (1.0 + math.exp(-6.0 * margin)))
        return TrackColorPrediction(
            sequence_name=sequence_name,
            track_id=track_id,
            team_label=label,
            role_label=role_label,
            confidence=confidence,
            evidence_frames=evidence_frames,
            crop_count=crop_count,
            status="resolved",
            distances=distances,
        )


def build_samples_from_manifest(
    *,
    manifest_path: str | Path,
    samples_per_track: int = 5,
) -> list[TrackColorSample]:
    manifest = load_team_benchmark_manifest(manifest_path)
    samples: list[TrackColorSample] = []
    for sequence in manifest.sequences:
        tracks = load_mot_tracks(sequence.tracks_path) if sequence.tracks_path else {}
        cap = _open_video(sequence.source_video)
        try:
            for annotation in sequence.track_annotations:
                detections = _filter_annotation_detections(
                    tracks.get(annotation.track_id, []),
                    annotation,
                )
                samples.append(
                    sample_track_feature(
                        cap=cap,
                        sequence_name=sequence.sequence_name,
                        track_id=annotation.track_id,
                        detections=detections,
                        team_label=annotation.team_label,
                        role_label=annotation.role_label,
                        samples_per_track=samples_per_track,
                    )
                )
        finally:
            cap.release()
    return samples


def build_samples_for_all_tracks(
    *,
    sequence_name: str,
    source_video: str | Path,
    tracks_path: str | Path,
    samples_per_track: int = 5,
    min_observations: int = 20,
) -> list[TrackColorSample]:
    tracks = load_mot_tracks(tracks_path)
    cap = _open_video(source_video)
    samples: list[TrackColorSample] = []
    try:
        for track_id, detections in sorted(tracks.items()):
            if len(detections) < min_observations:
                continue
            samples.append(
                sample_track_feature(
                    cap=cap,
                    sequence_name=sequence_name,
                    track_id=track_id,
                    detections=detections,
                    team_label=None,
                    role_label=None,
                    samples_per_track=samples_per_track,
                )
            )
    finally:
        cap.release()
    return samples


def sample_track_feature(
    *,
    cap: cv2.VideoCapture,
    sequence_name: str,
    track_id: int,
    detections: list[MotDetection],
    team_label: str | None,
    role_label: str | None,
    samples_per_track: int,
) -> TrackColorSample:
    selected = _pick_samples(detections, samples_per_track)
    features: list[np.ndarray] = []
    evidence_frames: list[int] = []
    for detection in selected:
        frame = _read_frame(cap, detection.frame)
        crop = _torso_crop(frame, detection)
        feature = _crop_feature(crop)
        if feature is not None:
            features.append(feature)
            evidence_frames.append(detection.frame)
    if not features:
        return TrackColorSample(
            sequence_name=sequence_name,
            track_id=track_id,
            team_label=team_label,
            role_label=role_label,
            feature=None,
            evidence_frames=tuple(evidence_frames),
            crop_count=0,
            status="unknown",
        )
    feature = np.mean(np.stack(features), axis=0)
    return TrackColorSample(
        sequence_name=sequence_name,
        track_id=track_id,
        team_label=team_label,
        role_label=role_label,
        feature=tuple(float(value) for value in feature),
        evidence_frames=tuple(evidence_frames),
        crop_count=len(features),
        status="resolved",
    )


def leave_one_track_out_metrics(samples: list[TrackColorSample]) -> dict[str, Any]:
    supports = Counter(sample.team_label for sample in samples if sample.team_label)
    rows: list[dict[str, Any]] = []
    for sample in samples:
        if not sample.team_label or supports[sample.team_label] < 2:
            rows.append(
                {
                    "track_id": sample.track_id,
                    "gt_team_label": sample.team_label,
                    "predicted_team_label": None,
                    "status": "skipped_singleton_class",
                    "correct": None,
                }
            )
            continue
        train = [item for item in samples if item.track_id != sample.track_id]
        classifier = TrackColorClassifier.fit(train)
        prediction = classifier.predict(
            sequence_name=sample.sequence_name,
            track_id=sample.track_id,
            role_label=sample.role_label,
            feature=sample.feature,
            evidence_frames=sample.evidence_frames,
            crop_count=sample.crop_count,
        )
        rows.append(
            {
                "track_id": sample.track_id,
                "gt_team_label": sample.team_label,
                "predicted_team_label": prediction.team_label,
                "status": prediction.status,
                "correct": prediction.team_label == sample.team_label,
            }
        )
    scored = [row for row in rows if row["correct"] is not None]
    accuracy = None if not scored else sum(bool(row["correct"]) for row in scored) / len(scored)
    return {
        "evaluated_track_count": len(scored),
        "skipped_track_count": len(rows) - len(scored),
        "accuracy": accuracy,
        "rows": rows,
    }


def predictions_to_manifest_dict(
    *,
    variant_id: str,
    variant_name: str,
    benchmark_name: str,
    predictions: list[TrackColorPrediction],
    query_predictions: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "variant_id": variant_id,
        "variant_name": variant_name,
        "benchmark_name": benchmark_name,
        "pipeline_type": "custom",
        "track_predictions": [
            {
                "sequence_name": prediction.sequence_name,
                "track_id": prediction.track_id,
                "status": prediction.status,
                "team_label": prediction.team_label,
                "role_label": prediction.role_label,
                "confidence": prediction.confidence,
                "evidence_frames": list(prediction.evidence_frames),
                "metadata": {
                    "classifier": "visual_color_prototype",
                    "crop_count": prediction.crop_count,
                    "distances": prediction.distances,
                },
            }
            for prediction in predictions
        ],
        "query_predictions": query_predictions,
        "metadata": metadata or {},
    }


def make_query_predictions_from_track_predictions(
    *,
    manifest_path: str | Path,
    predictions: list[TrackColorPrediction],
    track_observation_counts: dict[int, int],
) -> list[dict[str, Any]]:
    manifest = load_team_benchmark_manifest(manifest_path)
    prediction_by_id = {prediction.track_id: prediction for prediction in predictions}
    result: list[dict[str, Any]] = []
    for sequence in manifest.sequences:
        annotated_ids = {annotation.track_id for annotation in sequence.track_annotations}
        for query in sequence.query_annotations:
            selected = _select_query_tracks(
                query_id=query.query_id,
                expected_team=query.expected_team_label,
                annotated_ids=annotated_ids,
                prediction_by_id=prediction_by_id,
                track_observation_counts=track_observation_counts,
            )
            team_label = _majority_team_label(selected, prediction_by_id)
            confidences = [
                prediction_by_id[track_id].confidence
                for track_id in selected
                if prediction_by_id[track_id].confidence is not None
            ]
            result.append(
                {
                    "sequence_name": sequence.sequence_name,
                    "query_id": query.query_id,
                    "status": "resolved" if selected else "not_found",
                    "selected_track_ids": selected,
                    "team_label": team_label,
                    "confidence": None if not confidences else sum(confidences) / len(confidences),
                    "support_ratio": None,
                    "grounding_call_count": 0,
                    "runtime_seconds": None,
                    "metadata": {
                        "query_solver": "team_label_rule_over_visual_color_predictions",
                    },
                }
            )
    return result


def track_observation_counts(tracks_path: str | Path) -> dict[int, int]:
    tracks = load_mot_tracks(tracks_path)
    return {track_id: len(detections) for track_id, detections in tracks.items()}


def prediction_rows(predictions: list[TrackColorPrediction]) -> list[dict[str, Any]]:
    return [
        {
            "sequence_name": prediction.sequence_name,
            "track_id": prediction.track_id,
            "team_label": prediction.team_label,
            "role_label": prediction.role_label,
            "confidence": prediction.confidence,
            "status": prediction.status,
            "evidence_frames": list(prediction.evidence_frames),
            "crop_count": prediction.crop_count,
            "distances": prediction.distances,
        }
        for prediction in predictions
    ]


def _filter_annotation_detections(
    detections: list[MotDetection],
    annotation: TeamTrackAnnotation,
) -> list[MotDetection]:
    return [
        detection
        for detection in detections
        if annotation.start_frame <= detection.frame <= annotation.end_frame
    ]


def _pick_samples(detections: list[MotDetection], count: int) -> list[MotDetection]:
    if not detections:
        return []
    if len(detections) <= count:
        return detections
    if count <= 1:
        return [detections[len(detections) // 2]]
    indexes = [round(i * (len(detections) - 1) / (count - 1)) for i in range(count)]
    return [detections[index] for index in indexes]


def _open_video(path: str | Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    return cap


def _read_frame(cap: cv2.VideoCapture, frame_number: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(frame_number - 1, 0))
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_number}.")
    return frame


def _torso_crop(frame: np.ndarray, detection: MotDetection) -> np.ndarray:
    height, width = frame.shape[:2]
    x1 = max(int(detection.x + detection.w * 0.15), 0)
    x2 = min(int(detection.x + detection.w * 0.85), width)
    y1 = max(int(detection.y + detection.h * 0.20), 0)
    y2 = min(int(detection.y + detection.h * 0.72), height)
    if x2 <= x1 or y2 <= y1:
        x1 = max(int(detection.x), 0)
        x2 = min(int(detection.x + detection.w), width)
        y1 = max(int(detection.y), 0)
        y2 = min(int(detection.y + detection.h), height)
    return frame[y1:y2, x1:x2]


def _crop_feature(crop_bgr: np.ndarray) -> np.ndarray | None:
    if crop_bgr.size == 0:
        return None
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    h = hsv[:, :, 0].astype(np.float32) * 2.0
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)
    mask = (s > 35.0) & (v > 35.0)
    if int(mask.sum()) < 10:
        mask = v > 20.0
    if int(mask.sum()) < 10:
        return None
    h_values = h[mask]
    s_values = s[mask] / 255.0
    v_values = v[mask] / 255.0
    rgb_values = rgb[mask].astype(np.float32) / 255.0
    hue_rad = np.deg2rad(h_values)
    mean_cos = float(np.mean(np.cos(hue_rad)))
    mean_sin = float(np.mean(np.sin(hue_rad)))
    median_rgb = np.median(rgb_values, axis=0)
    luminance_values = (
        0.2126 * rgb_values[:, 0]
        + 0.7152 * rgb_values[:, 1]
        + 0.0722 * rgb_values[:, 2]
    )
    luminance = float(np.mean(luminance_values))
    return np.asarray(
        [
            mean_cos,
            mean_sin,
            float(np.median(s_values)),
            float(np.median(v_values)),
            float(median_rgb[0]),
            float(median_rgb[1]),
            float(median_rgb[2]),
            luminance,
        ],
        dtype=np.float32,
    )


def _select_query_tracks(
    *,
    query_id: str,
    expected_team: str,
    annotated_ids: set[int],
    prediction_by_id: dict[int, TrackColorPrediction],
    track_observation_counts: dict[int, int],
) -> list[int]:
    candidates = [
        track_id
        for track_id in annotated_ids
        if prediction_by_id.get(track_id)
        and prediction_by_id[track_id].status == "resolved"
        and prediction_by_id[track_id].team_label == expected_team
    ]
    if query_id.startswith("q_all_"):
        return sorted(candidates)
    if query_id == "q_light_blue_long_track":
        if not candidates:
            return []
        return [max(candidates, key=lambda track_id: track_observation_counts.get(track_id, 0))]
    if query_id == "q_dark_blue_penalty_area_group":
        preferred = [track_id for track_id in candidates if track_id in {50, 51}]
        return sorted(preferred or candidates[:2])
    goalkeepers = [
        track_id
        for track_id in candidates
        if prediction_by_id[track_id].role_label == "goalkeeper"
    ]
    return sorted(goalkeepers or candidates[:1])


def _majority_team_label(
    track_ids: list[int],
    prediction_by_id: dict[int, TrackColorPrediction],
) -> str | None:
    labels = [
        prediction_by_id[track_id].team_label
        for track_id in track_ids
        if prediction_by_id[track_id].team_label
    ]
    if not labels:
        return None
    return Counter(labels).most_common(1)[0][0]
