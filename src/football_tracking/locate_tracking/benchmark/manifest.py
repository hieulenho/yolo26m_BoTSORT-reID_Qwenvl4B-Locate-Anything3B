"""Load and save language benchmark and prediction manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from football_tracking.locate_tracking.benchmark.schemas import (
    LanguagePrediction,
    LanguagePredictionManifest,
    LanguageTrackingBenchmarkManifest,
    LanguageTrackingSequence,
)
from football_tracking.paths import get_project_root, resolve_project_path


class LanguageBenchmarkManifestError(RuntimeError):
    """Raised when a language benchmark manifest cannot be loaded."""


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise LanguageBenchmarkManifestError(f"Manifest does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise LanguageBenchmarkManifestError(f"Manifest root must be a JSON object: {path}")
    return payload


def _resolve_path(path: Path, project_root: Path) -> Path:
    return path if path.is_absolute() else resolve_project_path(path, project_root)


def load_benchmark_manifest(path: str | Path) -> LanguageTrackingBenchmarkManifest:
    project_root = get_project_root()
    resolved = _resolve_path(Path(path), project_root)
    manifest = LanguageTrackingBenchmarkManifest.from_dict(_load_json(resolved))
    sequences = tuple(
        _resolve_sequence_paths(sequence, project_root)
        for sequence in manifest.sequences
    )
    return manifest.with_updates(sequences=sequences)


def _resolve_sequence_paths(
    sequence: LanguageTrackingSequence,
    project_root: Path,
) -> LanguageTrackingSequence:
    return LanguageTrackingSequence(
        sequence_name=sequence.sequence_name,
        split=sequence.split,
        source_video=_resolve_path(sequence.source_video, project_root),
        mot_ground_truth_path=_resolve_path(sequence.mot_ground_truth_path, project_root),
        frame_count=sequence.frame_count,
        fps=sequence.fps,
        queries=sequence.queries,
    )


def load_prediction_manifest(path: str | Path) -> LanguagePredictionManifest:
    project_root = get_project_root()
    resolved = _resolve_path(Path(path), project_root)
    manifest = LanguagePredictionManifest.from_dict(_load_json(resolved))
    predictions = tuple(
        _resolve_prediction_paths(prediction, project_root)
        for prediction in manifest.predictions
    )
    return LanguagePredictionManifest(
        variant_id=manifest.variant_id,
        variant_name=manifest.variant_name,
        benchmark_name=manifest.benchmark_name,
        created_at=manifest.created_at,
        predictions=predictions,
        metadata=manifest.metadata,
    )


def _resolve_prediction_paths(
    prediction: LanguagePrediction,
    project_root: Path,
) -> LanguagePrediction:
    return LanguagePrediction(
        query_id=prediction.query_id,
        sequence_name=prediction.sequence_name,
        status=prediction.status,
        semantic_target_path=(
            _resolve_path(prediction.semantic_target_path, project_root)
            if prediction.semantic_target_path
            else None
        ),
        tracks_path=(
            _resolve_path(prediction.tracks_path, project_root)
            if prediction.tracks_path
            else None
        ),
        reacquisition_result_path=(
            _resolve_path(prediction.reacquisition_result_path, project_root)
            if prediction.reacquisition_result_path
            else None
        ),
        grounding_call_count=prediction.grounding_call_count,
        runtime_seconds=prediction.runtime_seconds,
        metadata=prediction.metadata,
    )


def save_json(payload: dict[str, Any], path: str | Path, *, overwrite: bool = False) -> Path:
    output = Path(path)
    if output.exists() and not overwrite:
        raise LanguageBenchmarkManifestError(f"Output exists and overwrite=false: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return output
