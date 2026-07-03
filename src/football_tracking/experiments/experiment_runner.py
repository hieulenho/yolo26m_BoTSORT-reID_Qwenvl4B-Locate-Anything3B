"""Run SORT and DeepSORT from a shared detection cache."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from football_tracking.detection.cache import create_detection_cache
from football_tracking.detection.cache_reader import DetectionCacheReader
from football_tracking.detection.cache_validation import validate_cache_for_sources
from football_tracking.detection.cache_writer import sequence_cache_dir
from football_tracking.detection.serialization import file_sha256
from football_tracking.evaluation.experiment_metrics import write_overall_metrics
from football_tracking.evaluation.multi_tracker_trackeval import evaluate_trackers_with_trackeval
from football_tracking.experiments.experiment_config import (
    CompareTrackersConfig,
    load_compare_trackers_config,
)
from football_tracking.experiments.schemas import (
    TRACKING_METRIC_NAMES,
    ExperimentDefinition,
    ExperimentResult,
    TrackerSpec,
)
from football_tracking.reporting.tracker_comparison_report import write_tracker_comparison_report
from football_tracking.tracking.cached_detection_source import CachedDetectionSource
from football_tracking.tracking.mot_writer import MotPredictionWriter
from football_tracking.tracking.sequence_runner import (
    SequenceSource,
    discover_mot_sequences,
    iter_source_frames,
)
from football_tracking.tracking.tracker_factory import (
    create_tracker,
    load_tracker_runtime_config,
)
from football_tracking.tracking.validation import (
    TrackValidationReport,
    validate_mot_prediction_file,
    write_track_validation_report,
)
from football_tracking.visualization.tracker_comparison import write_tracker_comparison_figures


class ExperimentRunnerError(RuntimeError):
    """Raised when an experiment cannot run safely."""


def _discover_sources(config: CompareTrackersConfig) -> list[SequenceSource]:
    return discover_mot_sequences(
        config.mot_root,
        config.split,
        config.seqmap,
        max_sequences=config.max_sequences,
    )


def _stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def _tracker_spec(config: CompareTrackersConfig, tracker_name: str) -> TrackerSpec:
    for spec in config.trackers:
        if spec.name == tracker_name:
            return spec
    raise ExperimentRunnerError(f"Tracker is not configured: {tracker_name}")


def _cache_hashes(
    config: CompareTrackersConfig,
    sources: list[SequenceSource],
) -> dict[str, str | None]:
    hashes: dict[str, str | None] = {}
    for source in sources:
        reader = DetectionCacheReader.for_sequence(
            config.detection_cache_root,
            config.split,
            source.name,
        )
        try:
            hashes[source.name] = reader.read_metadata().file_hash
        except Exception:  # noqa: BLE001
            hashes[source.name] = None
    return hashes


def _experiment_definition(
    config: CompareTrackersConfig,
    spec: TrackerSpec,
    sources: list[SequenceSource],
) -> ExperimentDefinition:
    payload = {
        "experiment_name": config.experiment_name,
        "tracker": spec.name,
        "tracker_config_hash": file_sha256(spec.config),
        "cache_hashes": _cache_hashes(config, sources),
        "confidence_threshold": config.confidence_threshold,
        "split": config.split,
        "sequences": [source.name for source in sources],
        "max_frames": config.max_frames_per_sequence,
        "smoke_only": config.smoke_only,
    }
    experiment_id = _stable_hash(payload)
    return ExperimentDefinition(
        experiment_id=experiment_id,
        experiment_name=config.experiment_name,
        tracker_name=spec.name,
        tracker_config=spec.config,
        tracker_config_hash=file_sha256(spec.config),
        detector_cache_root=config.detection_cache_root,
        detection_cache_hashes=payload["cache_hashes"],
        confidence_threshold=config.confidence_threshold,
        split=config.split,
        sequences=payload["sequences"],
        partial=config.max_frames_per_sequence is not None or config.allow_partial_sequences,
        smoke_only=config.smoke_only,
        output_directory=config.output_root / spec.name,
    )


def _cache_missing(config: CompareTrackersConfig, sources: list[SequenceSource]) -> bool:
    for source in sources:
        cache_dir = sequence_cache_dir(config.detection_cache_root, config.split, source.name)
        detections_path = cache_dir / "detections.jsonl"
        metadata_path = cache_dir / "metadata.json"
        if not detections_path.is_file() or not metadata_path.is_file():
            return True
    return False


def _ensure_detection_cache(
    config: CompareTrackersConfig,
    sources: list[SequenceSource],
) -> dict[str, Any]:
    cache_result: dict[str, Any] = {"created": False}
    if _cache_missing(config, sources):
        cache_result = create_detection_cache(
            config.detection_cache_config,
            overrides={
                "max_sequences": config.max_sequences,
                "max_frames": config.max_frames_per_sequence,
                "overwrite": True,
            },
            dry_run=False,
        )
        cache_result["created"] = True
    report = validate_cache_for_sources(
        config.detection_cache_root,
        config.split,
        sources,
        confidence_threshold=config.confidence_threshold,
        max_frames_per_sequence=config.max_frames_per_sequence,
        allow_partial_sequences=config.allow_partial_sequences,
    )
    if report.has_errors:
        raise ExperimentRunnerError(
            "Detection cache validation failed for experiment compatibility. "
            "See per-sequence validation.json files or run validate-detection-cache."
        )
    cache_result["validation"] = report.to_dict()
    return cache_result


def _dry_run_plan(
    config: CompareTrackersConfig,
    sources: list[SequenceSource],
    tracker_name: str | None = None,
) -> dict[str, Any]:
    trackers = [tracker_name] if tracker_name else [spec.name for spec in config.trackers]
    return {
        "dry_run": True,
        "experiment": config.experiment_name,
        "split": config.split,
        "cache_root": str(config.detection_cache_root),
        "confidence_threshold": config.confidence_threshold,
        "trackers": trackers,
        "sequence_count": len(sources),
        "sequences": [
            {
                "name": source.name,
                "frame_count": source.frame_count,
                "cache_dir": str(
                    sequence_cache_dir(config.detection_cache_root, config.split, source.name)
                ),
            }
            for source in sources
        ],
        "action": "validated experiment config and sequence list; no tracker or TrackEval run",
    }


def run_tracker_from_cache(
    experiment_config: str | Path,
    tracker_name: str,
    overrides: dict[str, Any] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    config = load_compare_trackers_config(experiment_config, overrides=overrides)
    sources = _discover_sources(config)
    if dry_run:
        return _dry_run_plan(config, sources, tracker_name=tracker_name)
    _ensure_detection_cache(config, sources)
    spec = _tracker_spec(config, tracker_name.lower())
    result, definition, validation = _run_tracker(config, spec, sources)
    trackeval = evaluate_trackers_with_trackeval(
        [spec.name],
        gt_root=config.mot_root,
        trackers_root=config.tracks_root,
        split=config.split,
        seqmap=config.seqmap,
        output_root=config.metrics_root / "trackeval",
        metrics=config.metrics,
        allow_partial_sequences=config.allow_partial_sequences,
    )
    result.metrics.update(trackeval[spec.name].metrics)
    result.metadata["trackeval_per_sequence"] = trackeval[spec.name].per_sequence
    if trackeval[spec.name].reason:
        result.warnings.append(trackeval[spec.name].reason)
    _write_tracker_artifacts(config, definition, result, validation)
    return {
        "definition": definition.to_dict(),
        "result": result.to_dict(),
        "validation": validation.to_dict(),
        "trackeval": {spec.name: trackeval[spec.name].to_dict()},
    }


def compare_trackers(
    config_path: str | Path,
    overrides: dict[str, Any] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    config = load_compare_trackers_config(config_path, overrides=overrides)
    sources = _discover_sources(config)
    if dry_run:
        return _dry_run_plan(config, sources)
    cache_result = _ensure_detection_cache(config, sources)
    results: list[ExperimentResult] = []
    definitions: list[ExperimentDefinition] = []
    validations: dict[str, TrackValidationReport] = {}
    for spec in config.trackers:
        try:
            result, definition, validation = _run_tracker(config, spec, sources)
            results.append(result)
            definitions.append(definition)
            validations[spec.name] = validation
        except Exception as exc:  # noqa: BLE001
            if config.fail_fast:
                raise
            results.append(_failed_result(config, spec, str(exc)))

    tracker_names = [result.tracker_name for result in results]
    trackeval = evaluate_trackers_with_trackeval(
        tracker_names,
        gt_root=config.mot_root,
        trackers_root=config.tracks_root,
        split=config.split,
        seqmap=config.seqmap,
        output_root=config.metrics_root / "trackeval",
        metrics=config.metrics,
        allow_partial_sequences=config.allow_partial_sequences,
    )
    for result in results:
        eval_result = trackeval.get(result.tracker_name)
        if eval_result is None:
            continue
        result.metrics.update(eval_result.metrics)
        result.metadata["trackeval_per_sequence"] = eval_result.per_sequence
        if eval_result.reason:
            result.warnings.append(eval_result.reason)

    artifact_paths = _write_comparison_outputs(config, results, definitions, validations, trackeval)
    return {
        "experiment": config.experiment_name,
        "smoke_only": config.smoke_only,
        "partial_sequences": config.max_frames_per_sequence is not None
        or config.allow_partial_sequences,
        "cache": cache_result,
        "results": [result.to_dict() for result in results],
        "trackeval": {name: value.to_dict() for name, value in trackeval.items()},
        "paths": artifact_paths,
    }


def evaluate_tracking_outputs(
    config_path: str | Path,
    overrides: dict[str, Any] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Evaluate existing MOT tracker outputs without rerunning trackers."""
    config = load_compare_trackers_config(config_path, overrides=overrides)
    sources = _discover_sources(config)
    tracker_names = [spec.name for spec in config.trackers]
    plan = [
        {
            "tracker": tracker_name,
            "sequence": source.name,
            "mot_path": str(
                config.tracks_root / tracker_name / config.split / f"{source.name}.txt"
            ),
        }
        for tracker_name in tracker_names
        for source in sources
    ]
    if dry_run:
        return {
            "dry_run": True,
            "split": config.split,
            "trackers": tracker_names,
            "sequence_count": len(sources),
            "planned_files": plan,
            "action": "validate existing MOT files and run official TrackEval",
        }

    validation_paths: dict[str, str] = {}
    validation_reports: dict[str, TrackValidationReport] = {}
    for tracker_name in tracker_names:
        report = TrackValidationReport()
        for source in sources:
            mot_path = config.tracks_root / tracker_name / config.split / f"{source.name}.txt"
            metadata_path = (
                config.tracks_root / tracker_name / config.split / f"{source.name}.metadata.json"
            )
            report.extend(
                validate_mot_prediction_file(mot_path, source.seqinfo_path, metadata_path)
            )
        validation_reports[tracker_name] = report
        validation_path = write_track_validation_report(
            report,
            config.metrics_root / f"{tracker_name}_mot_validation.json",
        )
        validation_paths[tracker_name] = str(validation_path)

    validation_failed = any(report.has_errors for report in validation_reports.values())
    if validation_failed:
        return {
            "dry_run": False,
            "status": "validation_failed",
            "validation": {name: report.to_dict() for name, report in validation_reports.items()},
            "paths": {"validation": validation_paths},
        }

    trackeval = evaluate_trackers_with_trackeval(
        tracker_names,
        gt_root=config.mot_root,
        trackers_root=config.tracks_root,
        split=config.split,
        seqmap=config.seqmap,
        output_root=config.metrics_root / "trackeval",
        metrics=config.metrics,
        allow_partial_sequences=config.allow_partial_sequences,
    )
    overall_rows = [
        _evaluation_overall_row(config, tracker_name, sources, trackeval[tracker_name])
        for tracker_name in tracker_names
    ]
    overall_csv = config.metrics_root / "tracking_evaluation_overall.csv"
    overall_json = config.metrics_root / "tracking_evaluation_overall.json"
    write_overall_metrics(overall_rows, overall_csv, overall_json)
    per_sequence_csv = config.metrics_root / "tracking_evaluation_per_sequence.csv"
    _write_evaluation_per_sequence_csv(config, sources, trackeval, per_sequence_csv)
    figures = write_tracker_comparison_figures(
        overall_rows,
        config.figures_root / "tracking_evaluation",
    )
    summary_path = config.metrics_root / "tracking_evaluation.json"
    summary = {
        "dry_run": False,
        "status": "completed",
        "split": config.split,
        "trackers": tracker_names,
        "sequence_count": len(sources),
        "validation": {name: report.to_dict() for name, report in validation_reports.items()},
        "trackeval": {name: value.to_dict() for name, value in trackeval.items()},
        "paths": {
            "overall_csv": str(overall_csv),
            "overall_json": str(overall_json),
            "per_sequence_csv": str(per_sequence_csv),
            "summary": str(summary_path),
            "validation": validation_paths,
            "figures": figures,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def _failed_result(
    config: CompareTrackersConfig,
    spec: TrackerSpec,
    message: str,
) -> ExperimentResult:
    return ExperimentResult(
        experiment_id=_stable_hash({"experiment": config.experiment_name, "tracker": spec.name}),
        tracker_name=spec.name,
        status="failed",
        sequence_count=0,
        frame_count=0,
        detection_count=0,
        emitted_track_count=0,
        unique_track_count=0,
        tracker_seconds=0.0,
        frame_read_seconds=0.0,
        cache_read_seconds=0.0,
        mot_write_seconds=0.0,
        total_seconds=0.0,
        tracker_fps=None,
        cached_pipeline_fps=None,
        metrics={name: None for name in TRACKING_METRIC_NAMES},
        errors=[message],
    )


def _run_tracker(
    config: CompareTrackersConfig,
    spec: TrackerSpec,
    sources: list[SequenceSource],
) -> tuple[ExperimentResult, ExperimentDefinition, TrackValidationReport]:
    definition = _experiment_definition(config, spec, sources)
    adapter = create_tracker(spec.name, spec.config, device="auto")
    sequence_summaries: list[dict[str, Any]] = []
    validation = TrackValidationReport()
    total_started = time.perf_counter()
    totals = {
        "frames": 0,
        "detections": 0,
        "tracks": 0,
        "tracker_seconds": 0.0,
        "frame_read_seconds": 0.0,
        "cache_read_seconds": 0.0,
        "mot_write_seconds": 0.0,
    }
    unique_tracks: set[tuple[str, int]] = set()
    for source in sources:
        summary, sequence_validation = _run_sequence_from_cache(config, spec, source, adapter)
        sequence_summaries.append(summary)
        validation.extend(sequence_validation)
        totals["frames"] += summary["frame_count"]
        totals["detections"] += summary["detection_count"]
        totals["tracks"] += summary["emitted_track_count"]
        totals["tracker_seconds"] += summary["tracker_seconds"]
        totals["frame_read_seconds"] += summary["frame_read_seconds"]
        totals["cache_read_seconds"] += summary["cache_read_seconds"]
        totals["mot_write_seconds"] += summary["mot_write_seconds"]
        unique_tracks.update((source.name, track_id) for track_id in summary["track_ids"])
    total_seconds = time.perf_counter() - total_started
    tracker_fps = (
        totals["frames"] / totals["tracker_seconds"]
        if totals["frames"] > 0 and totals["tracker_seconds"] > 0
        else None
    )
    cached_pipeline_seconds = (
        totals["frame_read_seconds"]
        + totals["cache_read_seconds"]
        + totals["tracker_seconds"]
        + totals["mot_write_seconds"]
    )
    cached_pipeline_fps = (
        totals["frames"] / cached_pipeline_seconds
        if totals["frames"] > 0 and cached_pipeline_seconds > 0
        else None
    )
    result = ExperimentResult(
        experiment_id=definition.experiment_id,
        tracker_name=spec.name,
        status="completed" if not validation.has_errors else "failed",
        sequence_count=len(sequence_summaries),
        frame_count=int(totals["frames"]),
        detection_count=int(totals["detections"]),
        emitted_track_count=int(totals["tracks"]),
        unique_track_count=len(unique_tracks),
        tracker_seconds=float(totals["tracker_seconds"]),
        frame_read_seconds=float(totals["frame_read_seconds"]),
        cache_read_seconds=float(totals["cache_read_seconds"]),
        mot_write_seconds=float(totals["mot_write_seconds"]),
        total_seconds=total_seconds,
        tracker_fps=tracker_fps,
        cached_pipeline_fps=cached_pipeline_fps,
        metrics={name: None for name in TRACKING_METRIC_NAMES},
        warnings=[],
        errors=[] if not validation.has_errors else ["MOT validation failed."],
        metadata={
            "definition": definition.to_dict(),
            "tracker_config": load_tracker_runtime_config(spec.name, spec.config, device="auto"),
            "sequences": sequence_summaries,
            "detection_source": "cache",
            "created_at": datetime.now(UTC).isoformat(),
        },
    )
    return result, definition, validation


def _run_sequence_from_cache(
    config: CompareTrackersConfig,
    spec: TrackerSpec,
    source: SequenceSource,
    adapter: Any,
) -> tuple[dict[str, Any], TrackValidationReport]:
    adapter.reset()
    cache_dir = sequence_cache_dir(config.detection_cache_root, config.split, source.name)
    detection_source = CachedDetectionSource(cache_dir, config.confidence_threshold)
    mot_path = config.tracks_root / spec.name / config.split / f"{source.name}.txt"
    metadata_path = config.tracks_root / spec.name / config.split / f"{source.name}.metadata.json"
    if mot_path.exists() and not config.overwrite:
        raise ExperimentRunnerError(f"MOT output exists and overwrite=false: {mot_path}")
    writer = MotPredictionWriter(mot_path, metadata_path)

    frame_count = 0
    detection_count = 0
    emitted_track_count = 0
    track_ids: set[int] = set()
    tracker_seconds = 0.0
    frame_read_seconds = 0.0
    cache_read_seconds = 0.0
    sequence_started = time.perf_counter()
    iterator = iter_source_frames(source, max_frames=config.max_frames_per_sequence)
    while True:
        frame_started = time.perf_counter()
        try:
            frame_item = next(iterator)
        except StopIteration:
            break
        frame_read_seconds += time.perf_counter() - frame_started
        frame = frame_item.image
        height, width = frame.shape[:2]

        cache_started = time.perf_counter()
        detections = detection_source.tracker_detections(frame_item.frame_index)
        cache_read_seconds += time.perf_counter() - cache_started

        tracker_started = time.perf_counter()
        tracks = adapter.update(
            frame_item.frame_index,
            source.name,
            detections,
            frame,
            width,
            height,
        )
        tracker_seconds += time.perf_counter() - tracker_started

        writer.add_tracks(tracks)
        frame_count += 1
        detection_count += len(detections)
        emitted_track_count += len(tracks)
        track_ids.update(track.track_id for track in tracks)

    mot_started = time.perf_counter()
    writer.write()
    metadata = {
        "sequence": source.name,
        "tracker": spec.name,
        "tracker_config": str(spec.config),
        "detection_source": "cache",
        "cache_dir": str(cache_dir),
        "confidence_threshold": config.confidence_threshold,
        "frame_count": frame_count,
        "detection_count": detection_count,
        "emitted_track_count": emitted_track_count,
        "unique_track_count": len(track_ids),
        "tracker_seconds": tracker_seconds,
        "frame_read_seconds": frame_read_seconds,
        "cache_read_seconds": cache_read_seconds,
        "total_seconds": time.perf_counter() - sequence_started,
        "smoke_only": config.smoke_only,
        "partial_sequence": config.max_frames_per_sequence is not None,
    }
    writer.write_metadata(metadata)
    mot_write_seconds = time.perf_counter() - mot_started
    validation = validate_mot_prediction_file(mot_path, source.seqinfo_path, metadata_path)
    return {
        **metadata,
        "mot_path": str(mot_path),
        "metadata_path": str(metadata_path),
        "mot_write_seconds": mot_write_seconds,
        "track_ids": sorted(track_ids),
    }, validation


def _write_tracker_artifacts(
    config: CompareTrackersConfig,
    definition: ExperimentDefinition,
    result: ExperimentResult,
    validation: TrackValidationReport,
) -> None:
    tracker_dir = config.output_root / definition.tracker_name
    tracker_dir.mkdir(parents=True, exist_ok=True)
    (tracker_dir / "definition.json").write_text(
        json.dumps(definition.to_dict(), indent=2, default=str),
        encoding="utf-8",
    )
    (tracker_dir / "result.json").write_text(
        json.dumps(result.to_dict(), indent=2, default=str),
        encoding="utf-8",
    )
    write_track_validation_report(
        validation,
        config.metrics_root / f"{definition.tracker_name}_mot_validation.json",
    )


def _write_comparison_outputs(
    config: CompareTrackersConfig,
    results: list[ExperimentResult],
    definitions: list[ExperimentDefinition],
    validations: dict[str, TrackValidationReport],
    trackeval: dict[str, Any],
) -> dict[str, str]:
    for definition, result in zip(definitions, results, strict=False):
        _write_tracker_artifacts(
            config,
            definition,
            result,
            validations.get(result.tracker_name, TrackValidationReport()),
        )
    overall_rows = [_overall_row(config, result) for result in results]
    overall_csv = config.metrics_root / "sort_vs_deepsort_overall.csv"
    overall_json = config.metrics_root / "sort_vs_deepsort_overall.json"
    write_overall_metrics(overall_rows, overall_csv, overall_json)
    per_sequence_csv = config.metrics_root / "sort_vs_deepsort_per_sequence.csv"
    _write_per_sequence_csv(results, per_sequence_csv)
    delta_json = config.metrics_root / "sort_vs_deepsort_delta.json"
    delta_json.write_text(
        json.dumps(_comparison_delta(results), indent=2, default=str),
        encoding="utf-8",
    )
    figures_dir = config.figures_root / "sort_vs_deepsort"
    figure_paths = write_tracker_comparison_figures(overall_rows, figures_dir)
    report_path = write_tracker_comparison_report(
        config=config,
        results=results,
        overall_rows=overall_rows,
        delta=_comparison_delta(results),
        trackeval={name: value.to_dict() for name, value in trackeval.items()},
        figures=figure_paths,
    )
    return {
        "overall_csv": str(overall_csv),
        "overall_json": str(overall_json),
        "per_sequence_csv": str(per_sequence_csv),
        "delta_json": str(delta_json),
        "report": str(report_path),
        "figures_dir": str(figures_dir),
    }


def _overall_row(config: CompareTrackersConfig, result: ExperimentResult) -> dict[str, Any]:
    definition = result.metadata.get("definition", {})
    return {
        "tracker": result.tracker_name,
        "confidence_threshold": config.confidence_threshold,
        "tracker_config_hash": definition.get("tracker_config_hash"),
        "sequence_count": result.sequence_count,
        "frame_count": result.frame_count,
        **{name: result.metrics.get(name) for name in TRACKING_METRIC_NAMES},
        "tracker_fps": result.tracker_fps,
        "cached_pipeline_fps": result.cached_pipeline_fps,
        "unique_predicted_ids": result.unique_track_count,
        "smoke_only": config.smoke_only,
        "partial_sequences": config.max_frames_per_sequence is not None
        or config.allow_partial_sequences,
    }


def _write_per_sequence_csv(results: list[ExperimentResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "tracker",
        "sequence",
        "frame_count",
        "detection_count",
        "emitted_track_count",
        "unique_track_count",
        "tracker_seconds",
        "tracker_fps",
        "mot_path",
        *TRACKING_METRIC_NAMES,
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            for sequence in result.metadata.get("sequences", []):
                frames = sequence.get("frame_count", 0)
                seconds = sequence.get("tracker_seconds", 0.0)
                writer.writerow(
                    {
                        "tracker": result.tracker_name,
                        "sequence": sequence.get("sequence"),
                        "frame_count": frames,
                        "detection_count": sequence.get("detection_count"),
                        "emitted_track_count": sequence.get("emitted_track_count"),
                        "unique_track_count": sequence.get("unique_track_count"),
                        "tracker_seconds": seconds,
                        "tracker_fps": frames / seconds if frames and seconds else None,
                        "mot_path": sequence.get("mot_path"),
                        **{
                            name: result.metadata.get("trackeval_per_sequence", {})
                            .get(sequence.get("sequence"), {})
                            .get(name)
                            for name in TRACKING_METRIC_NAMES
                        },
                    }
                )


def _evaluation_overall_row(
    config: CompareTrackersConfig,
    tracker_name: str,
    sources: list[SequenceSource],
    trackeval_result: Any,
) -> dict[str, Any]:
    stats = _existing_tracker_stats(config, tracker_name, sources)
    return {
        "tracker": tracker_name,
        "confidence_threshold": config.confidence_threshold,
        "tracker_config_hash": file_sha256(_tracker_spec(config, tracker_name).config),
        "sequence_count": len(sources),
        "frame_count": stats["frame_count"],
        **{name: trackeval_result.metrics.get(name) for name in TRACKING_METRIC_NAMES},
        "tracker_fps": stats["tracker_fps"],
        "cached_pipeline_fps": stats["cached_pipeline_fps"],
        "unique_predicted_ids": stats["unique_predicted_ids"],
        "smoke_only": config.smoke_only,
        "partial_sequences": config.max_frames_per_sequence is not None
        or config.allow_partial_sequences,
    }


def _existing_tracker_stats(
    config: CompareTrackersConfig,
    tracker_name: str,
    sources: list[SequenceSource],
) -> dict[str, Any]:
    totals = {
        "frame_count": 0,
        "unique_predicted_ids": 0,
        "tracker_seconds": 0.0,
        "cached_pipeline_seconds": 0.0,
    }
    for source in sources:
        metadata = _read_sequence_metadata(config, tracker_name, source.name)
        frame_count = int(metadata.get("frame_count", source.frame_count))
        tracker_seconds = float(metadata.get("tracker_seconds", 0.0) or 0.0)
        cached_pipeline_seconds = sum(
            float(metadata.get(key, 0.0) or 0.0)
            for key in (
                "frame_read_seconds",
                "cache_read_seconds",
                "tracker_seconds",
                "mot_write_seconds",
            )
        )
        totals["frame_count"] += frame_count
        totals["unique_predicted_ids"] += int(metadata.get("unique_track_count", 0) or 0)
        totals["tracker_seconds"] += tracker_seconds
        totals["cached_pipeline_seconds"] += cached_pipeline_seconds
    tracker_fps = (
        totals["frame_count"] / totals["tracker_seconds"]
        if totals["frame_count"] and totals["tracker_seconds"]
        else None
    )
    cached_pipeline_fps = (
        totals["frame_count"] / totals["cached_pipeline_seconds"]
        if totals["frame_count"] and totals["cached_pipeline_seconds"]
        else None
    )
    return {
        "frame_count": totals["frame_count"],
        "unique_predicted_ids": totals["unique_predicted_ids"],
        "tracker_fps": tracker_fps,
        "cached_pipeline_fps": cached_pipeline_fps,
    }


def _read_sequence_metadata(
    config: CompareTrackersConfig,
    tracker_name: str,
    sequence_name: str,
) -> dict[str, Any]:
    metadata_path = (
        config.tracks_root / tracker_name / config.split / f"{sequence_name}.metadata.json"
    )
    if not metadata_path.is_file():
        return {}
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_evaluation_per_sequence_csv(
    config: CompareTrackersConfig,
    sources: list[SequenceSource],
    trackeval: dict[str, Any],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "tracker",
        "sequence",
        "frame_count",
        "tracker_fps",
        "mot_path",
        *TRACKING_METRIC_NAMES,
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for tracker_name, result in trackeval.items():
            for source in sources:
                metadata = _read_sequence_metadata(config, tracker_name, source.name)
                frames = int(metadata.get("frame_count", source.frame_count))
                seconds = float(metadata.get("tracker_seconds", 0.0) or 0.0)
                metrics = result.per_sequence.get(source.name, {})
                writer.writerow(
                    {
                        "tracker": tracker_name,
                        "sequence": source.name,
                        "frame_count": frames,
                        "tracker_fps": frames / seconds if frames and seconds else None,
                        "mot_path": str(
                            config.tracks_root / tracker_name / config.split / f"{source.name}.txt"
                        ),
                        **{name: metrics.get(name) for name in TRACKING_METRIC_NAMES},
                    }
                )


def _comparison_delta(results: list[ExperimentResult]) -> dict[str, Any]:
    by_name = {result.tracker_name: result for result in results}
    sort = by_name.get("sort")
    deepsort = by_name.get("deepsort")
    if sort is None or deepsort is None:
        return {"available": False, "reason": "Both sort and deepsort results are required."}

    def metric_delta(name: str) -> float | int | None:
        left = sort.metrics.get(name)
        right = deepsort.metrics.get(name)
        if left is None or right is None:
            return None
        return right - left

    fps_delta = None
    if sort.tracker_fps is not None and deepsort.tracker_fps is not None:
        fps_delta = deepsort.tracker_fps - sort.tracker_fps
    return {
        "available": True,
        "deep_sort_minus_sort": {
            "HOTA": metric_delta("HOTA"),
            "IDF1": metric_delta("IDF1"),
            "AssA": metric_delta("AssA"),
            "IDSW": metric_delta("IDSW"),
            "tracker_fps": fps_delta,
        },
    }


def summarize_experiments(root: str | Path) -> dict[str, Any]:
    root_path = Path(root)
    results: list[dict[str, Any]] = []
    for path in sorted(root_path.rglob("result.json")):
        try:
            results.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return {
        "root": str(root_path),
        "result_count": len(results),
        "results": results,
    }
