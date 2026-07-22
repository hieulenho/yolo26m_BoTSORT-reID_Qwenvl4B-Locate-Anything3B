"""Create editable real-video language benchmark and prediction manifests."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from football_tracking.locate_tracking.benchmark.manifest import save_json
from football_tracking.locate_tracking.identity.segment_store import (
    save_semantic_target,
    save_semantic_target_timeline,
)
from football_tracking.locate_tracking.identity.semantic_target import (
    create_initial_semantic_target,
)


class LanguageBenchmarkTemplateError(RuntimeError):
    """Raised when a language benchmark template cannot be created."""


def create_language_benchmark_template(
    *,
    output_dir: str | Path,
    sequence_name: str,
    source_video: str | Path,
    tracks: str | Path,
    ground_truth: str | Path,
    frame_count: int,
    query_id: str,
    query: str,
    target_gt_track_id: int,
    evaluation_start_frame: int,
    evaluation_end_frame: int,
    fps: float | None = None,
    benchmark_name: str = "language_tracking_subset",
    benchmark_version: str = "0.1.0",
    dataset_name: str = "custom_video",
    split: str = "dev",
    query_mode: str = "single_target",
    query_category: str = "role",
    difficulty: str = "medium",
    variant_id: str = "a5_full_system",
    variant_name: str = "A5 full system",
    semantic_target: str | Path | None = None,
    semantic_target_id: str | None = None,
    raw_track_id: int | None = None,
    raw_start_frame: int | None = None,
    raw_end_frame: int | None = None,
    last_confirmed_frame: int | None = None,
    reacquisition_result: str | Path | None = None,
    loss_frame: int | None = None,
    reacquisition_start_frame: int | None = None,
    reacquisition_end_frame: int | None = None,
    gt_reappearance_frame: int | None = None,
    grounding_call_count: int = 0,
    runtime_seconds: float | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    out = Path(output_dir)
    if not sequence_name.strip():
        raise LanguageBenchmarkTemplateError("sequence_name must not be empty.")
    if not query_id.strip() or not query.strip():
        raise LanguageBenchmarkTemplateError("query_id and query must not be empty.")
    if frame_count < 1:
        raise LanguageBenchmarkTemplateError("frame_count must be >= 1.")
    if evaluation_end_frame < evaluation_start_frame:
        raise LanguageBenchmarkTemplateError("evaluation_end_frame must be >= start.")
    if evaluation_end_frame > frame_count:
        raise LanguageBenchmarkTemplateError("evaluation_end_frame exceeds frame_count.")

    source_video_path = Path(source_video)
    tracks_path = Path(tracks)
    ground_truth_path = Path(ground_truth)
    semantic_target_path = Path(semantic_target) if semantic_target else None
    created_semantic_target = False

    if semantic_target_path is None:
        if raw_track_id is None:
            raise LanguageBenchmarkTemplateError(
                "Pass --semantic-target, or pass --raw-track-id so a starter "
                "semantic_target.json can be created."
            )
        raw_start = raw_start_frame or evaluation_start_frame
        raw_end = raw_end_frame if raw_end_frame is not None else evaluation_end_frame
        semantic_target_path = out / "artifacts" / "semantic_target.json"
        target = create_initial_semantic_target(
            query=query,
            raw_track_id=raw_track_id,
            start_frame=raw_start,
            end_frame=raw_end,
            semantic_target_id=semantic_target_id,
        )
        if last_confirmed_frame is not None:
            target = target.with_updates(
                last_confirmed_frame=last_confirmed_frame,
                last_update_frame=last_confirmed_frame,
            )
        save_semantic_target(target, semantic_target_path, overwrite=overwrite)
        save_semantic_target_timeline(
            target,
            out / "artifacts" / "semantic_target_timeline.json",
            overwrite=overwrite,
        )
        created_semantic_target = True

    loss_events, reacquisition_events = _events(
        loss_frame=loss_frame,
        reacquisition_start_frame=reacquisition_start_frame,
        reacquisition_end_frame=reacquisition_end_frame,
        gt_reappearance_frame=gt_reappearance_frame,
    )
    benchmark_manifest = {
        "benchmark_name": benchmark_name,
        "benchmark_version": benchmark_version,
        "dataset_name": dataset_name,
        "split": split,
        "annotation_policy": (
            "Real-video language benchmark scaffold. Review GT IDs, frame ranges, "
            "and events before reporting results."
        ),
        "sequences": [
            {
                "sequence_name": sequence_name,
                "split": split,
                "source_video": str(source_video_path),
                "mot_ground_truth_path": str(ground_truth_path),
                "frame_count": frame_count,
                "fps": fps,
                "queries": [
                    {
                        "query_id": query_id,
                        "query_text": query,
                        "query_mode": query_mode,
                        "query_category": query_category,
                        "difficulty": difficulty,
                        "evaluation_start_frame": evaluation_start_frame,
                        "evaluation_end_frame": evaluation_end_frame,
                        "target_gt_track_ids": [target_gt_track_id],
                        "identity_segments": [
                            {
                                "gt_track_id": target_gt_track_id,
                                "start_frame": evaluation_start_frame,
                                "end_frame": evaluation_end_frame,
                                "visibility_notes": "TODO: verify manually.",
                            }
                        ],
                        "loss_events": loss_events,
                        "reacquisition_events": reacquisition_events,
                        "notes": "TODO: review annotation before using as benchmark.",
                    }
                ],
            }
        ],
        "metadata": {
            "generated_by": "create-language-benchmark-template",
            "requires_manual_review": True,
        },
    }

    prediction_manifest = {
        "variant_id": variant_id,
        "variant_name": variant_name,
        "benchmark_name": benchmark_name,
        "predictions": [
            {
                "sequence_name": sequence_name,
                "query_id": query_id,
                "status": "resolved",
                "semantic_target_path": str(semantic_target_path),
                "tracks_path": str(tracks_path),
                "reacquisition_result_path": (
                    None if reacquisition_result is None else str(reacquisition_result)
                ),
                "grounding_call_count": grounding_call_count,
                "runtime_seconds": runtime_seconds,
                "metadata": {
                    "created_semantic_target_template": created_semantic_target,
                    "manual_review_required": True,
                },
            }
        ],
        "metadata": {
            "uses_same_raw_mot_base": True,
            "generated_by": "create-language-benchmark-template",
        },
    }

    benchmark_path = save_json(
        benchmark_manifest,
        out / "benchmark_manifest.json",
        overwrite=overwrite,
    )
    prediction_path = save_json(
        prediction_manifest,
        out / f"predictions_{_slug(variant_id)}.json",
        overwrite=overwrite,
    )
    readme_path = _write_readme(
        output_dir=out,
        benchmark_path=benchmark_path,
        prediction_path=prediction_path,
        overwrite=overwrite,
    )

    missing = [
        str(path)
        for path in (
            source_video_path,
            tracks_path,
            ground_truth_path,
            semantic_target_path,
            Path(reacquisition_result) if reacquisition_result else None,
        )
        if path is not None and not Path(path).is_file()
    ]
    return {
        "status": "ok",
        "manual_review_required": True,
        "missing_inputs": missing,
        "paths": {
            "benchmark_manifest": str(benchmark_path),
            "prediction_manifest": str(prediction_path),
            "semantic_target": str(semantic_target_path),
            "readme": str(readme_path),
        },
        "next_commands": {
            "validate": (
                ".\\.venv\\Scripts\\python.exe -m "
                "football_tracking.locate_tracking.cli validate-language-benchmark "
                f"--manifest {benchmark_path}"
            ),
            "evaluate": (
                ".\\.venv\\Scripts\\python.exe -m "
                "football_tracking.locate_tracking.cli run-language-benchmark "
                f"--manifest {benchmark_path} "
                f"--predictions {prediction_path} "
                "--output-dir outputs\\locate_tracking\\benchmark\\subset\\a5_full_system "
                "--overwrite"
            ),
        },
    }


def _events(
    *,
    loss_frame: int | None,
    reacquisition_start_frame: int | None,
    reacquisition_end_frame: int | None,
    gt_reappearance_frame: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    values = (
        loss_frame,
        reacquisition_start_frame,
        reacquisition_end_frame,
        gt_reappearance_frame,
    )
    if all(value is None for value in values):
        return [], []
    if any(value is None for value in values):
        raise LanguageBenchmarkTemplateError(
            "Reacquisition annotation needs all of --loss-frame, "
            "--reacquisition-start-frame, --reacquisition-end-frame, "
            "and --gt-reappearance-frame."
        )
    assert loss_frame is not None
    assert reacquisition_start_frame is not None
    assert reacquisition_end_frame is not None
    assert gt_reappearance_frame is not None
    return (
        [
            {
                "event_id": "loss_001",
                "frame_start": loss_frame,
                "frame_end": loss_frame,
                "reason": "TODO: describe visual loss or raw-ID fragmentation.",
            }
        ],
        [
            {
                "event_id": "reaq_001",
                "loss_event_id": "loss_001",
                "target_lost_frame": loss_frame,
                "candidate_search_start": reacquisition_start_frame,
                "candidate_search_end": reacquisition_end_frame,
                "gt_reappearance_frame": gt_reappearance_frame,
                "evaluation_start_frame": reacquisition_start_frame,
                "evaluation_end_frame": reacquisition_end_frame,
                "notes": "TODO: verify manually.",
            }
        ],
    )


def _write_readme(
    *,
    output_dir: Path,
    benchmark_path: Path,
    prediction_path: Path,
    overwrite: bool,
) -> Path:
    path = output_dir / "README.md"
    if path.exists() and not overwrite:
        raise LanguageBenchmarkTemplateError(f"Output exists and overwrite=false: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# Language Benchmark Template",
                "",
                "Review these files before reporting numbers:",
                "",
                f"- `{benchmark_path}`: GT annotation manifest.",
                f"- `{prediction_path}`: prediction artifact manifest.",
                "",
                "Checklist:",
                "",
                "1. Confirm `target_gt_track_ids` uses dataset GT IDs, not predicted IDs.",
                "2. Confirm `identity_segments` frame ranges match visible GT target boxes.",
                "3. Confirm prediction `semantic_target_path` points to runtime output only.",
                "4. Run validation before benchmark evaluation.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_\\-]+", "_", value.strip())
    return slug or "variant"
