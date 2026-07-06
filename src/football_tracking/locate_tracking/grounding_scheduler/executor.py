"""Execute grounding plans with the M1 GroundingService."""

from __future__ import annotations

import json
from pathlib import Path

from football_tracking.locate_tracking.grounding.cache import GroundingCache
from football_tracking.locate_tracking.grounding.service import GroundingService
from football_tracking.locate_tracking.grounding_scheduler.planner import load_grounding_plan
from football_tracking.locate_tracking.grounding_scheduler.schemas import (
    GroundingExecutionManifest,
    GroundingPlan,
)
from football_tracking.locate_tracking.video.frame_extractor import (
    extract_video_frame,
    save_extracted_frame,
)


class GroundingExecutionError(RuntimeError):
    """Raised when event-triggered grounding execution fails."""


def execute_grounding_plan(
    *,
    plan: GroundingPlan,
    source_video: str | Path,
    output_dir: str | Path,
    grounding_service: GroundingService,
    plan_path: str | Path | None = None,
    overwrite: bool = False,
) -> GroundingExecutionManifest:
    video = Path(source_video)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    executed: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    for item in plan.items:
        request_dir = output / item.request_id
        request_dir.mkdir(parents=True, exist_ok=True)
        frame_results: list[dict[str, object]] = []
        for frame_index in item.selected_frames:
            image_path = request_dir / f"frame_{frame_index:06d}.jpg"
            result_path = request_dir / f"frame_{frame_index:06d}.grounding.json"
            if not image_path.exists() or overwrite:
                frame = extract_video_frame(video, frame_index)
                save_extracted_frame(frame, image_path)
            result = grounding_service.ground_image(
                image_path=image_path,
                query=item.query,
                output_path=result_path,
                overwrite=overwrite,
            )
            frame_results.append(
                {
                    "frame_index": frame_index,
                    "image_path": str(image_path),
                    "grounding_result_path": str(result_path),
                    "cache_hit": result.cache_hit,
                    "box_count": len(result.boxes),
                }
            )
        executed.append(
            {
                "request_id": item.request_id,
                "event_id": item.event_id,
                "event_type": item.event_type,
                "raw_track_id": item.raw_track_id,
                "frames": frame_results,
            }
        )
    manifest = GroundingExecutionManifest(
        plan_path=Path(plan_path) if plan_path is not None else Path("grounding_plan.json"),
        source_video=video,
        output_dir=output,
        executed_requests=tuple(executed),
        skipped_requests=tuple(skipped),
    )
    manifest_path = output / "grounding_execution_manifest.json"
    if manifest_path.exists() and not overwrite:
        raise GroundingExecutionError(
            f"Execution manifest exists and overwrite=false: {manifest_path}"
        )
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, default=str),
        encoding="utf-8",
    )
    return manifest


def execute_grounding_plan_from_path(
    *,
    plan_path: str | Path,
    source_video: str | Path | None,
    output_dir: str | Path,
    grounding_service: GroundingService,
    overwrite: bool = False,
) -> GroundingExecutionManifest:
    plan = load_grounding_plan(plan_path)
    video = source_video if source_video is not None else plan.source_video
    if video is None:
        raise GroundingExecutionError("source_video is required when the plan has no source_video.")
    return execute_grounding_plan(
        plan=plan,
        source_video=video,
        output_dir=output_dir,
        grounding_service=grounding_service,
        plan_path=plan_path,
        overwrite=overwrite,
    )


def make_cached_grounding_service(
    *,
    backend,
    cache_dir: str | Path,
    cache_enabled: bool = True,
    overwrite: bool = False,
) -> GroundingService:
    return GroundingService(
        backend=backend,
        cache=GroundingCache(cache_dir, enabled=cache_enabled, overwrite=overwrite),
        overwrite=overwrite,
    )
