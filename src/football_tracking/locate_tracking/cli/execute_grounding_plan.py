"""CLI helper for executing an event-triggered grounding plan."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from football_tracking.locate_tracking.cli.locate_image import (
    _build_backend,
    load_locate_image_config,
)
from football_tracking.locate_tracking.grounding_scheduler.executor import (
    execute_grounding_plan_from_path,
    make_cached_grounding_service,
)


def run_execute_grounding_plan(
    *,
    grounding_config_path: str | Path,
    plan_path: str | Path,
    source_video: str | Path | None,
    output_dir: str | Path,
    backend_name: str | None,
    model_id: str | None,
    device: str | None,
    torch_dtype: str | None,
    max_new_tokens: int | None,
    overwrite: bool,
) -> dict[str, Any]:
    config = load_locate_image_config(
        grounding_config_path,
        overrides={
            "backend_name": backend_name,
            "model_id": model_id,
            "device": device,
            "torch_dtype": torch_dtype,
            "max_new_tokens": max_new_tokens,
            "overwrite": overwrite,
        },
    )
    service = make_cached_grounding_service(
        backend=_build_backend(config),
        cache_dir=config.cache_directory,
        cache_enabled=config.cache_enabled,
        overwrite=config.overwrite,
    )
    manifest = execute_grounding_plan_from_path(
        plan_path=plan_path,
        source_video=source_video,
        output_dir=output_dir,
        grounding_service=service,
        overwrite=config.overwrite,
    )
    frame_count = sum(
        len(request.get("frames", ())) for request in manifest.executed_requests
    )
    return {
        "status": "ok",
        "executed_request_count": len(manifest.executed_requests),
        "executed_frame_count": frame_count,
        "skipped_request_count": len(manifest.skipped_requests),
        "paths": {
            "manifest_json": str(manifest.output_dir / "grounding_execution_manifest.json"),
            "output_dir": str(manifest.output_dir),
        },
        "note": "grounding results are saved only; no track reacquisition is performed",
    }
