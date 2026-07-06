from __future__ import annotations

from pathlib import Path

from football_tracking.locate_tracking.grounding.backend import MockGroundingBackend
from football_tracking.locate_tracking.grounding_scheduler.executor import (
    execute_grounding_plan,
    make_cached_grounding_service,
)
from football_tracking.locate_tracking.grounding_scheduler.schemas import (
    GroundingPlan,
    GroundingPlanItem,
)
from tests.locate_tracking.appearance_test_utils import tiny_video


def _plan(source_video: Path) -> GroundingPlan:
    return GroundingPlan(
        query="player",
        source_video=source_video,
        items=(
            GroundingPlanItem(
                request_id="ground_test",
                event_id="event_test",
                event_type="target_absent",
                severity="warning",
                query="player",
                raw_track_id=7,
                selected_frames=(1, 2),
                reason="test",
            ),
        ),
    )


def test_grounding_executor_uses_mock_backend_and_cache(tmp_path: Path) -> None:
    video = tiny_video(tmp_path / "video.avi", frame_count=3)
    backend = MockGroundingBackend(default_response="<box>[10,10,20,20]</box>")
    service = make_cached_grounding_service(
        backend=backend,
        cache_dir=tmp_path / "cache",
        overwrite=True,
    )

    manifest = execute_grounding_plan(
        plan=_plan(video),
        source_video=video,
        output_dir=tmp_path / "exec",
        grounding_service=service,
        overwrite=True,
    )
    execute_grounding_plan(
        plan=_plan(video),
        source_video=video,
        output_dir=tmp_path / "exec",
        grounding_service=service,
        overwrite=True,
    )

    assert (tmp_path / "exec" / "grounding_execution_manifest.json").is_file()
    assert len(manifest.executed_requests) == 1
    assert backend.call_count == 2
