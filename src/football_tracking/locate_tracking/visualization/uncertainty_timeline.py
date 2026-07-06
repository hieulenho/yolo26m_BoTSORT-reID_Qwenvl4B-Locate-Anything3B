"""Human-readable summaries for uncertainty monitoring outputs."""

from __future__ import annotations

from pathlib import Path

from football_tracking.locate_tracking.events.schemas import UncertaintyEvent
from football_tracking.locate_tracking.grounding_scheduler.schemas import GroundingPlan
from football_tracking.locate_tracking.monitoring.schemas import MonitoringAssessment


def write_uncertainty_summary(
    *,
    assessment: MonitoringAssessment,
    events: tuple[UncertaintyEvent, ...],
    grounding_plan: GroundingPlan,
    output_path: str | Path,
    overwrite: bool = False,
) -> Path:
    output = Path(output_path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Uncertainty summary exists and overwrite=false: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    triggered = [signal for signal in assessment.signals if signal.triggered]
    lines = [
        "# Uncertainty Monitoring Summary",
        "",
        f"- Query: `{assessment.timeline.query}`",
        f"- Current raw track ID: `{assessment.timeline.current_raw_track_id}`",
        f"- Frame range: `{assessment.timeline.start_frame}-{assessment.timeline.end_frame}`",
        f"- Aggregate severity: `{assessment.aggregate_severity}`",
        f"- Triggered signals: `{len(triggered)}`",
        f"- Events: `{len(events)}`",
        f"- Planned grounding requests: `{len(grounding_plan.items)}`",
        "",
        "## Events",
        "",
    ]
    if events:
        for event in events[:50]:
            lines.append(
                "- "
                f"`{event.event_type}` `{event.severity}` "
                f"frames `{event.frame_start}-{event.frame_end}` "
                f"track `{event.raw_track_id}`"
            )
        if len(events) > 50:
            lines.append(f"- ... {len(events) - 50} more events in `uncertainty_events.jsonl`.")
    else:
        lines.append("- No triggered uncertainty events.")
    lines.extend(["", "## Planned Grounding", ""])
    if grounding_plan.items:
        for item in grounding_plan.items:
            frames = ", ".join(str(frame) for frame in item.selected_frames)
            lines.append(
                f"- `{item.request_id}` from `{item.event_type}` on frames `{frames}`"
            )
    else:
        lines.append("- No grounding requests selected.")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output
