"""Human-readable summary for semantic target reacquisition."""

from __future__ import annotations

from pathlib import Path

from football_tracking.locate_tracking.reacquisition.schemas import ReacquisitionRun


def write_reacquisition_summary(
    *,
    run: ReacquisitionRun,
    output_path: str | Path,
    overwrite: bool = False,
) -> Path:
    output = Path(output_path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Reacquisition summary exists and overwrite=false: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    decision = run.decision
    lines = [
        "# Semantic Target Reacquisition Summary",
        "",
        f"- Semantic target: `{run.semantic_target_id}`",
        f"- Search window: `{run.search_window.start_frame}-{run.search_window.end_frame}`",
        f"- Decision: `{decision.status}`",
        f"- Previous raw track: `{decision.previous_raw_track_id}`",
        f"- Selected raw track: `{decision.selected_raw_track_id}`",
        f"- Final score: `{decision.final_score}`",
        f"- Reason: `{decision.reason}`",
        "",
        "## Candidate Ranking",
        "",
    ]
    ranked = sorted(
        (candidate for candidate in run.candidates if candidate.rank is not None),
        key=lambda item: item.rank or 9999,
    )
    if not ranked:
        lines.append("- No ranked candidate.")
    for candidate in ranked[:20]:
        lines.append(
            f"- Rank `{candidate.rank}` raw track `{candidate.raw_track_id}` "
            f"score `{candidate.final_score}` status `{candidate.status}`"
        )
        if candidate.rejection_reasons:
            lines.append(f"  Reasons: `{', '.join(candidate.rejection_reasons)}`")
    rejected = [candidate for candidate in run.candidates if candidate.rejection_reasons]
    lines.extend(["", "## Rejected Candidates", ""])
    if rejected:
        for candidate in rejected[:20]:
            lines.append(
                f"- Raw track `{candidate.raw_track_id}`: "
                f"`{', '.join(candidate.rejection_reasons)}`"
            )
    else:
        lines.append("- No hard-gate rejections.")
    lines.extend(
        [
            "",
            "## Safety",
            "",
            "- Raw MOT IDs are not modified.",
            "- M1-M5 artifacts are consumed read-only.",
            "- Appearance references remain frozen during ranking and probation.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output
