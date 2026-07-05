"""Markdown report for shared-cache tracker comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from football_tracking.experiments.experiment_config import CompareTrackersConfig
from football_tracking.experiments.schemas import ExperimentResult


def _fmt(value: Any) -> str:
    if value is None:
        return "not available"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_tracker_comparison_report(
    config: CompareTrackersConfig,
    results: list[ExperimentResult],
    overall_rows: list[dict[str, Any]],
    delta: dict[str, Any],
    trackeval: dict[str, Any],
    figures: list[str],
) -> Path:
    path = config.metrics_root / "tracker_comparison_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    tracker_names = [row.get("tracker") for row in overall_rows]
    title = " vs ".join(str(name) for name in tracker_names) if tracker_names else "Trackers"
    lines = [
        f"# {title} Tracker Comparison",
        "",
        "## Goal",
        "Compare all configured trackers using the same cached detector outputs.",
        "",
        "## Dataset",
        f"- MOT root: `{config.mot_root}`",
        f"- Split: `{config.split}`",
        f"- Seqmap: `{config.seqmap}`",
        f"- Smoke only: `{config.smoke_only}`",
        f"- Partial sequences: `{config.max_frames_per_sequence is not None}`",
        "",
        "## Fair Comparison Protocol",
        f"- Detection cache root: `{config.detection_cache_root}`",
        f"- Confidence threshold: `{config.confidence_threshold}`",
        "- Every tracker reads the same per-frame cache files.",
        "- The detector is not run separately per tracker during comparison.",
        "- Ground truth and cached detections are not modified by the runner.",
        "",
        "## Results",
        "| Tracker | Frames | Detections | Tracks | HOTA | DetA | AssA | IDF1 | "
        "IDSW | Tracker FPS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overall_rows:
        result = _result_for(row.get("tracker"), results)
        lines.append(
            "| "
            f"{row.get('tracker')} | "
            f"{row.get('frame_count')} | "
            f"{result.detection_count} | "
            f"{result.emitted_track_count} | "
            f"{_fmt(row.get('HOTA'))} | "
            f"{_fmt(row.get('DetA'))} | "
            f"{_fmt(row.get('AssA'))} | "
            f"{_fmt(row.get('IDF1'))} | "
            f"{_fmt(row.get('IDSW'))} | "
            f"{_fmt(row.get('tracker_fps'))} |"
        )
    lines.extend(
        [
            "",
            "## Delta",
            f"`delta`: `{delta}`",
            "",
            "## TrackEval",
        ]
    )
    for tracker_name, payload in trackeval.items():
        lines.extend(
            [
                f"- `{tracker_name}` available: `{payload.get('available')}`",
                f"  reason: {_fmt(payload.get('reason'))}",
                f"  raw: `{payload.get('raw_output_path')}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "- DetA mainly reflects detector quality and output policy when both trackers "
            "share detections.",
            "- AssA, IDF1, and IDSW are more directly tied to association behavior.",
            "- Appearance-based trackers usually trade speed for stronger identity association.",
            "- Smoke results are plumbing checks, not official project results.",
            "",
            "## Figures",
        ]
    )
    if figures:
        lines.extend(f"- `{figure}`" for figure in figures)
    else:
        lines.append("- No figures were generated because official metrics were unavailable.")
    lines.extend(
        [
            "",
            "## Limitations",
            "- No test split result is reported unless a test command is explicitly run.",
            "- No metric is synthesized when official TrackEval output is unavailable.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _result_for(tracker_name: Any, results: list[ExperimentResult]) -> ExperimentResult:
    for result in results:
        if result.tracker_name == tracker_name:
            return result
    if not results:
        raise ValueError("No experiment results are available.")
    return results[0]
