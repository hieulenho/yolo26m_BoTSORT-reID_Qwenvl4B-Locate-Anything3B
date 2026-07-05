"""Final Markdown report generator for the tracking project."""

from __future__ import annotations

import csv
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from football_tracking.paths import get_project_root, resolve_project_path


class FinalReportError(RuntimeError):
    """Raised when the final report cannot be generated."""


@dataclass(frozen=True)
class FinalReportConfig:
    project_root: Path
    config_path: Path
    title: str
    detector_metrics: Path | None
    tracker_overall: Path | None
    benchmark_markdown: Path | None
    dataset_audit: Path | None
    figures_root: Path | None
    markdown_path: Path
    pdf_path: Path | None
    make_pdf: bool


def _mapping(value: Any, section: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FinalReportError(f"{section} must be a mapping.")
    return value


def _resolve_optional_path(value: Any, project_root: Path, section: str) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise FinalReportError(f"{section} must be a non-empty path string when set.")
    path = Path(value)
    return path.resolve() if path.is_absolute() else resolve_project_path(path, project_root)


def _resolve_required_path(value: Any, project_root: Path, section: str) -> Path:
    path = _resolve_optional_path(value, project_root, section)
    if path is None:
        raise FinalReportError(f"{section} is required.")
    return path


def load_final_report_config(
    config_path: str | Path,
    output_override: str | Path | None = None,
) -> FinalReportConfig:
    project_root = get_project_root()
    path = Path(config_path)
    resolved = path.resolve() if path.is_absolute() else resolve_project_path(path, project_root)
    if not resolved.is_file():
        raise FinalReportError(f"Report config does not exist: {resolved}")
    raw = _mapping(yaml.safe_load(resolved.read_text(encoding="utf-8")), "report config root")
    report = _mapping(raw.get("report", {}), "report")
    inputs = _mapping(raw.get("inputs", {}), "inputs")
    output = _mapping(raw.get("output", {}), "output")
    runtime = _mapping(raw.get("runtime", {}), "runtime")
    markdown_path = _resolve_required_path(
        output_override or output.get("markdown"),
        project_root,
        "output.markdown",
    )
    return FinalReportConfig(
        project_root=project_root,
        config_path=resolved,
        title=str(report.get("title", "Football Tracking Report")),
        detector_metrics=_resolve_optional_path(
            inputs.get("detector_metrics"),
            project_root,
            "inputs.detector_metrics",
        ),
        tracker_overall=_resolve_optional_path(
            inputs.get("tracker_overall"),
            project_root,
            "inputs.tracker_overall",
        ),
        benchmark_markdown=_resolve_optional_path(
            inputs.get("benchmark_markdown"),
            project_root,
            "inputs.benchmark_markdown",
        ),
        dataset_audit=_resolve_optional_path(
            inputs.get("dataset_audit"),
            project_root,
            "inputs.dataset_audit",
        ),
        figures_root=_resolve_optional_path(
            inputs.get("figures_root"),
            project_root,
            "inputs.figures_root",
        ),
        markdown_path=markdown_path,
        pdf_path=_resolve_optional_path(output.get("pdf"), project_root, "output.pdf"),
        make_pdf=bool(runtime.get("make_pdf", True)),
    )


def generate_final_report(
    config_path: str | Path,
    output_override: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    config = load_final_report_config(config_path, output_override=output_override)
    if dry_run:
        return {
            "dry_run": True,
            "markdown": str(config.markdown_path),
            "pdf": str(config.pdf_path) if config.pdf_path else None,
        }
    config.markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown = _render_markdown(config)
    config.markdown_path.write_text(markdown, encoding="utf-8")
    pdf_path, pdf_reason = _write_optional_pdf(config, markdown)
    metadata_path = config.markdown_path.with_suffix(".json")
    metadata = {
        "markdown": str(config.markdown_path),
        "pdf": str(pdf_path) if pdf_path else None,
        "pdf_reason": pdf_reason,
        "inputs": {
            "detector_metrics": str(config.detector_metrics) if config.detector_metrics else None,
            "tracker_overall": str(config.tracker_overall) if config.tracker_overall else None,
            "benchmark_markdown": (
                str(config.benchmark_markdown) if config.benchmark_markdown else None
            ),
            "dataset_audit": str(config.dataset_audit) if config.dataset_audit else None,
            "figures_root": str(config.figures_root) if config.figures_root else None,
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"dry_run": False, "paths": {**metadata, "metadata": str(metadata_path)}}


def _render_markdown(config: FinalReportConfig) -> str:
    detector = _load_json_object(config.detector_metrics)
    tracker_rows = _load_tracker_rows(config.tracker_overall)
    benchmark = _read_text(config.benchmark_markdown)
    audit = _load_json_object(config.dataset_audit)
    cli = r".\.venv\Scripts\python.exe -m football_tracking.cli"
    command_lines = [
        f"{cli} doctor",
        f"{cli} prepare-dataset --config configs/sportsmot_data.yaml --overwrite",
        f"{cli} train-detector --config configs/yolo26m_sportsmot_football_train.yaml --device 0",
        f"{cli} evaluate-detector --config configs/yolo26m_sportsmot_football_eval.yaml",
        f"{cli} cache-detections --config configs/detection_cache_yolo26m_all.yaml --overwrite",
        f"{cli} compare-trackers --config "
        "configs/compare_trackers_yolo26m_botsort_identity_stable_all.yaml --overwrite",
        f"{cli} render-video --config configs/render_video.yaml --overwrite",
        f"{cli} analyze-tracking-vlm --config configs/vlm_qwen4b_tracking.yaml "
        "--run-model --overwrite",
        f"{cli} benchmark --config configs/benchmark.yaml",
        f"{cli} generate-report --config configs/report.yaml",
    ]
    lines = [
        f"# {config.title}",
        "",
        "## Overview",
        "This report summarizes the fine-tuned YOLO detector, shared-cache tracker comparison, "
        "TrackEval metrics, Qwen VLM analysis artifacts, runtime benchmark, and generated videos.",
        "",
        "## Dataset",
        _dataset_summary(audit),
        "",
        "## Detector",
        _detector_summary(detector),
        "",
        "## Tracking Evaluation",
        _tracker_table(tracker_rows),
        "",
        "## Benchmark",
        benchmark.strip() if benchmark else "Benchmark table is not available yet.",
        "",
        "## Figures",
        _figure_list(config.figures_root),
        "",
        "## Reproducible Commands",
        "```powershell",
        *command_lines,
        "```",
        "",
        "## Notes",
        "- All configured trackers are compared from the same cached detector outputs.",
        "- TrackEval is the source of truth for HOTA, MOTA, IDF1, IDSW, FP, and FN.",
        "- Qwen VLM consumes keyframes, crops, prompts, and tracking metadata after tracking.",
        "- Test split results should only be reported after final validation decisions are frozen.",
        "",
    ]
    return "\n".join(lines)


def _load_json_object(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _load_tracker_rows(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, list) else []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_text(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


def _dataset_summary(audit: dict[str, Any] | None) -> str:
    if not audit:
        return "Dataset audit summary is not available yet."
    summary = audit.get("summary", audit)
    parts = []
    for key in ("sequence_count", "frame_count", "object_count", "track_count"):
        if key in summary:
            parts.append(f"- {key.replace('_', ' ').title()}: `{summary[key]}`")
    return "\n".join(parts) if parts else "Dataset audit file exists but has no compact summary."


def _detector_summary(detector: dict[str, Any] | None) -> str:
    if not detector:
        return "Detector metrics are not available yet."
    metrics = detector.get("metrics", detector)
    rows = [
        ("Model", metrics.get("model")),
        ("Checkpoint", metrics.get("checkpoint")),
        ("Precision", metrics.get("precision")),
        ("Recall", metrics.get("recall")),
        ("mAP50", metrics.get("map50")),
        ("mAP50-95", metrics.get("map50_95")),
        ("Reason", metrics.get("reason")),
    ]
    return "\n".join(f"- {name}: `{_fmt(value)}`" for name, value in rows)


def _tracker_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "Tracking metrics are not available yet."
    fields = ["tracker", "HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "FP", "FN", "tracker_fps"]
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _field in fields) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(field)) for field in fields) + " |")
    return "\n".join(lines)


def _figure_list(figures_root: Path | None) -> str:
    if figures_root is None or not figures_root.is_dir():
        return "No figures directory is available yet."
    paths = sorted(figures_root.rglob("*.png"))
    if not paths:
        return "No PNG figures were generated yet."
    return "\n".join(f"- `{path}`" for path in paths)


def _fmt(value: Any) -> str:
    if value is None or value == "":
        return "not available"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _write_optional_pdf(config: FinalReportConfig, markdown: str) -> tuple[Path | None, str | None]:
    if not config.make_pdf:
        return None, "PDF generation disabled in config."
    if config.pdf_path is None:
        return None, "No PDF output path configured."
    if (
        importlib.util.find_spec("markdown") is None
        or importlib.util.find_spec("weasyprint") is None
    ):
        return None, "PDF skipped because markdown/weasyprint is not installed."
    import markdown as markdown_lib  # type: ignore[import-not-found]
    import weasyprint  # type: ignore[import-not-found]

    html = markdown_lib.markdown(markdown, extensions=["tables", "fenced_code"])
    config.pdf_path.parent.mkdir(parents=True, exist_ok=True)
    weasyprint.HTML(string=html, base_url=str(config.project_root)).write_pdf(str(config.pdf_path))
    return config.pdf_path, None
