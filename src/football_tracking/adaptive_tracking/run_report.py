"""Build an auditable end-to-end report for one adaptive tracking run."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from football_tracking.detection.serialization import file_sha256, runtime_versions


class AdaptiveRunReportError(RuntimeError):
    """Raised when an adaptive run report cannot be assembled."""


def build_adaptive_run_report(
    *,
    run_root: str | Path,
    tracking_metadata: str | Path,
    semantic_metadata: str | Path,
    output_path: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    root = Path(run_root)
    tracking_path = Path(tracking_metadata)
    semantic_path = Path(semantic_metadata)
    output = Path(output_path)
    required = {
        "discovery": root / "discovery" / "scene_discovery.json",
        "route": root / "plan" / "detector_route.json",
        "tracking": tracking_path,
        "fused_semantics": root / "fused_track_semantics.json",
        "semantic_render": semantic_path,
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not path.is_file()]
    if missing:
        raise AdaptiveRunReportError("Missing required run artifact(s): " + "; ".join(missing))
    if output.exists() and not overwrite:
        raise AdaptiveRunReportError(f"Run report exists and overwrite=false: {output}")

    discovery = _read_json(required["discovery"])
    route = _read_json(required["route"])
    tracking = _read_json(tracking_path)
    fused = _read_json(required["fused_semantics"])
    semantic_render = _read_json(semantic_path)
    qwen_path = root / "qwen_track_semantics" / "vlm_answer.json"
    locate_path = root / "locate_verification" / "grounding_verification.json"
    qwen = _read_json(qwen_path) if qwen_path.is_file() else None
    locate = _read_json(locate_path) if locate_path.is_file() else None

    report = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "run_root": str(root.resolve()),
        "status": _overall_status(tracking, qwen, locate),
        "scene": {
            "domain": _domain_name(discovery.get("domain")),
            "description": discovery.get("context", ""),
            "discovered_class_count": len(discovery.get("objects", [])),
            "objects": discovery.get("objects", []),
            "keyframe_count": len(discovery.get("keyframes", [])),
            "inference_seconds": discovery.get("metadata", {}).get("inference_seconds"),
        },
        "route": route,
        "tracking": {
            "tracker": tracking.get("tracker"),
            "output_mot": tracking.get("output_mot"),
            "output_video": tracking.get("output_video"),
            "tracker_diagnostics": tracking.get("tracker_diagnostics"),
            "frame_count": tracking.get("frame_count"),
            "detection_count": tracking.get("detection_count"),
            "unique_track_count": tracking.get("unique_track_count"),
            "timing": tracking.get("timing", {}),
            "validation": tracking.get("validation"),
        },
        "qwen_track_semantics": _qwen_metrics(qwen),
        "locateanything_verification": _locate_metrics(locate),
        "semantic_fusion": fused.get("summary", {}),
        "render": {
            "timing": semantic_render.get("timing", {}),
            "semantics": semantic_render.get("semantics_summary", {}),
            "video": semantic_render.get("video", {}),
        },
        "evaluation_scope": {
            "ground_truth_accuracy_available": False,
            "reason": (
                "This raw-video run has no human semantic GT manifest. Coverage and "
                "confidence are operational metrics, not class accuracy."
            ),
            "tracking_gt_benchmark": (
                "Use the SportsMOT benchmark report for HOTA/MOTA/IDF1/IDSW comparison."
            ),
        },
        "hardware": runtime_versions(),
        "artifacts": _artifact_manifest(
            [*required.values(), qwen_path, locate_path]
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(output)
    markdown_path = output.with_suffix(".md")
    temporary_markdown = markdown_path.with_suffix(markdown_path.suffix + ".tmp")
    temporary_markdown.write_text(_markdown(report), encoding="utf-8")
    temporary_markdown.replace(markdown_path)
    return {
        "status": "ok",
        "report": str(output.resolve()),
        "markdown": str(markdown_path.resolve()),
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AdaptiveRunReportError(f"Invalid JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise AdaptiveRunReportError(f"JSON artifact root must be an object: {path}")
    return payload


def _overall_status(
    tracking: dict[str, Any],
    qwen: dict[str, Any] | None,
    locate: dict[str, Any] | None,
) -> str:
    if tracking.get("errors"):
        return "failed"
    if qwen and qwen.get("status") in {"failed", "model_failed"}:
        return "partial"
    if locate and locate.get("summary", {}).get("error_count", 0):
        return "partial"
    return "ok"


def _qwen_metrics(payload: dict[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {"status": "not_run"}
    return {
        "status": payload.get("status"),
        "model_id": payload.get("model_id"),
        "quantization": payload.get("quantization"),
        "batch_count": payload.get("batch_count"),
        "image_count": payload.get("image_count"),
        "timing": payload.get("timing", {}),
        "cuda_memory": payload.get("cuda_memory", {}),
        "coverage": payload.get("coverage", {}),
        "parse_failure_count": len(payload.get("parse_failures", [])),
    }


def _locate_metrics(payload: dict[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {"status": "not_run"}
    summary = payload.get("summary", {})
    return {
        "status": "skipped" if summary.get("skipped") else "ok",
        "summary": summary,
        "timing": payload.get("timing", {}),
        "cuda_memory": payload.get("cuda_memory", {}),
    }


def _artifact_manifest(paths: list[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in paths
        if path.is_file()
    ]


def _markdown(report: dict[str, Any]) -> str:
    scene = report["scene"]
    tracking = report["tracking"]
    qwen = report["qwen_track_semantics"]
    locate = report["locateanything_verification"]
    fusion = report["semantic_fusion"]
    timing = tracking.get("timing", {})
    diagnostics = tracking.get("tracker_diagnostics") or {}
    return "\n".join(
        [
            "# Adaptive tracking run",
            "",
            f"- Status: **{report['status']}**",
            f"- Domain: **{scene.get('domain')}**",
            f"- Dynamic classes: **{scene.get('discovered_class_count')}**",
            f"- Detector route: **{report['route'].get('route_name')}**",
            f"- Tracker: **{tracking.get('tracker')}**",
            "- Raw/stable class switches: "
            f"**{diagnostics.get('raw_class_switches', 0)} / "
            f"{diagnostics.get('stable_class_switches', 0)}**",
            f"- Frames: **{tracking.get('frame_count')}**",
            f"- Detector FPS: **{_fmt(timing.get('detector_fps'))}**",
            f"- Tracker FPS: **{_fmt(timing.get('tracker_fps'))}**",
            f"- End-to-end FPS: **{_fmt(timing.get('end_to_end_fps'))}**",
            f"- Qwen batches: **{qwen.get('batch_count', 'not run')}**",
            f"- Locate requests: **{locate.get('summary', {}).get('request_count', 0)}**",
            f"- Accepted semantic tracks: **{fusion.get('accepted_count', 0)}**",
            "",
            "Semantic class accuracy is not reported for this raw video because no human "
            "ground-truth manifest was supplied.",
            "",
        ]
    )


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def _domain_name(value: Any) -> str:
    if isinstance(value, dict):
        name = value.get("name") or value.get("canonical_name")
        return str(name or "unknown")
    return str(value or "unknown")
