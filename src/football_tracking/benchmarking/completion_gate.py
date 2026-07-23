"""Evidence-based release gate for the multi-domain tracking benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class CompletionGateError(RuntimeError):
    """Raised when completion evidence is malformed or output is unsafe to replace."""


def build_completion_gate(
    *,
    dataset_readiness: str | Path,
    semantic_gt_status: str | Path,
    idsw_review_status: str | Path,
    idsw_agreement: str | Path,
    realtime_report: str | Path,
    semantic_comparison: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    evidence_paths = {
        "dataset_readiness": Path(dataset_readiness).resolve(),
        "semantic_gt_status": Path(semantic_gt_status).resolve(),
        "idsw_review_status": Path(idsw_review_status).resolve(),
        "idsw_agreement": Path(idsw_agreement).resolve(),
        "realtime_report": Path(realtime_report).resolve(),
        "semantic_comparison": Path(semantic_comparison).resolve(),
    }
    evidence = {name: _optional_json(path) for name, path in evidence_paths.items()}
    checks = [
        _dataset_check(evidence["dataset_readiness"]),
        _status_check(
            "semantic_ground_truth",
            evidence["semantic_gt_status"],
            expected="ready",
        ),
        _status_check(
            "idsw_human_review",
            evidence["idsw_review_status"],
            expected="ready",
        ),
        _status_check(
            "idsw_inter_reviewer_agreement",
            evidence["idsw_agreement"],
            expected="agreed",
        ),
        _realtime_check(evidence["realtime_report"]),
        _semantic_comparison_check(evidence["semantic_comparison"]),
    ]
    payload = {
        "schema_version": 1,
        "complete": all(check["passed"] for check in checks),
        "passed_check_count": sum(check["passed"] for check in checks),
        "check_count": len(checks),
        "checks": checks,
        "evidence": {
            name: {"path": str(path), "exists": path.is_file()}
            for name, path in evidence_paths.items()
        },
    }
    root = Path(output_dir).resolve()
    json_path = root / "completion_gate.json"
    markdown_path = root / "completion_gate.md"
    if not overwrite and (json_path.exists() or markdown_path.exists()):
        raise CompletionGateError(f"Completion gate output exists: {root}")
    root.mkdir(parents=True, exist_ok=True)
    _write_atomic(json_path, json.dumps(payload, indent=2))
    _write_atomic(markdown_path, _markdown(payload))
    return payload


def _optional_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CompletionGateError(f"Invalid JSON evidence: {path}") from exc
    if not isinstance(payload, dict):
        raise CompletionGateError(f"Evidence root must be an object: {path}")
    return payload


def _dataset_check(payload: dict[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return _check("official_multidomain_gt", False, "dataset readiness is missing")
    if "required_source_count" in payload:
        required_count = int(payload.get("required_source_count", 0))
        required_ready = int(payload.get("required_ready_count", 0))
        optional_count = int(payload.get("optional_source_count", 0))
        optional_ready = int(payload.get("optional_ready_count", 0))
        return _check(
            "official_multidomain_gt",
            required_count > 0 and required_ready == required_count,
            f"{required_ready}/{required_count} required sources are ready; "
            f"{optional_ready}/{optional_count} optional sources are ready",
        )
    source_count = int(payload.get("source_count", 0))
    ready_count = int(payload.get("ready_count", 0))
    return _check(
        "official_multidomain_gt",
        source_count > 0 and ready_count == source_count,
        f"{ready_count}/{source_count} registered sources are ready",
    )


def _status_check(
    name: str, payload: dict[str, Any] | None, *, expected: str
) -> dict[str, Any]:
    if payload is None:
        return _check(name, False, "evidence is missing")
    status = str(payload.get("status", "missing"))
    return _check(name, status == expected, f"status={status}; expected={expected}")


def _realtime_check(payload: dict[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return _check("physical_realtime", False, "physical realtime report is missing")
    runs = payload.get("runs", [])
    if not isinstance(runs, list):
        runs = []
    valid_runs = [
        row
        for row in runs
        if isinstance(row, dict)
        and row.get("source_kind") in {"webcam", "rtsp"}
        and int(row.get("frames_processed", 0)) >= 150
        and row.get("p95_latency_ms") is not None
        and row.get("processing_fps") is not None
        and row.get("drop_rate") is not None
    ]
    required_profiles = (
        "bounded_tracking_only",
        "bounded_semantic_deferred",
        "no_drop_semantic_deferred",
    )
    counts = {
        profile: sum(
            _realtime_profile(str(row.get("name", ""))) == profile
            for row in valid_runs
        )
        for profile in required_profiles
    }
    complete = all(count >= 3 for count in counts.values())
    detail = ", ".join(f"{name}={count}/3" for name, count in counts.items())
    return _check(
        "physical_realtime",
        complete,
        f"valid repeated webcam/RTSP runs: {detail}",
    )


def _realtime_profile(name: str) -> str:
    normalized = name.strip().replace("\\", "/").split("/")[-1]
    head, marker, repeat = normalized.rpartition("_r")
    if marker and repeat.isdigit():
        return head
    return normalized


def _semantic_comparison_check(payload: dict[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return _check("semantic_abc", False, "A/B/C comparison is missing")
    rows = payload.get("pipelines", [])
    required = (
        "semantic_accuracy",
        "semantic_macro_f1",
        "fine_label_accuracy",
        "unknown_rejection_f1",
        "hallucination_rate",
        "semantic_cold_seconds",
        "sequential_peak_gib",
    )
    valid = [
        row
        for row in rows
        if isinstance(row, dict) and all(row.get(key) is not None for key in required)
    ] if isinstance(rows, list) else []
    return _check(
        "semantic_abc",
        len(valid) == 3,
        f"{len(valid)}/3 pipelines contain all required quality and cost metrics",
    )


def _check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Multi-domain completion gate",
        "",
        f"Overall: **{'PASS' if payload['complete'] else 'INCOMPLETE'}** "
        f"({payload['passed_check_count']}/{payload['check_count']} checks)",
        "",
        "| Check | Result | Evidence |",
        "|---|---|---|",
    ]
    for row in payload["checks"]:
        lines.append(
            f"| {row['name']} | {'PASS' if row['passed'] else 'BLOCKED'} | "
            f"{row['detail']} |"
        )
    lines.extend(
        [
            "",
            "A blocked gate is not a software failure. It identifies missing independent "
            "evidence that must not be replaced by model-generated labels.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_atomic(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)
