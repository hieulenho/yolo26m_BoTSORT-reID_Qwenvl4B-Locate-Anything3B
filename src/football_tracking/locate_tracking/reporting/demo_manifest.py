"""Build a lightweight demo manifest from benchmark results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class DemoManifestError(RuntimeError):
    """Raised when a demo manifest cannot be generated."""


def build_demo_manifest(
    *,
    evaluation: str | Path,
    output_dir: str | Path,
    max_cases: int = 5,
    overwrite: bool = False,
) -> dict[str, Any]:
    rows = _load_rows(evaluation)
    selected = sorted(
        rows,
        key=lambda row: (
            -(float(row.get("target_f1") or 0.0)),
            str(row.get("query_id")),
        ),
    )[:max_cases]
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "demo_manifest.json"
    md_path = root / "demo_manifest.md"
    for path in (json_path, md_path):
        if path.exists() and not overwrite:
            raise DemoManifestError(f"Output exists and overwrite=false: {path}")
    payload = {
        "case_count": len(selected),
        "cases": [
            {
                "sequence_name": row.get("sequence_name"),
                "query_id": row.get("query_id"),
                "query_text": row.get("query_text"),
                "target_f1": row.get("target_f1"),
                "continuity": row.get("target_continuity_ratio"),
                "note": "Use source video plus semantic target overlay artifacts when available.",
            }
            for row in selected
        ],
    }
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    md_path.write_text(_markdown(payload), encoding="utf-8")
    return {"status": "ok", "paths": {"json": str(json_path), "markdown": str(md_path)}}


def _load_rows(evaluation: str | Path) -> list[dict[str, Any]]:
    path = Path(evaluation)
    if path.is_dir():
        path = path / "per_query_metrics.json"
    if not path.is_file():
        raise DemoManifestError(f"Per-query metrics do not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise DemoManifestError("Per-query metrics JSON must be a list.")
    return [dict(row) for row in payload]


def _markdown(payload: dict[str, Any]) -> str:
    lines = ["# Language Tracking Demo Manifest", ""]
    for case in payload["cases"]:
        lines.append(f"- `{case['query_id']}`: {case['query_text']} (F1={case['target_f1']})")
    return "\n".join(lines) + "\n"
