"""Collect compact evidence snippets for failure cases."""

from __future__ import annotations

from typing import Any


def collect_failure_evidence(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "query_text": row.get("query_text"),
        "query_category": row.get("query_category"),
        "difficulty": row.get("difficulty"),
        "target_precision": row.get("target_precision"),
        "target_recall": row.get("target_recall"),
        "target_f1": row.get("target_f1"),
        "continuity": row.get("target_continuity_ratio"),
        "frames_to_reacquire": row.get("frames_to_reacquire"),
    }
