"""Validation-only ranking helpers for tracker configs."""

from __future__ import annotations

from typing import Any


def ranking_key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    hota = float(row.get("HOTA") or -1.0)
    idf1 = float(row.get("IDF1") or -1.0)
    assa = float(row.get("AssA") or -1.0)
    idsw = float(row.get("IDSW") if row.get("IDSW") is not None else 1e12)
    fps = float(row.get("tracker_fps") or -1.0)
    return hota, idf1, assa, -idsw, fps


def rank_tracker_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=ranking_key, reverse=True)
