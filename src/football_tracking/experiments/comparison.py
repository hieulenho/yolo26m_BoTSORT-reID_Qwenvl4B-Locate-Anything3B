"""Comparison helpers for tracker experiment rows."""

from __future__ import annotations

from typing import Any


def tracker_rows_by_name(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("tracker")): row for row in rows if row.get("tracker")}
