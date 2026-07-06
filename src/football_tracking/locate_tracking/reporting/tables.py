"""Markdown table helpers for language benchmark reports."""

from __future__ import annotations

from typing import Any


def metric_table(metrics: dict[str, Any], keys: tuple[str, ...]) -> str:
    lines = ["| Metric | Value |", "|---|---:|"]
    for key in keys:
        lines.append(f"| {key} | {_fmt(metrics.get(key))} |")
    return "\n".join(lines)


def rows_table(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> str:
    if not rows:
        return "No rows available."
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _field in fields) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(field)) for field in fields) + " |")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
