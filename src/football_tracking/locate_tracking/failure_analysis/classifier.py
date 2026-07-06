"""Deterministic failure categorization for language benchmark query rows."""

from __future__ import annotations

from typing import Any

from football_tracking.locate_tracking.failure_analysis.schemas import FailureCase


def classify_failure(row: dict[str, Any]) -> FailureCase | None:
    sequence = str(row.get("sequence_name", "unknown"))
    query = str(row.get("query_id", "unknown"))
    status = str(row.get("status", "missing_prediction"))
    if status != "resolved":
        return FailureCase(
            sequence_name=sequence,
            query_id=query,
            category=f"query_{status}",
            reason="query did not produce a resolved semantic target",
            severity="high",
            evidence={"status": status},
        )
    if row.get("initial_selection_correct") is False:
        return FailureCase(
            sequence_name=sequence,
            query_id=query,
            category="initial_selection_error",
            reason="first predicted target frame did not match GT target",
            severity="high",
            evidence={"target_precision": row.get("target_precision")},
        )
    if _lt(row.get("target_precision"), 0.5):
        return FailureCase(
            sequence_name=sequence,
            query_id=query,
            category="low_precision",
            reason="too many predicted semantic target frames match non-target boxes",
            severity="medium",
            evidence={"target_precision": row.get("target_precision")},
        )
    if _lt(row.get("target_recall"), 0.5):
        return FailureCase(
            sequence_name=sequence,
            query_id=query,
            category="low_recall",
            reason="semantic target missed too many GT target frames",
            severity="medium",
            evidence={"target_recall": row.get("target_recall")},
        )
    if int(row.get("semantic_target_switches") or 0) > 0:
        return FailureCase(
            sequence_name=sequence,
            query_id=query,
            category="semantic_target_switch",
            reason="semantic target produced persistent wrong-target predictions",
            severity="high",
            evidence={"semantic_target_switches": row.get("semantic_target_switches")},
        )
    if (
        int(row.get("reacquisition_opportunity_count") or 0) > 0
        and int(row.get("reacquisition_success_count") or 0) == 0
    ):
        return FailureCase(
            sequence_name=sequence,
            query_id=query,
            category="reacquisition_failed",
            reason="annotated recovery opportunity was not successfully recovered",
            severity="medium",
            evidence={
                "opportunities": row.get("reacquisition_opportunity_count"),
                "false_reacquisition_count": row.get("false_reacquisition_count"),
            },
        )
    return None


def _lt(value: Any, threshold: float) -> bool:
    if value is None:
        return False
    try:
        return float(value) < threshold
    except (TypeError, ValueError):
        return False
