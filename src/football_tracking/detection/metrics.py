"""Baseline metric parsing from Ultralytics validator results."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _attr_or_key(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


@dataclass(frozen=True)
class BaselineMetrics:
    precision: float | None = None
    recall: float | None = None
    map50: float | None = None
    map50_95: float | None = None
    map75: float | None = None
    image_count: int | None = None
    ground_truth_count: int | None = None
    prediction_count: int | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "map50": self.map50,
            "map50_95": self.map50_95,
            "map75": self.map75,
            "image_count": self.image_count,
            "ground_truth_count": self.ground_truth_count,
            "prediction_count": self.prediction_count,
            "reason": self.reason,
        }


def metrics_not_available(reason: str) -> BaselineMetrics:
    return BaselineMetrics(reason=reason)


def parse_ultralytics_metrics(result: Any) -> BaselineMetrics:
    box = _attr_or_key(result, "box")
    results_dict = _attr_or_key(result, "results_dict") or {}
    precision = _finite_or_none(_attr_or_key(box, "mp"))
    recall = _finite_or_none(_attr_or_key(box, "mr"))
    map50 = _finite_or_none(_attr_or_key(box, "map50"))
    map50_95 = _finite_or_none(_attr_or_key(box, "map"))
    map75 = _finite_or_none(_attr_or_key(box, "map75"))

    precision = (
        precision
        if precision is not None
        else _finite_or_none(results_dict.get("metrics/precision(B)"))
    )
    recall = (
        recall if recall is not None else _finite_or_none(results_dict.get("metrics/recall(B)"))
    )
    map50 = map50 if map50 is not None else _finite_or_none(results_dict.get("metrics/mAP50(B)"))
    map50_95 = (
        map50_95
        if map50_95 is not None
        else _finite_or_none(results_dict.get("metrics/mAP50-95(B)"))
    )
    map75 = map75 if map75 is not None else _finite_or_none(results_dict.get("metrics/mAP75(B)"))

    reason = None
    if all(value is None for value in (precision, recall, map50, map50_95, map75)):
        reason = "Ultralytics metrics object did not expose detection metrics."
    return BaselineMetrics(
        precision=precision,
        recall=recall,
        map50=map50,
        map50_95=map50_95,
        map75=map75,
        reason=reason,
    )


DetectionMetrics = BaselineMetrics


def metrics_to_flat_record(metrics: BaselineMetrics) -> dict[str, Any]:
    return metrics.to_dict()
