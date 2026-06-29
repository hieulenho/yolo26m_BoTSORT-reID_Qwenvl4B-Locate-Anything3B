from pathlib import Path

from football_tracking.detection.evaluate import evaluate_with_ultralytics
from football_tracking.detection.metrics import parse_ultralytics_metrics
from football_tracking.reporting.baseline_report import write_baseline_report


class _BoxMetrics:
    mp = 0.1
    mr = 0.2
    map50 = 0.3
    map = 0.4
    map75 = 0.5


class _Result:
    box = _BoxMetrics()


class _FakeModel:
    def val(self, **_kwargs: object) -> _Result:
        return _Result()


def test_parse_ultralytics_metrics_keeps_map50_and_map50_95_distinct() -> None:
    metrics = parse_ultralytics_metrics(_Result())

    assert metrics.precision == 0.1
    assert metrics.recall == 0.2
    assert metrics.map50 == 0.3
    assert metrics.map50_95 == 0.4
    assert metrics.map75 == 0.5


def test_evaluate_with_fake_model_and_missing_metrics_report_null(tmp_path: Path) -> None:
    metrics = evaluate_with_ultralytics(
        "yolov8m.pt",
        tmp_path / "dataset.yaml",
        "val",
        640,
        0.25,
        0.7,
        1,
        "cpu",
        model=_FakeModel(),
    )
    missing = parse_ultralytics_metrics(object())
    report_paths = write_baseline_report(
        {
            "dataset": {"split": "val"},
            "model": {"weights": "yolov8m.pt"},
            "inference": {"imgsz": 640, "conf": 0.25, "iou": 0.7},
            "metrics": missing.to_dict(),
            "timing": {},
            "counts": {},
            "runtime": {},
        },
        tmp_path,
    )

    assert metrics.map50 == 0.3
    assert missing.map50 is None
    assert "not available" in report_paths["markdown"].read_text(encoding="utf-8")
