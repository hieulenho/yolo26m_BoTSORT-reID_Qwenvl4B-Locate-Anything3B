from football_tracking.reporting.detector_report import write_finetuned_report
from football_tracking.reporting.detector_tables import markdown_metric_table


def test_detector_report_uses_not_available_for_missing_metrics(tmp_path) -> None:
    table = markdown_metric_table({"split": "val", "precision": None})
    path = write_finetuned_report(tmp_path, {"split": "val", "precision": None})

    assert "not available" in table
    assert "not executed" in path.read_text(encoding="utf-8")
