from __future__ import annotations

import json

from football_tracking.benchmarking.benchmark import generate_benchmark
from football_tracking.reporting.final_report import generate_final_report


def test_benchmark_and_final_report_outputs(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FOOTBALL_TRACKING_ROOT", str(tmp_path))
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    metrics_dir = tmp_path / "outputs" / "metrics" / "experiments"
    metrics_dir.mkdir(parents=True)
    detector_path = tmp_path / "outputs" / "metrics" / "detector.json"
    detector_path.parent.mkdir(parents=True, exist_ok=True)
    detector_path.write_text(
        json.dumps(
            {
                "metrics": {
                    "model": "detector_fixture",
                    "precision": 0.7,
                    "recall": 0.6,
                    "map50": 0.8,
                    "map50_95": 0.5,
                }
            }
        ),
        encoding="utf-8",
    )
    overall_path = metrics_dir / "overall.json"
    overall_path.write_text(
        json.dumps(
            [
                {
                    "tracker": "sort",
                    "HOTA": 40.0,
                    "DetA": 50.0,
                    "AssA": 30.0,
                    "MOTA": 55.0,
                    "IDF1": 45.0,
                    "IDSW": 7,
                    "FP": 10,
                    "FN": 20,
                    "tracker_fps": 100.0,
                }
            ]
        ),
        encoding="utf-8",
    )
    per_sequence_path = metrics_dir / "per_sequence.csv"
    per_sequence_path.write_text(
        "tracker,sequence,HOTA,IDF1\nsort,seq,40.0,45.0\n",
        encoding="utf-8",
    )
    benchmark_config = tmp_path / "benchmark.yaml"
    benchmark_config.write_text(
        "\n".join(
            [
                "inputs:",
                "  detector_metrics: outputs/metrics/detector.json",
                "  tracker_overall: outputs/metrics/experiments/overall.json",
                "  tracker_per_sequence: outputs/metrics/experiments/per_sequence.csv",
                "output:",
                "  root: outputs/metrics/benchmark",
                "  figures_root: outputs/figures/benchmark",
            ]
        ),
        encoding="utf-8",
    )

    benchmark = generate_benchmark(benchmark_config)

    assert benchmark["row_count"] == 1
    assert (tmp_path / "outputs" / "metrics" / "benchmark" / "benchmark.csv").is_file()
    assert (tmp_path / "outputs" / "figures" / "benchmark" / "hota.png").is_file()

    report_config = tmp_path / "report.yaml"
    report_config.write_text(
        "\n".join(
            [
                "report:",
                "  title: Fixture Report",
                "inputs:",
                "  detector_metrics: outputs/metrics/detector.json",
                "  tracker_overall: outputs/metrics/experiments/overall.json",
                "  benchmark_markdown: outputs/metrics/benchmark/benchmark.md",
                "  figures_root: outputs/figures/benchmark",
                "output:",
                "  markdown: outputs/reports/tracking_report.md",
                "  pdf: outputs/reports/tracking_report.pdf",
                "runtime:",
                "  make_pdf: false",
            ]
        ),
        encoding="utf-8",
    )

    report = generate_final_report(report_config)
    report_path = tmp_path / "outputs" / "reports" / "tracking_report.md"

    assert report["paths"]["markdown"] == str(report_path)
    assert "Fixture Report" in report_path.read_text(encoding="utf-8")
