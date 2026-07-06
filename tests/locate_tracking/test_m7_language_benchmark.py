"""Milestone 7 language benchmark, ablation, failure, report, and demo tests."""

from __future__ import annotations

import json
from pathlib import Path

from football_tracking.locate_tracking.benchmark.evaluator import evaluate_language_benchmark
from football_tracking.locate_tracking.benchmark.validation import validate_benchmark_manifest
from football_tracking.locate_tracking.cli.__main__ import main as locate_cli_main
from football_tracking.locate_tracking.experiments.runner import run_language_ablation
from football_tracking.locate_tracking.failure_analysis.summary import analyze_failures
from football_tracking.locate_tracking.reporting.demo_manifest import build_demo_manifest
from football_tracking.locate_tracking.reporting.report_builder import generate_language_report

MANIFEST = Path("data/language_tracking/benchmark_manifest.json")
PRED_FULL = Path("data/language_tracking/smoke/predictions_full_system.json")
PRED_BASELINE = Path("data/language_tracking/smoke/predictions_single_frame.json")


def test_language_benchmark_validation_and_full_evaluation(tmp_path: Path) -> None:
    report = validate_benchmark_manifest(MANIFEST)
    assert report.error_count == 0
    assert report.query_count == 1

    evaluation = evaluate_language_benchmark(
        manifest_path=MANIFEST,
        prediction_manifest_path=PRED_FULL,
        output_dir=tmp_path / "full",
        overwrite=True,
    )

    assert evaluation.aggregate["micro_target_f1"] == 1.0
    assert evaluation.aggregate["reacquisition_success_rate"] == 1.0
    assert evaluation.per_query[0]["raw_id_transitions_along_semantic_target"] == 1
    assert (tmp_path / "full" / "per_query_metrics.csv").is_file()


def test_language_baseline_failure_analysis_report_and_demo(tmp_path: Path) -> None:
    baseline = evaluate_language_benchmark(
        manifest_path=MANIFEST,
        prediction_manifest_path=PRED_BASELINE,
        output_dir=tmp_path / "baseline",
        overwrite=True,
    )
    assert baseline.aggregate["micro_target_recall"] == 0.5

    failures = analyze_failures(
        evaluation=tmp_path / "baseline",
        output_dir=tmp_path / "failures",
        overwrite=True,
    )
    assert failures["failure_count"] == 1
    assert "reacquisition_failed" in failures["category_counts"]

    report = generate_language_report(
        evaluation=tmp_path / "baseline",
        failures=tmp_path / "failures" / "failure_cases.json",
        output=tmp_path / "report.md",
        overwrite=True,
    )
    assert Path(report["paths"]["markdown"]).is_file()
    assert "Language Benchmark Metrics" in (tmp_path / "report.md").read_text(encoding="utf-8")

    demo = build_demo_manifest(
        evaluation=tmp_path / "baseline",
        output_dir=tmp_path / "demo",
        overwrite=True,
    )
    assert Path(demo["paths"]["json"]).is_file()


def test_language_ablation_runner_smoke(tmp_path: Path) -> None:
    config = tmp_path / "ablation.yaml"
    config.write_text(
        f"""
benchmark:
  manifest: {MANIFEST.as_posix()}
  iou_threshold: 0.5
variants:
  - variant_id: a0
    name: A0
    prediction_manifest: {PRED_BASELINE.as_posix()}
  - variant_id: a5
    name: A5
    prediction_manifest: {PRED_FULL.as_posix()}
output:
  directory: {str(tmp_path / "ablation").replace(chr(92), "/")}
""".strip(),
        encoding="utf-8",
    )

    dry = run_language_ablation(config, dry_run=True)
    assert dry["dry_run"] is True
    result = run_language_ablation(config, overwrite=True)

    assert result["variant_count"] == 2
    rows = {row["variant_id"]: row for row in result["rows"]}
    assert rows["a5"]["micro_target_f1"] > rows["a0"]["micro_target_f1"]
    assert (tmp_path / "ablation" / "ablation_results.csv").is_file()


def test_language_benchmark_cli_smoke(tmp_path: Path) -> None:
    validation_path = tmp_path / "validation.json"
    assert (
        locate_cli_main(
            [
                "validate-language-benchmark",
                "--manifest",
                str(MANIFEST),
                "--output",
                str(validation_path),
            ]
        )
        == 0
    )
    assert json.loads(validation_path.read_text(encoding="utf-8"))["summary"]["errors"] == 0

    output_dir = tmp_path / "cli_eval"
    assert (
        locate_cli_main(
            [
                "run-language-benchmark",
                "--manifest",
                str(MANIFEST),
                "--predictions",
                str(PRED_FULL),
                "--output-dir",
                str(output_dir),
                "--overwrite",
            ]
        )
        == 0
    )
    assert (output_dir / "aggregate_metrics.json").is_file()

    assert (
        locate_cli_main(
            [
                "build-language-demo",
                "--evaluation",
                str(output_dir),
                "--output-dir",
                str(tmp_path / "demo_cli"),
                "--overwrite",
            ]
        )
        == 0
    )
