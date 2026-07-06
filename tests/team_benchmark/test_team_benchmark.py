from __future__ import annotations

import json
from pathlib import Path

from football_tracking.locate_tracking.cli.__main__ import main as locate_cli_main
from football_tracking.team_benchmark.comparison import compare_team_benchmark_evaluations
from football_tracking.team_benchmark.evaluator import evaluate_team_benchmark
from football_tracking.team_benchmark.validation import validate_team_benchmark_manifest

MANIFEST = Path("data/team_benchmark/smoke/benchmark_manifest.json")
PRED_A = Path("data/team_benchmark/smoke/predictions_pipeline_a_qwen.json")
PRED_B = Path("data/team_benchmark/smoke/predictions_pipeline_b_locate_qwen.json")


def test_team_benchmark_validation_and_pipeline_a(tmp_path: Path) -> None:
    report = validate_team_benchmark_manifest(MANIFEST)
    assert report.error_count == 0
    assert report.annotated_track_count == 3
    assert report.query_count == 2

    evaluation = evaluate_team_benchmark(
        manifest_path=MANIFEST,
        prediction_manifest_path=PRED_A,
        output_dir=tmp_path / "pipeline_a",
        overwrite=True,
    )

    assert evaluation.aggregate["track_team_accuracy"] == 1.0
    assert evaluation.aggregate["query_selected_track_exact_accuracy"] == 1.0
    assert evaluation.aggregate["correct_id_correct_team_rate"] == 1.0
    assert (tmp_path / "pipeline_a" / "team_benchmark_summary.md").is_file()


def test_team_benchmark_pipeline_b_and_comparison(tmp_path: Path) -> None:
    eval_a = evaluate_team_benchmark(
        manifest_path=MANIFEST,
        prediction_manifest_path=PRED_A,
        output_dir=tmp_path / "pipeline_a",
        overwrite=True,
    )
    eval_b = evaluate_team_benchmark(
        manifest_path=MANIFEST,
        prediction_manifest_path=PRED_B,
        output_dir=tmp_path / "pipeline_b",
        overwrite=True,
    )

    assert eval_b.aggregate["query_selected_track_exact_accuracy"] == 0.5
    assert eval_b.aggregate["query_resolved_rate"] == 0.5
    assert eval_b.aggregate["grounding_calls_per_query"] == 8.0

    comparison = compare_team_benchmark_evaluations(
        evaluations=(tmp_path / "pipeline_a", tmp_path / "pipeline_b"),
        output_dir=tmp_path / "comparison",
        overwrite=True,
    )
    assert comparison["variant_count"] == 2
    rows = {row["variant_id"]: row for row in comparison["rows"]}
    assert rows[eval_a.variant_id]["correct_id_correct_team_rate"] == 1.0
    assert rows[eval_b.variant_id]["correct_id_correct_team_rate"] == 0.5
    assert (tmp_path / "comparison" / "team_benchmark_comparison.csv").is_file()


def test_team_benchmark_cli_smoke(tmp_path: Path) -> None:
    validation_path = tmp_path / "validation.json"
    assert (
        locate_cli_main(
            [
                "validate-team-benchmark",
                "--manifest",
                str(MANIFEST),
                "--output",
                str(validation_path),
            ]
        )
        == 0
    )
    assert json.loads(validation_path.read_text(encoding="utf-8"))["summary"]["errors"] == 0

    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    assert (
        locate_cli_main(
            [
                "run-team-benchmark",
                "--manifest",
                str(MANIFEST),
                "--predictions",
                str(PRED_A),
                "--output-dir",
                str(out_a),
                "--overwrite",
            ]
        )
        == 0
    )
    assert (
        locate_cli_main(
            [
                "run-team-benchmark",
                "--manifest",
                str(MANIFEST),
                "--predictions",
                str(PRED_B),
                "--output-dir",
                str(out_b),
                "--overwrite",
            ]
        )
        == 0
    )
    assert (
        locate_cli_main(
            [
                "compare-team-benchmarks",
                "--evaluation",
                str(out_a),
                "--evaluation",
                str(out_b),
                "--output-dir",
                str(tmp_path / "comparison"),
                "--overwrite",
            ]
        )
        == 0
    )
    assert (tmp_path / "comparison" / "team_benchmark_comparison.md").is_file()
