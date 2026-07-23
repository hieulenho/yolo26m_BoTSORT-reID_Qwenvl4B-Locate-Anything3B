from __future__ import annotations

import json
from pathlib import Path

import yaml

from football_tracking.benchmarking.multidomain_tracking_summary import (
    build_multidomain_tracking_summary,
)


def test_build_multidomain_tracking_summary_selects_and_weights_evidence(
    tmp_path: Path,
) -> None:
    metrics = tmp_path / "metrics.json"
    metrics.write_text(
        json.dumps(
            {
                "trackers": [
                    {
                        "tracker": "ocsort",
                        "frame_count": 30,
                        "sequence_count": 1,
                        "HOTA": 70,
                        "DetA": 60,
                        "AssA": 80,
                        "MOTA": 75,
                        "IDF1": 85,
                        "IDSW": 3,
                        "FP": 4,
                        "FN": 5,
                        "Frag": 2,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps({"tracking": {"frame_count": 30, "timing": {"total_pipeline_seconds": 2}}}),
        encoding="utf-8",
    )
    hardware = tmp_path / "hardware.json"
    hardware.write_text(json.dumps({"hardware": {"gpu_name": "test GPU"}}), encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "output_dir": str(tmp_path / "out"),
                "publish_report_dir": str(tmp_path / "published"),
                "publish_figure_dir": str(tmp_path / "published_figures"),
                "hardware_source": str(hardware),
                "benchmarks": [
                    {
                        "id": "traffic",
                        "domain": "traffic",
                        "metrics_source": str(metrics),
                        "selector": {
                            "container": "trackers",
                            "key": "tracker",
                            "value": "ocsort",
                        },
                        "fps": {"reports": [str(report)]},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = build_multidomain_tracking_summary(config, overwrite=True)

    row = result["benchmarks"][0]
    assert row["HOTA"] == 70.0
    assert row["end_to_end_fps"] == 15.0
    assert row["idsw_per_1000_frames"] == 100.0
    assert Path(result["paths"]["markdown"]).is_file()
    assert Path(result["published_paths"]["markdown"]).is_file()
    assert len(result["published_figures"]) == 3
    published_markdown = Path(result["published_paths"]["markdown"]).read_text(
        encoding="utf-8"
    )
    assert "(../assets/benchmarks/multidomain_tracking_quality.png)" in published_markdown
