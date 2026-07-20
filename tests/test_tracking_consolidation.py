from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml

from football_tracking.benchmarking.tracking_consolidation import (
    TrackingConsolidationError,
    consolidate_tracking_benchmark,
)


def _write_source(root: Path, tracker: str, *, frames: int = 3) -> tuple[Path, Path]:
    summary = root / f"{tracker}.json"
    per_sequence = root / f"{tracker}.csv"
    summary.write_text(
        json.dumps(
            [
                {
                    "tracker": tracker,
                    "confidence_threshold": 0.1,
                    "sequence_count": 1,
                    "frame_count": frames,
                    "HOTA": 50.0,
                    "DetA": 60.0,
                    "AssA": 40.0,
                    "LocA": 90.0,
                    "MOTA": 70.0,
                    "MOTP": 85.0,
                    "IDF1": 55.0,
                    "IDP": 54.0,
                    "IDR": 56.0,
                    "IDSW": 2,
                    "FP": 3,
                    "FN": 4,
                    "Frag": 1,
                    "tracker_fps": 100.0,
                    "cached_pipeline_fps": 80.0,
                    "unique_predicted_ids": 5,
                    "tracker_config_hash": "abc",
                    "smoke_only": False,
                    "partial_sequences": False,
                }
            ]
        ),
        encoding="utf-8",
    )
    with per_sequence.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("tracker", "sequence", "frame_count", "HOTA", "IDF1", "IDSW"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "tracker": tracker,
                "sequence": "seq1",
                "frame_count": frames,
                "HOTA": 50,
                "IDF1": 55,
                "IDSW": 2,
            }
        )
    return summary, per_sequence


def _write_config(root: Path, summary: Path, per_sequence: Path) -> Path:
    dataset = root / "dataset"
    cache = root / "cache"
    dataset.mkdir()
    cache.mkdir()
    config = root / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "benchmark": {
                    "dataset": str(dataset),
                    "split": "all",
                    "detector_cache_root": str(cache),
                },
                "expected": {
                    "sequence_count": 1,
                    "frame_count": 3,
                    "confidence_threshold": 0.1,
                },
                "sources": [
                    {
                        "tracker": "tracker_a",
                        "display_name": "Tracker A",
                        "summary": str(summary),
                        "per_sequence": str(per_sequence),
                    }
                ],
                "output": {
                    "root": str(root / "output"),
                    "overwrite": False,
                    "write_figures": False,
                },
            }
        ),
        encoding="utf-8",
    )
    return config


def test_consolidation_writes_audited_outputs(tmp_path: Path) -> None:
    summary, per_sequence = _write_source(tmp_path, "tracker_a")
    config = _write_config(tmp_path, summary, per_sequence)

    result = consolidate_tracking_benchmark(config)

    assert result["status"] == "ok"
    assert result["tracker_count"] == 1
    manifest = json.loads((tmp_path / "output" / "benchmark_manifest.json").read_text())
    assert manifest["compatibility_contract"]["frame_count"] == 3
    assert manifest["sources"][0]["summary_sha256"]


def test_consolidation_rejects_partial_or_incompatible_source(tmp_path: Path) -> None:
    summary, per_sequence = _write_source(tmp_path, "tracker_a", frames=2)
    config = _write_config(tmp_path, summary, per_sequence)

    with pytest.raises(TrackingConsolidationError, match="frame_count=2"):
        consolidate_tracking_benchmark(config)
