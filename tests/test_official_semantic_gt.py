from pathlib import Path

import yaml

from football_tracking.benchmarking.official_semantic_gt import (
    audit_official_semantic_gt,
)


def _manifest(path: Path, domain: str, track_id: int) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "require_review_metadata": False,
                "ground_truth_source": "official MOT annotations",
                "samples": [
                    {
                        "sample_id": domain,
                        "ground_truth": {
                            "domain": domain,
                            "tracks": [
                                {
                                    "track_id": track_id,
                                    "class_label": "object",
                                }
                            ],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_official_semantic_gt_requires_multiple_valid_domains(
    tmp_path: Path,
) -> None:
    traffic = _manifest(tmp_path / "traffic.yaml", "traffic", 1)
    wildlife = _manifest(tmp_path / "wildlife.yaml", "wildlife", 2)

    result = audit_official_semantic_gt(
        [traffic, wildlife],
        minimum_domains=2,
        minimum_tracks=2,
    )

    assert result["status"] == "ready"
    assert result["domains"] == ["traffic", "wildlife"]
    assert result["track_count"] == 2


def test_official_semantic_gt_rejects_unreviewed_model_labels(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path / "draft.yaml", "traffic", 1)
    payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    payload["ground_truth_source"] = "model proposal"
    manifest.write_text(yaml.safe_dump(payload), encoding="utf-8")

    result = audit_official_semantic_gt(
        [manifest],
        minimum_domains=1,
        minimum_tracks=1,
    )

    assert result["status"] == "blocked"
    assert "ground_truth_source is not official" in result["issues"]
