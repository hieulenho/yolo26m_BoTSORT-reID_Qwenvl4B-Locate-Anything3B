import json
from pathlib import Path

from football_tracking.data.schemas import BoundingBoxXYXY
from football_tracking.detection.schemas import Detection
from football_tracking.detection.serialization import (
    write_predictions_jsonl,
    write_predictions_summary_csv,
    write_run_metadata,
    write_yolo_prediction_labels,
)


def _detection() -> Detection:
    return Detection(
        frame_index=1,
        sequence_name="sequence_001",
        bbox_xyxy=BoundingBoxXYXY(16, 12, 32, 24),
        confidence=0.75,
        source_class_id=0,
        source_class_name="person",
        target_class_id=0,
        target_class_name="player",
        image_width=64,
        image_height=48,
        metadata={"image_path": "image.ppm"},
    )


def test_detection_serialization_outputs_jsonl_yolo_csv_and_metadata(tmp_path: Path) -> None:
    detection = _detection()

    jsonl = write_predictions_jsonl([detection], tmp_path / "predictions.jsonl")
    labels = write_yolo_prediction_labels([detection], tmp_path / "labels")
    csv_path = write_predictions_summary_csv([detection], tmp_path / "summary.csv")
    metadata_path = write_run_metadata({"ok": True, "value": None}, tmp_path / "metadata.json")

    payload = json.loads(jsonl.read_text(encoding="utf-8").splitlines()[0])
    label_fields = labels[0].read_text(encoding="utf-8").split()

    assert payload["sequence_name"] == "sequence_001"
    assert len(label_fields) == 6
    assert label_fields[:5] == ["0", "0.375000", "0.375000", "0.250000", "0.250000"]
    assert "sequence_001" in csv_path.read_text(encoding="utf-8")
    assert json.loads(metadata_path.read_text(encoding="utf-8"))["value"] is None
