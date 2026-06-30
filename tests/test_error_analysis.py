from football_tracking.data.schemas import BoundingBoxXYXY
from football_tracking.detection.error_analysis import analyze_detection_errors
from football_tracking.detection.schemas import Detection


def _detection(box: BoundingBoxXYXY, confidence: float = 0.9) -> Detection:
    return Detection(
        frame_index=1,
        sequence_name="seq",
        bbox_xyxy=box,
        confidence=confidence,
        source_class_id=0,
        source_class_name="person",
        target_class_id=0,
        target_class_name="player",
        image_width=100,
        image_height=100,
    )


def test_error_analysis_counts_tp_fp_fn_and_localization() -> None:
    result = analyze_detection_errors(
        [
            _detection(BoundingBoxXYXY(0, 0, 10, 10)),
            _detection(BoundingBoxXYXY(20, 20, 30, 30)),
            _detection(BoundingBoxXYXY(80, 80, 90, 90)),
        ],
        [BoundingBoxXYXY(0, 0, 10, 10), BoundingBoxXYXY(22, 22, 32, 32)],
    )

    assert result.true_positives == 1
    assert result.localization_errors == 1
    assert result.false_positives == 1
    assert result.false_negatives == 1
