from football_tracking.data.bbox import (
    clip_xyxy_to_image,
    is_valid_bbox,
    xywh_to_xyxy,
    xyxy_to_xywh,
    xyxy_to_yolo_normalized,
    yolo_normalized_to_xyxy,
)
from football_tracking.data.schemas import BoundingBoxXYWH, BoundingBoxXYXY


def test_xyxy_to_xywh() -> None:
    assert xyxy_to_xywh(BoundingBoxXYXY(10, 8, 22, 34)) == BoundingBoxXYWH(10, 8, 12, 26)


def test_xyxy_to_yolo_normalized() -> None:
    values = xyxy_to_yolo_normalized(BoundingBoxXYXY(10, 8, 22, 34), 64, 48)

    assert values == (0.25, 0.4375, 0.1875, 0.5416666666666666)


def test_clipping() -> None:
    clipped = clip_xyxy_to_image(BoundingBoxXYXY(-5, 5, 70, 50), 64, 48)

    assert clipped == BoundingBoxXYXY(0, 5, 64, 48)


def test_invalid_box() -> None:
    assert not is_valid_bbox(BoundingBoxXYXY(10, 10, 10, 20))


def test_yolo_round_trip_conversion() -> None:
    source = BoundingBoxXYXY(10, 8, 22, 34)
    yolo = xyxy_to_yolo_normalized(source, 64, 48)
    restored = yolo_normalized_to_xyxy(*yolo, image_width=64, image_height=48)

    assert xywh_to_xyxy(xyxy_to_xywh(restored)) == restored
    assert abs(restored.x1 - source.x1) < 1e-9
    assert abs(restored.y2 - source.y2) < 1e-9
