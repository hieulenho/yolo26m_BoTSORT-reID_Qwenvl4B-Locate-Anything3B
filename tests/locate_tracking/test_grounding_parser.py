from __future__ import annotations

from football_tracking.locate_tracking.grounding.parser import (
    parse_locate_anything_response,
)


def _parse(raw: str, width: int = 1000, height: int = 500):
    return parse_locate_anything_response(
        raw,
        query="goalkeeper wearing green",
        image_width=width,
        image_height=height,
    )


def test_parser_one_valid_box() -> None:
    result = _parse("<ref>goalkeeper</ref><box><100><200><500><800></box>")

    assert not result.errors
    assert len(result.boxes) == 1
    assert result.boxes[0].label == "goalkeeper"
    assert result.boxes[0].bbox_xyxy == (100.0, 100.0, 500.0, 400.0)


def test_parser_multiple_boxes() -> None:
    result = _parse(
        "<ref>player</ref><box><100><200><300><800></box>"
        "<ref>player</ref><box><400><220><600><850></box>"
    )

    assert not result.errors
    assert len(result.boxes) == 2
    assert [box.label for box in result.boxes] == ["player", "player"]


def test_parser_no_object_output() -> None:
    result = _parse("<box>none</box>")

    assert result.boxes == ()
    assert result.errors == ()


def test_parser_malformed_output() -> None:
    result = _parse("I cannot find it.")

    assert result.boxes == ()
    assert result.errors


def test_parser_incomplete_coordinates() -> None:
    result = _parse("<ref>player</ref><box><100><200><300></box>")

    assert result.boxes == ()
    assert "Expected four" in result.errors[0]


def test_parser_inverted_box() -> None:
    result = _parse("<ref>player</ref><box><500><200><100><800></box>")

    assert result.boxes == ()
    assert "x2 > x1" in result.errors[0]


def test_parser_out_of_range_coordinates() -> None:
    result = _parse("<ref>player</ref><box><100><200><1200><800></box>")

    assert result.boxes == ()
    assert "[0, 1000]" in result.errors[0]


def test_parser_border_coordinates() -> None:
    result = _parse("<ref>pitch</ref><box><0><0><1000><1000></box>", width=640, height=480)

    assert not result.errors
    assert result.boxes[0].bbox_xyxy == (0.0, 0.0, 640.0, 480.0)


def test_parser_preserves_label() -> None:
    result = _parse("<ref>goalkeeper</ref><box><10><20><30><40></box>")

    assert result.boxes[0].label == "goalkeeper"


def test_parser_extra_surrounding_text() -> None:
    result = _parse(
        "The object is here: <ref>goalkeeper</ref><box><100><200><500><800></box>."
    )

    assert len(result.boxes) == 1
    assert not result.errors

