"""Parser for LocateAnything phrase-grounding text outputs."""

from __future__ import annotations

import re
from dataclasses import dataclass

from football_tracking.locate_tracking.grounding.coordinates import (
    CoordinateError,
    normalized_to_pixel_xyxy,
    validate_normalized_xyxy,
)
from football_tracking.locate_tracking.grounding.schemas import GroundedBox

_REF_BOX_PATTERN = re.compile(
    r"<ref>\s*(?P<label>.*?)\s*</ref>\s*<box>\s*(?P<box>.*?)\s*</box>",
    re.IGNORECASE | re.DOTALL,
)
_BOX_PATTERN = re.compile(r"<box>\s*(?P<box>.*?)\s*</box>", re.IGNORECASE | re.DOTALL)
_ANGLE_NUMBER_PATTERN = re.compile(r"<\s*(-?\d+(?:\.\d+)?)\s*>")
_PLAIN_NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass(frozen=True)
class ParseResult:
    boxes: tuple[GroundedBox, ...]
    errors: tuple[str, ...]


def _box_span_from_ref_match(match: re.Match[str]) -> tuple[int, int]:
    text = match.group(0)
    box_start = text.lower().find("<box>")
    box_end = text.lower().rfind("</box>")
    if box_start < 0 or box_end < 0:
        return match.span()
    return match.start() + box_start, match.start() + box_end + len("</box>")


def _span_is_consumed(span: tuple[int, int], consumed_spans: list[tuple[int, int]]) -> bool:
    start, end = span
    return any(start >= used_start and end <= used_end for used_start, used_end in consumed_spans)


def _parse_normalized_box(content: str) -> tuple[int, int, int, int] | None:
    if content.strip().lower() == "none":
        return None
    matches = _ANGLE_NUMBER_PATTERN.findall(content)
    if not matches:
        matches = _PLAIN_NUMBER_PATTERN.findall(content)
    if len(matches) != 4:
        raise ValueError(f"Expected four box coordinates, found {len(matches)}.")
    return validate_normalized_xyxy([float(item) for item in matches])


def _label_or_query(label: str | None, query: str) -> str:
    cleaned = (label or "").strip()
    return cleaned if cleaned else query


def _parse_box_match(
    *,
    label: str | None,
    box_content: str,
    query: str,
    image_width: int,
    image_height: int,
    index: int,
) -> GroundedBox | None:
    normalized = _parse_normalized_box(box_content)
    if normalized is None:
        return None
    pixel_bbox = normalized_to_pixel_xyxy(normalized, image_width, image_height)
    return GroundedBox(
        label=_label_or_query(label, query),
        bbox_xyxy=pixel_bbox,
        normalized_bbox=normalized,
        confidence=None,
        query=query,
        metadata={
            "source": "locate_anything_parser",
            "box_index": index,
        },
    )


def parse_locate_anything_response(
    raw_response: str,
    *,
    query: str,
    image_width: int,
    image_height: int,
) -> ParseResult:
    """Parse LocateAnything text into validated grounded boxes.

    Invalid boxes are reported through ``errors`` and never converted into
    detections.  ``<box>none</box>`` is treated as a valid no-object response.
    """

    text = raw_response or ""
    boxes: list[GroundedBox] = []
    errors: list[str] = []
    consumed_box_spans: list[tuple[int, int]] = []

    for match in _REF_BOX_PATTERN.finditer(text):
        consumed_box_spans.append(_box_span_from_ref_match(match))
        try:
            parsed = _parse_box_match(
                label=match.group("label"),
                box_content=match.group("box"),
                query=query,
                image_width=image_width,
                image_height=image_height,
                index=len(boxes),
            )
        except (CoordinateError, ValueError) as exc:
            errors.append(f"Invalid LocateAnything box for label '{match.group('label')}': {exc}")
            continue
        if parsed is not None:
            boxes.append(parsed)

    for match in _BOX_PATTERN.finditer(text):
        if _span_is_consumed(match.span(), consumed_box_spans):
            continue
        try:
            parsed = _parse_box_match(
                label=None,
                box_content=match.group("box"),
                query=query,
                image_width=image_width,
                image_height=image_height,
                index=len(boxes),
            )
        except (CoordinateError, ValueError) as exc:
            errors.append(f"Invalid LocateAnything box: {exc}")
            continue
        if parsed is not None:
            boxes.append(parsed)

    if "<box" not in text.lower():
        errors.append("LocateAnything response did not contain any <box> tags.")
    return ParseResult(boxes=tuple(boxes), errors=tuple(errors))
