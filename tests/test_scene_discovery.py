from __future__ import annotations

from football_tracking.vlm.scene_discovery import (
    _parse_or_fallback,
    _parse_vlm_response,
)


def test_parse_complete_scene_discovery_json() -> None:
    response = '''
    ```json
    {
      "domain": {"name": "traffic", "confidence": 0.9},
      "objects": [{"canonical_name": "car", "action": "track"}],
      "background_regions": ["road"]
    }
    ```
    '''

    parsed = _parse_vlm_response(response)

    assert parsed["domain"]["name"] == "traffic"
    assert parsed["objects"][0]["canonical_name"] == "car"


def test_partial_parser_keeps_complete_objects_before_truncation() -> None:
    response = '''{
      "domain": {
        "name": "urban_intersection",
        "confidence": 0.95,
        "description": "Traffic with a quoted brace: } and escaped quote: \\\"."
      },
      "objects": [
        {"canonical_name": "car", "action": "track", "confidence": 0.9},
        {"canonical_name": "bus", "action": "track", "confidence": 0.8},
        {"canonical_name": "road marking", "action": "context", "attributes": ["whi'''

    parsed, warning = _parse_or_fallback(response)

    assert parsed["domain"]["name"] == "urban_intersection"
    assert [item["canonical_name"] for item in parsed["objects"]] == ["car", "bus"]
    assert parsed["background_regions"] == []
    assert warning is not None
    assert "Recovered complete fields" in warning


def test_partial_parser_falls_back_when_no_complete_evidence_exists() -> None:
    parsed, warning = _parse_or_fallback('{"domain": {"name": "traf')

    assert parsed["domain"]["name"] == "unknown"
    assert parsed["objects"] == []
    assert warning is not None
