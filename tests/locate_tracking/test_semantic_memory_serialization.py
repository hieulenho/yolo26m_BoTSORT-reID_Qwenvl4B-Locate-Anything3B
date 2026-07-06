from __future__ import annotations

import json
from pathlib import Path

from football_tracking.locate_tracking.semantic_memory.aggregator import (
    build_semantic_memory,
)
from football_tracking.locate_tracking.semantic_memory.decision_policy import (
    decide_final_resolution,
)
from football_tracking.locate_tracking.semantic_memory.schemas import SemanticMemoryConfig
from football_tracking.locate_tracking.semantic_memory.serialization import (
    load_final_resolution,
    load_frame_resolution,
    load_semantic_memory,
    save_final_resolution,
    save_semantic_memory,
)
from tests.locate_tracking.semantic_test_utils import resolved_frame


def test_semantic_memory_and_final_resolution_round_trip(tmp_path: Path) -> None:
    config = SemanticMemoryConfig(min_usable_frames=1, min_support_frames=1)
    memory = build_semantic_memory(
        query="player",
        frame_resolutions=(resolved_frame(1, 7),),
        config=config,
    )
    final = decide_final_resolution(memory, config)

    memory_path = save_semantic_memory(memory, tmp_path / "semantic_memory.json")
    final_path = save_final_resolution(final, tmp_path / "final_resolution.json")

    assert load_semantic_memory(memory_path).to_dict() == memory.to_dict()
    assert load_final_resolution(final_path).to_dict() == final.to_dict()


def test_load_frame_resolution_validates_required_keys(tmp_path: Path) -> None:
    path = tmp_path / "association.json"
    path.write_text(json.dumps(resolved_frame(1, 7).to_dict()), encoding="utf-8")

    assert load_frame_resolution(path)["frame_index"] == 1
