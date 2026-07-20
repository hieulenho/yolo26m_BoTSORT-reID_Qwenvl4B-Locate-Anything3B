from __future__ import annotations

import json
from pathlib import Path

import yaml

from football_tracking.adaptive_tracking.config_builder import (
    build_tracking_payload,
    write_adaptive_plan,
)
from football_tracking.adaptive_tracking.ontology import (
    VocabularyRegistry,
    normalize_objects,
)
from football_tracking.adaptive_tracking.router import build_detector_route
from football_tracking.adaptive_tracking.schemas import SceneDiscovery
from football_tracking.adaptive_tracking.semantic_cache import (
    SemanticCache,
    discovery_cache_key,
)
from football_tracking.adaptive_tracking.shot_sampling import detect_shot_starts


def _registry() -> VocabularyRegistry:
    return VocabularyRegistry.load("configs/ontology/vocabulary_registry.yaml")


def _discovery(domain: str, raw_objects: list[dict]) -> SceneDiscovery:
    return SceneDiscovery(
        source_video="F:/videos/example.mp4",
        domain=domain,
        domain_confidence=0.9,
        description="test scene",
        objects=normalize_objects(raw_objects, registry=_registry()),
    )


def test_vocabulary_merges_aliases_and_preserves_unknown_classes() -> None:
    objects = normalize_objects(
        [
            {"name": "red automobile", "action": "track", "confidence": 0.8},
            {"name": "car", "action": "detect", "confidence": 0.9},
            {"name": "surgical instrument", "action": "track", "confidence": 0.7},
            {"name": "road", "action": "context", "confidence": 0.95},
        ],
        registry=_registry(),
    )

    by_name = {item.canonical_name: item for item in objects}
    assert by_name["car"].action == "track"
    assert by_name["car"].coco_id == 2
    assert "red" in by_name["car"].attributes
    assert by_name["surgical instrument"].open_vocabulary is True
    assert by_name["road"].action == "context"


def test_router_selects_football_coco_and_open_vocabulary_paths() -> None:
    football = build_detector_route(
        _discovery(
            "football",
            [
                {"name": "player", "action": "track", "confidence": 0.95},
                {"name": "sports ball", "action": "detect", "confidence": 0.8},
            ],
        )
    )
    traffic = build_detector_route(
        _discovery(
            "traffic",
            [
                {"name": "car", "action": "track", "confidence": 0.9},
                {"name": "bus", "action": "track", "confidence": 0.8},
            ],
        )
    )
    medical = build_detector_route(
        _discovery(
            "medical",
            [
                {
                    "name": "surgical instrument",
                    "action": "track",
                    "confidence": 0.8,
                }
            ],
        )
    )

    assert football.route_name == "football_finetuned"
    assert football.checkpoint.endswith("yolo26m_best.pt")
    assert football.deferred_classes == ("sports ball",)
    assert traffic.route_name == "coco_pretrained"
    assert traffic.class_ids == (2, 5)
    assert medical.route_name == "open_vocabulary"
    assert medical.backend == "ultralytics_yoloe"
    assert medical.class_names == ("surgical instrument",)


def test_generated_plan_uses_deepocsort_and_runtime_vocabulary(tmp_path: Path) -> None:
    discovery = _discovery(
        "medical",
        [{"name": "wheelchair", "action": "track", "confidence": 0.9}],
    )
    route = build_detector_route(discovery)
    payload = build_tracking_payload(
        source_video=tmp_path / "source.mp4",
        output_video=tmp_path / "tracked.mp4",
        route=route,
        overwrite=True,
    )
    paths = write_adaptive_plan(
        output_dir=tmp_path / "run",
        discovery=discovery,
        route=route,
        tracking_payload=payload,
        overwrite=True,
    )

    generated = yaml.safe_load(paths["tracking_config"].read_text(encoding="utf-8"))
    plan = json.loads(paths["plan"].read_text(encoding="utf-8"))
    assert generated["tracker"]["name"] == "deepocsort_reid"
    assert generated["model"]["backend"] == "ultralytics_yoloe"
    assert generated["model"]["text_classes"] == ["wheelchair"]
    assert plan["locateanything_policy"]["mode"] == "event_triggered"


def test_semantic_cache_key_changes_with_model_or_video(tmp_path: Path) -> None:
    video = tmp_path / "video.bin"
    video.write_bytes(b"video-content")
    key_a = discovery_cache_key(
        video,
        model_id="qwen-a",
        prompt_version="v2",
        sampling={"frames": 3},
    )
    key_b = discovery_cache_key(
        video,
        model_id="qwen-b",
        prompt_version="v2",
        sampling={"frames": 3},
    )
    cache = SemanticCache(tmp_path / "cache")
    discovery = _discovery("traffic", [{"name": "car", "confidence": 0.9}])
    cache.save(key_a, discovery)

    assert key_a != key_b
    assert cache.load(key_a) == discovery
    assert cache.load(key_b) is None


def test_shot_start_detection_respects_threshold_and_minimum_gap() -> None:
    starts = detect_shot_starts(
        [1, 31, 61, 91, 121],
        [0.0, 0.8, 0.9, 0.2, 0.7],
        threshold=0.6,
        min_gap_frames=60,
    )
    assert starts == [1, 61, 121]
