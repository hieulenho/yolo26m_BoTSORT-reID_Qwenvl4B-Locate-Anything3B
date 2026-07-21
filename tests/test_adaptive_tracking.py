from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from football_tracking.adaptive_tracking.config_builder import (
    build_tracker_routing_payload,
    build_tracking_payload,
    write_adaptive_plan,
)
from football_tracking.adaptive_tracking.grounding_verification import (
    _map_bbox_to_source,
    _prepare_grounding_input,
    build_grounding_plan,
)
from football_tracking.adaptive_tracking.ontology import (
    VocabularyRegistry,
    VocabularyRegistryError,
    normalize_objects,
)
from football_tracking.adaptive_tracking.router import build_detector_route
from football_tracking.adaptive_tracking.schemas import SceneDiscovery
from football_tracking.adaptive_tracking.semantic_cache import (
    SemanticCache,
    discovery_cache_key,
)
from football_tracking.adaptive_tracking.semantic_fusion import (
    TrackSemanticEvidence,
    fuse_track_semantics,
    normalize_semantic_evidence,
    parse_locate_evidence,
    parse_qwen_answer,
)
from football_tracking.adaptive_tracking.semantic_queue import (
    SemanticCacheView,
    SemanticEventQueue,
    process_semantic_queue,
)
from football_tracking.adaptive_tracking.semantic_render import (
    _select_fitting_text,
    render_semantic_video,
)
from football_tracking.adaptive_tracking.shot_sampling import (
    OnlineShotChangeDetector,
    _select_representative_candidates,
    detect_shot_starts,
)
from football_tracking.adaptive_tracking.temporal_memory import TemporalSemanticMemory
from football_tracking.data.schemas import BoundingBoxXYXY
from football_tracking.tracking.schemas import TrackOutput
from football_tracking.vlm.tracking_context import MotTrackRow


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


def test_vocabulary_registry_rejects_conflicting_aliases(tmp_path: Path) -> None:
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        """
classes:
  - name: car
    aliases: [vehicle]
    coco_class: car
    default_action: track
  - name: truck
    aliases: [vehicle]
    coco_class: truck
    default_action: track
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(VocabularyRegistryError, match="maps to both"):
        VocabularyRegistry.load(registry)


def test_vocabulary_registry_rejects_unknown_coco_mapping(tmp_path: Path) -> None:
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        """
classes:
  - name: vehicle
    coco_class: imaginary vehicle
    default_action: track
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(VocabularyRegistryError, match="Unknown COCO class"):
        VocabularyRegistry.load(registry)


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
    assert football.class_names == ("player", "sports ball")
    assert football.class_ids == (0, 32)
    assert football.tracker_class_ids == (0,)
    assert football.supplemental_detectors[0]["backend"] == "ultralytics"
    assert football.supplemental_detectors[0]["every_n_frames"] == 6
    assert traffic.route_name == "coco_pretrained"
    assert traffic.class_ids == (2, 5)
    assert traffic.tracker_class_ids == (2, 5)
    assert medical.route_name == "open_vocabulary"
    assert medical.backend == "ultralytics_yoloe"
    assert medical.class_names == ("surgical instrument",)


def test_realtime_stable_profile_keeps_small_detector_and_uses_tracktrack() -> None:
    discovery = _discovery(
        "traffic",
        [{"name": "car", "action": "track", "confidence": 0.9}],
    )
    route = build_detector_route(discovery, profile="realtime_stable")
    routing = build_tracker_routing_payload(route)
    tracking = build_tracking_payload(
        source_video="F:/videos/traffic.mp4",
        output_video="F:/videos/traffic_tracking.mp4",
        route=route,
    )

    assert route.checkpoint == "yolo26n.pt"
    assert routing["tracker"]["default"]["name"] == "tracktrack"
    assert tracking["detector"]["imgsz"] == 640


def test_router_promotes_highest_confidence_detect_class_when_no_track_class() -> None:
    route = build_detector_route(
        _discovery(
            "wildlife",
            [
                {"name": "kingfisher", "action": "detect", "confidence": 0.98},
                {"name": "branch", "action": "detect", "confidence": 0.95},
            ],
        )
    )

    assert route.route_name == "open_vocabulary"
    assert route.tracker_class_ids == (0,)
    assert "promoted" in route.reason


def test_router_promotes_people_instead_of_classroom_furniture() -> None:
    route = build_detector_route(
        _discovery(
            "education",
            [
                {"name": "desk", "action": "detect", "confidence": 0.99},
                {"name": "student", "action": "detect", "confidence": 0.9},
                {"name": "teacher", "action": "detect", "confidence": 0.85},
                {"name": "book", "action": "detect", "confidence": 0.8},
            ],
        )
    )

    assert route.route_name == "coco_open_composite"
    assert route.tracker_class_ids == (0,)
    assert "desk" not in route.reason.split("were promoted", maxsplit=1)[0]


def test_router_recognizes_football_from_scene_description() -> None:
    discovery = SceneDiscovery(
        source_video="F:/videos/example.mp4",
        domain="sports",
        domain_confidence=0.95,
        description="A professional football match in a stadium.",
        objects=normalize_objects(
            [
                {"name": "player", "action": "track", "confidence": 0.95},
                {"name": "goalkeeper", "action": "track", "confidence": 0.9},
                {"name": "referee", "action": "track", "confidence": 0.9},
            ],
            registry=_registry(),
        ),
    )

    route = build_detector_route(discovery)

    assert route.route_name == "football_finetuned"
    assert route.class_ids == (0,)


def test_coco_router_merges_semantic_roles_with_the_same_detector_id() -> None:
    discovery = _discovery(
        "education",
        [
            {"name": "student", "action": "track", "confidence": 0.95},
            {"name": "teacher", "action": "track", "confidence": 0.9},
            {"name": "chair", "action": "detect", "confidence": 0.8},
        ],
    )

    route = build_detector_route(discovery)

    assert route.class_ids == (0, 56)
    assert route.class_names == ("person", "chair")
    assert route.tracker_class_ids == (0,)


def test_router_does_not_defer_a_non_person_football_tracking_class() -> None:
    route = build_detector_route(
        _discovery(
            "football",
            [
                {"name": "player", "action": "track", "confidence": 0.95},
                {"name": "sports ball", "action": "track", "confidence": 0.8},
            ],
        )
    )

    assert route.route_name == "football_finetuned"
    assert route.class_ids == (0, 32)
    assert route.tracker_class_ids == (0, 32)
    assert route.primary_class_ids == (0,)
    assert route.supplemental_detectors[0]["every_n_frames"] == 3


def test_route_keeps_detect_only_classes_out_of_tracker() -> None:
    route = build_detector_route(
        _discovery(
            "traffic",
            [
                {"name": "car", "action": "track", "confidence": 0.9},
                {"name": "traffic light", "action": "detect", "confidence": 0.9},
            ],
        )
    )

    assert route.class_ids == (2, 9)
    assert route.tracker_class_ids == (2,)


def test_generated_plan_uses_profile_tracker_and_runtime_vocabulary(tmp_path: Path) -> None:
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
    assert generated["tracker"]["name"] == "adaptive_routed"
    assert generated["tracker"]["config"].endswith("tracker_routing.generated.yaml")
    tracker_routing = yaml.safe_load(paths["tracker_routing"].read_text(encoding="utf-8"))
    assert tracker_routing["tracker"]["default"]["name"] == "ocsort"
    assert tracker_routing["tracker"]["routes"][0]["route_name"] == "open_vocabulary"
    assert tracker_routing["tracker"]["routes"][0]["class_names"] == ["wheelchair"]
    assert generated["model"]["backend"] == "ultralytics_yoloe"
    assert generated["model"]["text_classes"] == ["wheelchair"]
    assert plan["locateanything_policy"]["mode"] == "event_triggered"


def test_football_supplemental_detector_preserves_routed_class_ids(tmp_path: Path) -> None:
    discovery = _discovery(
        "football",
        [
            {"name": "player", "action": "track", "confidence": 0.95},
            {"name": "sports ball", "action": "detect", "confidence": 0.9},
        ],
    )

    payload = build_tracking_payload(
        source_video=tmp_path / "source.mp4",
        output_video=tmp_path / "tracked.mp4",
        route=build_detector_route(discovery),
    )

    assert payload["detector"]["preserve_source_classes"] is True
    assert payload["detector"]["class_ids"] == [0]
    assert payload["detector"]["tracker_class_ids"] == [0]
    assert payload["detector"]["source_class_names"] == {
        "0": "player",
        "32": "sports ball",
    }
    supplemental = payload["model"]["supplemental_detectors"][0]
    assert supplemental["input_class_ids"] == [32]
    assert supplemental["output_class_ids"] == [32]


def test_router_composes_coco_and_open_vocabulary_detectors() -> None:
    route = build_detector_route(
        _discovery(
            "traffic",
            [
                {"name": "car", "action": "track", "confidence": 0.95},
                {
                    "name": "delivery robot",
                    "action": "track",
                    "confidence": 0.85,
                },
                {
                    "name": "traffic light",
                    "action": "detect",
                    "confidence": 0.8,
                },
            ],
        )
    )

    assert route.route_name == "coco_open_composite"
    assert route.primary_class_ids == (2, 9)
    assert route.class_ids == (2, 9, 1000)
    assert route.tracker_class_ids == (2, 1000)
    assert route.supplemental_detectors[0]["backend"] == "ultralytics_yoloe"
    assert route.supplemental_detectors[0]["text_classes"] == ["delivery robot"]


def test_balanced_profile_uses_tracktrack(tmp_path: Path) -> None:
    discovery = _discovery(
        "traffic",
        [{"name": "car", "action": "track", "confidence": 0.9}],
    )
    route = build_detector_route(discovery, profile="balanced")

    payload = build_tracking_payload(
        source_video=tmp_path / "source.mp4",
        output_video=tmp_path / "tracked.mp4",
        route=route,
    )

    assert payload["tracker"]["name"] == "adaptive_routed"
    routing = build_tracker_routing_payload(route)
    assert routing["tracker"]["default"]["name"] == "tracktrack"
    assert routing["tracker"]["default"]["class_ids"] == list(
        route.tracker_class_ids
    )
    assert routing["tracker"]["routes"] == []


def test_accuracy_profile_uses_identity_stable_botsort(tmp_path: Path) -> None:
    discovery = _discovery(
        "traffic",
        [{"name": "car", "action": "track", "confidence": 0.9}],
    )
    route = build_detector_route(discovery, profile="accuracy")

    payload = build_tracking_payload(
        source_video=tmp_path / "source.mp4",
        output_video=tmp_path / "tracked.mp4",
        route=route,
    )

    assert payload["tracker"]["name"] == "adaptive_routed"
    routing = build_tracker_routing_payload(route)
    assert routing["tracker"]["default"]["name"] == "botsort_reid"
    assert routing["tracker"]["default"]["config"].endswith("botsort_reid_identity_stable.yaml")


def test_small_fast_class_receives_motion_specific_tracker() -> None:
    route = build_detector_route(
        _discovery(
            "football",
            [
                {"name": "player", "action": "track", "confidence": 0.95},
                {"name": "sports ball", "action": "track", "confidence": 0.9},
            ],
        )
    )

    routing = build_tracker_routing_payload(route)
    assert routing["tracker"]["default"]["class_ids"] == [0]
    assert len(routing["tracker"]["routes"]) == 1
    small_fast = routing["tracker"]["routes"][0]
    assert small_fast["route_name"] == "small_fast"
    assert small_fast["class_names"] == ["sports ball"]
    assert small_fast["config"].endswith("ocsort_small_fast.yaml")


def test_regular_classes_share_one_default_motion_route() -> None:
    route = build_detector_route(
        _discovery(
            "traffic",
            [
                {"name": "car", "action": "track", "confidence": 0.95},
                {"name": "bus", "action": "track", "confidence": 0.9},
                {"name": "truck", "action": "track", "confidence": 0.9},
            ],
        )
    )

    routing = build_tracker_routing_payload(route)

    assert routing["tracker"]["routes"] == []
    assert routing["tracker"]["default"]["class_ids"] == list(
        route.tracker_class_ids
    )


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


def test_semantic_cache_ignores_corrupt_or_mismatched_entries(tmp_path: Path) -> None:
    cache = SemanticCache(tmp_path / "cache")
    path = cache.path_for("a" * 64)
    path.parent.mkdir(parents=True)
    path.write_text("{broken", encoding="utf-8")
    assert cache.load("a" * 64) is None

    path.write_text(
        json.dumps(
            {
                "cache_key": "b" * 64,
                "discovery": _discovery("traffic", [{"name": "car", "confidence": 0.9}]).to_dict(),
            }
        ),
        encoding="utf-8",
    )
    assert cache.load("a" * 64) is None


def test_shot_start_detection_respects_threshold_and_minimum_gap() -> None:
    starts = detect_shot_starts(
        [1, 31, 61, 91, 121],
        [0.0, 0.8, 0.9, 0.2, 0.7],
        threshold=0.6,
        min_gap_frames=60,
    )
    assert starts == [1, 61, 121]


def test_single_long_shot_fills_keyframe_budget_with_temporal_coverage() -> None:
    candidates = [
        {
            "frame_index": frame_index,
            "quality": quality,
            "transition": 0.0,
            "shot_index": 0,
        }
        for frame_index, quality in ((1, 1.0), (101, 2.0), (201, 4.0), (301, 3.0))
    ]

    selected = _select_representative_candidates(
        candidates,
        [candidates],
        max_keyframes=4,
        frame_count=301,
    )

    assert [row["frame_index"] for row in selected] == [1, 101, 201, 301]


def test_grounding_plan_is_event_triggered_for_open_or_uncertain_classes(
    tmp_path: Path,
) -> None:
    discovery = SceneDiscovery(
        source_video="video.mp4",
        domain="medical",
        domain_confidence=0.8,
        description="medical scene",
        objects=normalize_objects(
            [
                {"name": "person", "confidence": 0.9, "action": "track"},
                {
                    "name": "surgical instrument",
                    "confidence": 0.8,
                    "action": "track",
                },
            ],
            registry=_registry(),
        ),
        keyframes=(
            {"frame_index": 10, "path": "frame10.jpg"},
            {"frame_index": 20, "path": "frame20.jpg"},
        ),
    )

    plan = build_grounding_plan(
        discovery,
        output_path=tmp_path / "grounding_plan.json",
        overwrite=True,
    )

    assert plan["summary"]["request_count"] == 2
    assert {row["class_label"] for row in plan["requests"]} == {"surgical instrument"}
    assert all(row["trigger"] == "open_vocabulary_class" for row in plan["requests"])


def test_grounding_plan_adds_low_confidence_qwen_track_verification(
    tmp_path: Path,
) -> None:
    discovery = SceneDiscovery(
        source_video="video.mp4",
        domain="traffic",
        domain_confidence=0.9,
        description="traffic",
        objects=normalize_objects(
            [{"name": "car", "confidence": 0.95, "action": "track"}],
            registry=_registry(),
        ),
        keyframes=({"frame_index": 10, "path": "frame10.jpg"},),
    )
    qwen_answer = tmp_path / "vlm_answer.json"
    qwen_answer.write_text(
        json.dumps(
            {
                "answer": {
                    "track_predictions": [
                        {
                            "track_id": 7,
                            "class_label": "ambulance",
                            "confidence": 0.5,
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    plan = build_grounding_plan(
        discovery,
        output_path=tmp_path / "grounding_plan.json",
        qwen_answer=qwen_answer,
        overwrite=True,
    )

    assert plan["summary"]["request_count"] == 1
    assert plan["requests"][0]["class_label"] == "ambulance"
    assert plan["requests"][0]["expected_track_ids"] == [7]
    assert plan["requests"][0]["trigger"] == "low_qwen_semantic_confidence"


def test_grounding_plan_uses_discovered_classes_for_unknown_qwen_tracks(
    tmp_path: Path,
) -> None:
    discovery = SceneDiscovery(
        source_video="video.mp4",
        domain="education",
        domain_confidence=0.9,
        description="classroom",
        objects=normalize_objects(
            [
                {"name": "student", "confidence": 0.95, "action": "track"},
                {"name": "teacher", "confidence": 0.9, "action": "track"},
            ],
            registry=_registry(),
        ),
        keyframes=({"frame_index": 10, "path": "frame10.jpg"},),
    )
    qwen_answer = tmp_path / "vlm_answer.json"
    qwen_answer.write_text(
        json.dumps(
            {
                "answer": {
                    "track_predictions": [
                        {
                            "track_id": 12,
                            "class_label": "unknown",
                            "confidence": 0.0,
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    plan = build_grounding_plan(
        discovery,
        output_path=tmp_path / "grounding_plan.json",
        qwen_answer=qwen_answer,
        overwrite=True,
    )

    assert plan["summary"]["uncertain_track_count"] == 1
    assert plan["summary"]["request_count"] == 2
    assert {row["class_label"] for row in plan["requests"]} == {
        "student",
        "teacher",
    }
    assert all(row["expected_track_ids"] == [12] for row in plan["requests"])


def test_grounding_plan_can_benchmark_explicit_tracks_without_qwen(
    tmp_path: Path,
) -> None:
    discovery = SceneDiscovery(
        source_video="video.mp4",
        domain="sports",
        domain_confidence=0.9,
        description="football",
        objects=normalize_objects(
            [
                {"name": "player", "confidence": 0.95, "action": "track"},
                {"name": "referee", "confidence": 0.9, "action": "track"},
            ],
            registry=_registry(),
        ),
        keyframes=({"frame_index": 5, "path": "discovery.jpg"},),
    )
    context = tmp_path / "vlm_context.json"
    context.write_text(
        json.dumps(
            {
                "keyframes": [],
                "crops": [
                    {"track_id": 7, "frame_index": 30, "path": "track7.jpg"},
                    {"track_id": 9, "frame_index": 40, "path": "track9.jpg"},
                ],
            }
        ),
        encoding="utf-8",
    )

    plan = build_grounding_plan(
        discovery,
        output_path=tmp_path / "grounding_plan.json",
        semantic_context=context,
        verify_track_ids=[7, 9],
        max_expected_tracks_per_class=2,
        overwrite=True,
    )

    assert plan["policy"] == "explicit_track_benchmark"
    assert plan["summary"]["explicit_track_count"] == 2
    assert plan["summary"]["request_count"] == 4
    assert {row["frame_index"] for row in plan["requests"]} == {30, 40}
    assert {tuple(row["expected_track_ids"]) for row in plan["requests"]} == {
        (7,),
        (9,),
    }


def test_grounding_plan_triggers_identity_reacquisition_for_long_gap(
    tmp_path: Path,
) -> None:
    discovery = _discovery(
        "traffic",
        [{"name": "car", "confidence": 0.95, "action": "track"}],
    )
    context = tmp_path / "vlm_context.json"
    context.write_text(
        json.dumps(
            {
                "keyframes": [],
                "crops": [{"track_id": 17, "frame_index": 90, "path": "track17.jpg"}],
                "tracking_diagnostics": {
                    "fragmented_tracks": [{"track_id": 17, "gap_count": 2, "max_gap_frames": 24}]
                },
            }
        ),
        encoding="utf-8",
    )

    plan = build_grounding_plan(
        discovery,
        output_path=tmp_path / "plan.json",
        semantic_context=context,
        overwrite=True,
    )

    assert plan["summary"]["reacquisition_track_count"] == 1
    assert plan["requests"][0]["trigger"] == "identity_reacquisition"
    assert plan["requests"][0]["expected_track_ids"] == [17]
    assert plan["requests"][0]["frame_index"] == 90


def test_grounding_plan_targets_unknown_track_on_annotated_keyframe(
    tmp_path: Path,
) -> None:
    discovery = SceneDiscovery(
        source_video="video.mp4",
        domain="traffic",
        domain_confidence=0.9,
        description="traffic",
        objects=normalize_objects(
            [{"name": "car", "confidence": 0.95, "action": "track"}],
            registry=_registry(),
        ),
        keyframes=({"frame_index": 5, "path": "discovery.jpg"},),
    )
    qwen_root = tmp_path / "qwen"
    qwen_root.mkdir()
    (qwen_root / "vlm_answer.json").write_text(
        json.dumps(
            {
                "answer": {
                    "track_predictions": [
                        {"track_id": 12, "class_label": "unknown", "confidence": 0.0}
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    (qwen_root / "vlm_context.json").write_text(
        json.dumps(
            {
                "keyframes": [
                    {
                        "frame_index": 10,
                        "path": "annotated.jpg",
                        "visible_track_ids": [12, 15],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    plan = build_grounding_plan(
        discovery,
        output_path=tmp_path / "grounding_plan.json",
        qwen_answer=qwen_root / "vlm_answer.json",
        overwrite=True,
    )

    request = plan["requests"][0]
    assert request["frame_index"] == 10
    assert request["image_path"] == "annotated.jpg"
    assert request["target_track_id"] == 12
    assert "ID 12" in request["query"]


def test_target_grounding_crop_maps_local_box_back_to_source(tmp_path: Path) -> None:
    request = {
        "request_id": "track_7",
        "frame_index": 10,
        "image_path": str(tmp_path / "unused.jpg"),
        "query": "the car inside the tracking box labeled ID 7",
        "localized_query": "the car",
        "target_track_id": 7,
    }
    row = MotTrackRow(10, 7, 20, 30, 10, 20, 0.9)

    prepared = _prepare_grounding_input(
        request=request,
        source_video=None,
        frame_rows=[row],
        output_dir=tmp_path / "crops",
        frame_cache={10: np.zeros((100, 100, 3), dtype=np.uint8)},
        crop_padding=0.0,
        crop_size=100,
    )

    assert prepared["mode"] == "target_crop"
    assert prepared["query"] == "the car"
    assert Path(prepared["image_path"]).is_file()
    assert prepared["roi"]["scale"] == 5.0
    assert _map_bbox_to_source((0.0, 0.0, 50.0, 100.0), prepared["roi"]) == (
        20.0,
        30.0,
        30.0,
        50.0,
    )


def test_grounding_plan_uses_track_crop_frame_when_keyframe_misses_target(
    tmp_path: Path,
) -> None:
    discovery = _discovery(
        "traffic",
        [{"name": "car", "confidence": 0.95, "action": "track"}],
    )
    qwen_root = tmp_path / "qwen"
    qwen_root.mkdir()
    (qwen_root / "vlm_answer.json").write_text(
        json.dumps(
            {
                "answer": {
                    "track_predictions": [
                        {"track_id": 99, "class_label": "unknown", "confidence": 0.0}
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    (qwen_root / "vlm_context.json").write_text(
        json.dumps(
            {
                "keyframes": [{"frame_index": 5, "path": "keyframe.jpg", "visible_track_ids": [1]}],
                "crops": [{"track_id": 99, "frame_index": 50, "path": "track99.jpg"}],
            }
        ),
        encoding="utf-8",
    )

    plan = build_grounding_plan(
        discovery,
        output_path=tmp_path / "plan.json",
        qwen_answer=qwen_root / "vlm_answer.json",
        overwrite=True,
    )

    request = plan["requests"][0]
    assert request["frame_index"] == 50
    assert request["target_track_id"] == 99


def test_semantic_fusion_accepts_consensus_and_rejects_conflict() -> None:
    evidence = [
        TrackSemanticEvidence(1, "car", 0.9, "qwen", {"color": "red"}),
        TrackSemanticEvidence(1, "car", 0.8, "locateanything"),
        TrackSemanticEvidence(2, "car", 0.6, "qwen"),
        TrackSemanticEvidence(2, "bus", 0.6, "locateanything"),
    ]

    result = fuse_track_semantics(
        evidence,
        unknown_threshold=0.45,
        minimum_margin=0.10,
    )

    by_id = {row["track_id"]: row for row in result["tracks"]}
    assert by_id[1]["class_label"] == "car"
    assert by_id[1]["attributes"] == {"color": "red"}
    assert by_id[2]["class_label"] == "unknown"
    assert result["summary"]["accepted_count"] == 1


def test_hierarchical_fusion_accepts_supported_species() -> None:
    evidence = parse_qwen_answer(
        {
            "answer": {
                "track_predictions": [
                    {
                        "track_id": 11,
                        "class_label": "bird",
                        "fine_label": "common kingfisher",
                        "taxonomy_path": ["animal", "bird", "common kingfisher"],
                        "confidence": 0.95,
                        "fine_confidence": 0.88,
                        "observations": [
                            {
                                "frame_index": 10,
                                "class_label": "bird",
                                "fine_label": "common kingfisher",
                                "confidence": 0.94,
                                "fine_confidence": 0.86,
                            },
                            {
                                "frame_index": 30,
                                "class_label": "bird",
                                "fine_label": "common kingfisher",
                                "confidence": 0.96,
                                "fine_confidence": 0.90,
                            },
                        ],
                    }
                ]
            }
        }
    )

    row = fuse_track_semantics(evidence)["tracks"][0]

    assert row["class_label"] == "bird"
    assert row["accepted"] is True
    assert row["fine_label"] == "common kingfisher"
    assert row["fine_accepted"] is True
    assert row["display_label"] == "bird > common kingfisher"
    assert row["taxonomy_path"] == ["animal", "bird", "common kingfisher"]


def test_hierarchical_fusion_rejects_conflicting_subtype_only() -> None:
    result = fuse_track_semantics(
        [
            TrackSemanticEvidence(
                12,
                "car",
                0.95,
                "qwen",
                evidence_frames=(10,),
                fine_label="sedan",
                fine_confidence=0.8,
            ),
            TrackSemanticEvidence(
                12,
                "car",
                0.93,
                "qwen",
                evidence_frames=(20,),
                fine_label="hatchback",
                fine_confidence=0.8,
            ),
        ]
    )

    row = result["tracks"][0]
    assert row["class_label"] == "car"
    assert row["accepted"] is True
    assert row["fine_label"] == "unknown"
    assert row["fine_accepted"] is False
    assert row["display_label"] == "car"


def test_hierarchical_fusion_rejects_under_threshold_subtype() -> None:
    result = fuse_track_semantics(
        [
            TrackSemanticEvidence(
                5,
                "bus",
                0.9,
                "qwen",
                evidence_frames=(10,),
                fine_label="articulated",
                fine_confidence=0.843,
            ),
            TrackSemanticEvidence(
                5,
                "bus",
                0.9,
                "qwen",
                evidence_frames=(30,),
                fine_label="articulated",
                fine_confidence=0.843,
            ),
        ]
    )

    row = result["tracks"][0]
    assert row["class_label"] == "bus"
    assert row["accepted"] is True
    assert row["fine_label"] == "unknown"
    assert row["fine_accepted"] is False
    assert row["fine_unknown_reason"] == ("low_confidence_or_conflicting_fine_grained_evidence")


def test_scene_schema_round_trips_fine_grained_discovery_fields() -> None:
    payload = {
        "source_video": "bird.webm",
        "domain": {"name": "wildlife", "confidence": 0.9, "description": "bird"},
        "objects": [
            {
                "canonical_name": "bird",
                "display_name": "Bird",
                "action": "track",
                "confidence": 0.9,
                "fine_grained_candidates": ["Common_Kingfisher", "bird"],
                "semantic_facets": ["species", "color"],
                "taxonomy_hint": "species",
            }
        ],
    }

    restored = SceneDiscovery.from_dict(payload)
    serialized = restored.to_dict()

    assert restored.objects[0].fine_grained_candidates == ("common kingfisher",)
    assert restored.objects[0].semantic_facets == ("species", "color")
    assert serialized["schema_version"] == "2.2"


def test_online_shot_detector_reports_a_hard_cut_once() -> None:
    detector = OnlineShotChangeDetector(
        threshold=0.5,
        min_gap_frames=2,
        check_interval_frames=1,
    )
    red = np.zeros((32, 32, 3), dtype=np.uint8)
    red[:, :, 2] = 255
    green = np.zeros((32, 32, 3), dtype=np.uint8)
    green[:, :, 1] = 255

    first_cut, _ = detector.update(1, red)
    repeated_cut, _ = detector.update(2, red)
    hard_cut, score = detector.update(3, green)

    assert first_cut is False
    assert repeated_cut is False
    assert hard_cut is True
    assert score >= 0.5


def test_semantic_event_queue_drops_work_when_pending_limit_is_reached(
    tmp_path: Path,
) -> None:
    queue = SemanticEventQueue(tmp_path / "queue", context_id="test", max_pending_events=1)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    first = TrackOutput.from_xyxy(
        frame_index=1,
        sequence_name="stream",
        track_id=1,
        bbox_xyxy=BoundingBoxXYXY(4, 4, 40, 40),
        confidence=0.9,
        class_id=0,
        class_name="person",
    )
    second = TrackOutput.from_xyxy(
        frame_index=2,
        sequence_name="stream",
        track_id=2,
        bbox_xyxy=BoundingBoxXYXY(8, 8, 44, 44),
        confidence=0.8,
        class_id=0,
        class_name="person",
    )

    assert queue.enqueue(frame=frame, frame_index=1, track=first, reason="new")
    assert queue.enqueue(frame=frame, frame_index=2, track=second, reason="new") is None
    assert queue.pending_count == 1
    assert queue.dropped_full == 1


def test_qwen_temporal_observations_are_fused_in_frame_order() -> None:
    evidence = parse_qwen_answer(
        {
            "answer": {
                "track_predictions": [
                    {
                        "track_id": 4,
                        "class_label": "car",
                        "confidence": 0.8,
                        "observations": [
                            {"frame_index": 10, "class_label": "car", "confidence": 0.8},
                            {"frame_index": 40, "class_label": "car", "confidence": 0.9},
                        ],
                    }
                ]
            }
        }
    )

    result = fuse_track_semantics(evidence)
    row = result["tracks"][0]

    assert [item.evidence_frames for item in evidence] == [(10,), (40,)]
    assert row["class_label"] == "car"
    assert row["temporal_observation_count"] == 2
    assert row["temporal_span_frames"] == 30
    assert row["label_transition_count"] == 0


def test_temporal_instability_rejects_recent_label_flip() -> None:
    result = fuse_track_semantics(
        [
            TrackSemanticEvidence(8, "car", 0.9, "qwen", evidence_frames=(10,)),
            TrackSemanticEvidence(8, "bus", 0.8, "qwen", evidence_frames=(20,)),
            TrackSemanticEvidence(8, "car", 0.7, "qwen", evidence_frames=(30,)),
        ],
        minimum_margin=0.0,
        minimum_temporal_stability=0.8,
    )

    row = result["tracks"][0]
    assert row["class_label"] == "unknown"
    assert row["label_transition_count"] == 2


def test_same_frame_multimodel_evidence_is_one_temporal_observation() -> None:
    result = fuse_track_semantics(
        [
            TrackSemanticEvidence(5, "car", 0.9, "qwen", evidence_frames=(10,)),
            TrackSemanticEvidence(
                5,
                "car",
                0.8,
                "locateanything",
                evidence_frames=(10,),
            ),
            TrackSemanticEvidence(5, "car", 0.9, "qwen", evidence_frames=(30,)),
        ]
    )

    row = result["tracks"][0]
    assert row["class_label"] == "car"
    assert row["temporal_observation_count"] == 2
    assert row["label_transition_count"] == 0


def test_temporal_semantic_memory_is_bounded_and_round_trips(tmp_path: Path) -> None:
    memory_path = tmp_path / "semantic_memory.json"
    memory = TemporalSemanticMemory(context_id="video-a")
    memory.merge(
        [
            TrackSemanticEvidence(
                3,
                "car",
                0.7 + frame_index / 100,
                "qwen",
                evidence_frames=(frame_index,),
            )
            for frame_index in (10, 20, 30)
        ],
        max_observations_per_track=2,
    )
    memory.save(memory_path)

    loaded = TemporalSemanticMemory.load(memory_path, context_id="video-a")

    assert [row.evidence_frames for row in loaded.observations] == [(20,), (30,)]
    assert json.loads(memory_path.read_text(encoding="utf-8"))["summary"] == {
        "track_count": 1,
        "observation_count": 2,
    }


def test_semantic_evidence_maps_generated_subclass_to_registry_parent() -> None:
    rows = normalize_semantic_evidence(
        [TrackSemanticEvidence(7, "forward", 0.9, "qwen")],
        _registry(),
    )

    assert rows[0].class_label == "player"
    assert rows[0].attributes["specific_class"] == "forward"


def test_semantic_fusion_rejects_single_low_confidence_prediction() -> None:
    result = fuse_track_semantics(
        [TrackSemanticEvidence(3, "ambulance", 0.2, "qwen")],
        unknown_threshold=0.45,
        minimum_margin=0.10,
    )

    row = result["tracks"][0]
    assert row["class_label"] == "unknown"
    assert row["absolute_confidence"] == 0.2
    assert row["consensus"] == 1.0
    assert row["confidence"] == 0.2


def test_parse_qwen_dynamic_answer() -> None:
    answer = {
        "answer": json.dumps(
            {
                "track_predictions": [
                    {
                        "track_id": 7,
                        "class_label": "ambulance",
                        "attributes": {"color": "white"},
                        "confidence": 0.82,
                        "evidence_frames": [15],
                    }
                ]
            }
        )
    }

    rows = parse_qwen_answer(answer)

    assert rows[0].track_id == 7
    assert rows[0].class_label == "ambulance"
    assert rows[0].attributes["color"] == "white"


def test_parse_locate_ignores_wrong_expected_track() -> None:
    rows = parse_locate_evidence(
        {
            "associations": [
                {
                    "track_id": 7,
                    "frame_index": 10,
                    "class_label": "ambulance",
                    "confidence": 0.9,
                    "accepted_for_fusion": False,
                },
                {
                    "track_id": 8,
                    "frame_index": 10,
                    "class_label": "ambulance",
                    "confidence": 0.8,
                    "accepted_for_fusion": True,
                },
            ]
        }
    )

    assert [row.track_id for row in rows] == [8]


def test_semantic_renderer_keeps_unknown_tracks_visible(tmp_path: Path) -> None:
    import cv2

    source = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(
        str(source),
        cv2.VideoWriter_fourcc(*"mp4v"),
        5.0,
        (64, 48),
    )
    try:
        for _index in range(3):
            writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    finally:
        writer.release()
    tracks = tmp_path / "tracks.txt"
    tracks.write_text(
        "1,1,4,5,12,18,0.9,1,1\n1,2,30,5,12,18,0.8,1,1\n"
        "2,1,5,5,12,18,0.9,1,1\n2,2,31,5,12,18,0.8,1,1\n",
        encoding="utf-8",
    )
    semantics = tmp_path / "semantics.json"
    semantics.write_text(
        json.dumps(
            {
                "tracks": [
                    {
                        "track_id": 1,
                        "class_label": "car",
                        "accepted": True,
                        "confidence": 0.9,
                        "attributes": {"color": "red"},
                    },
                    {
                        "track_id": 2,
                        "class_label": "unknown",
                        "accepted": False,
                        "confidence": 0.4,
                        "attributes": {},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    result = render_semantic_video(
        source_video=source,
        tracks_path=tracks,
        semantics_path=semantics,
        output_video=tmp_path / "rendered.mp4",
        overwrite=True,
        max_frames=2,
    )

    assert result["video"]["rendered_frame_count"] == 2
    assert result["video"]["requested_max_frames"] == 2
    assert result["semantics_summary"]["track_count"] == 2
    assert result["semantics_summary"]["track_coverage"] == 0.5
    assert result["semantics_summary"]["box_count"] == 4
    assert Path(result["output_video"]).is_file()


def test_semantic_label_text_is_bounded_to_frame_width() -> None:
    import cv2

    text = _select_fitting_text(
        [
            "ID 17 | bird > common kingfisher | color=blue_head,orange_breast 0.95",
            "ID 17 | bird > common kingfisher 0.95",
            "ID 17 | bird > common kingfisher",
        ],
        max_width=220,
    )
    width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0][0] + 8

    assert width <= 220
    assert text.startswith("ID 17")


def _semantic_worker_fixture(
    tmp_path: Path,
) -> tuple[SemanticEventQueue, Path]:
    queue = SemanticEventQueue(tmp_path / "queue", context_id="camera-1")
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    track = TrackOutput.from_xyxy(
        frame_index=10,
        sequence_name="live",
        track_id=7,
        bbox_xyxy=BoundingBoxXYXY(20, 10, 60, 70),
        confidence=0.9,
        class_id=2,
        class_name="object",
    )
    event = queue.enqueue(
        frame=frame,
        frame_index=10,
        track=track,
        reason="unknown_track",
    )
    assert event is not None and event.is_file()
    assert (
        queue.enqueue(
            frame=frame,
            frame_index=20,
            track=track,
            reason="unknown_track",
        )
        is None
    )

    vlm_config = tmp_path / "vlm.yaml"
    vlm_config.write_text(
        """
input:
  source_video: placeholder.mp4
  tracks: placeholder.txt
output:
  dir: outputs/test
sampling:
  max_keyframes: 1
  max_tracks: 0
  max_crops_per_track: 1
  max_model_images: 1
model:
  model_id: Qwen/Qwen3-VL-4B-Instruct
  quantization: 8bit
prompt:
  output_schema: dynamic
runtime:
  overwrite: true
""".strip(),
        encoding="utf-8",
    )
    return queue, vlm_config


def test_realtime_semantic_queue_worker_updates_memory_and_cache(
    tmp_path: Path,
) -> None:
    queue, vlm_config = _semantic_worker_fixture(tmp_path)

    def fake_runner(_config, jobs):
        return {
            "model_id": "fixture",
            "quantization": "8bit",
            "timing": {"inference_seconds": 0.01},
            "cuda_memory": {},
            "batches": [
                {
                    "batch_id": jobs[0]["batch_id"],
                    "answer": json.dumps(
                        {
                            "track_predictions": [
                                {
                                    "track_id": 7,
                                    "class_label": "car",
                                    "confidence": 0.9,
                                    "observations": [
                                        {
                                            "frame_index": 10,
                                            "class_label": "car",
                                            "confidence": 0.9,
                                        }
                                    ],
                                }
                            ]
                        }
                    ),
                }
            ],
        }

    semantic_output = tmp_path / "semantic_cache.json"
    result = process_semantic_queue(
        queue_dir=queue.root,
        vlm_config_path=vlm_config,
        semantic_output=semantic_output,
        memory_path=tmp_path / "memory.json",
        runner=fake_runner,
    )
    cache = SemanticCacheView(semantic_output)

    assert result["processed_event_count"] == 1
    assert cache.refresh() is True
    assert cache.accepted(7)["class_label"] == "car"
    assert len(list(queue.processed_dir.glob("*.json"))) == 1


def test_semantic_worker_quarantines_invalid_model_output(tmp_path: Path) -> None:
    queue, vlm_config = _semantic_worker_fixture(tmp_path)

    def invalid_runner(_config, jobs):
        return {
            "batches": [{"batch_id": job["batch_id"], "answer": "not valid JSON"} for job in jobs]
        }

    result = process_semantic_queue(
        queue_dir=queue.root,
        vlm_config_path=vlm_config,
        semantic_output=tmp_path / "semantic_cache.json",
        memory_path=tmp_path / "memory.json",
        runner=invalid_runner,
    )

    assert result["status"] == "no_valid_evidence"
    assert result["failed_event_count"] == 1
    assert len(list(queue.failed_dir.glob("*.json"))) == 1
    assert not list(queue.pending_dir.glob("*.json"))
    assert not list(queue.processing_dir.glob("*.json"))


def test_semantic_worker_requeues_events_after_runner_failure(tmp_path: Path) -> None:
    queue, vlm_config = _semantic_worker_fixture(tmp_path)

    def failing_runner(_config, _jobs):
        raise RuntimeError("simulated OOM")

    with pytest.raises(RuntimeError, match="simulated OOM"):
        process_semantic_queue(
            queue_dir=queue.root,
            vlm_config_path=vlm_config,
            semantic_output=tmp_path / "semantic_cache.json",
            memory_path=tmp_path / "memory.json",
            runner=failing_runner,
        )

    assert len(list(queue.pending_dir.glob("*.json"))) == 1
    assert not list(queue.processing_dir.glob("*.json"))
