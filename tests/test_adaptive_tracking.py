from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from football_tracking.adaptive_tracking.config_builder import (
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
from football_tracking.adaptive_tracking.semantic_render import render_semantic_video
from football_tracking.adaptive_tracking.shot_sampling import detect_shot_starts
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

    assert route.route_name == "coco_pretrained"
    assert route.class_ids == (0, 32)
    assert route.tracker_class_ids == (0, 32)


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
    assert generated["tracker"]["name"] == "ocsort"
    assert generated["tracker"]["config"].endswith("ocsort_realtime.yaml")
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
    assert payload["detector"]["class_ids"] == [0, 32]
    assert payload["detector"]["tracker_class_ids"] == [0]


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

    assert payload["tracker"]["name"] == "tracktrack"
    assert payload["tracker"]["config"].endswith("tracktrack_realtime.yaml")


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

    assert payload["tracker"]["name"] == "botsort_reid"
    assert payload["tracker"]["config"].endswith(
        "botsort_reid_identity_stable.yaml"
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
                "discovery": _discovery(
                    "traffic", [{"name": "car", "confidence": 0.9}]
                ).to_dict(),
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
    assert {row["class_label"] for row in plan["requests"]} == {
        "surgical instrument"
    }
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
                "keyframes": [
                    {"frame_index": 5, "path": "keyframe.jpg", "visible_track_ids": [1]}
                ],
                "crops": [
                    {"track_id": 99, "frame_index": 50, "path": "track99.jpg"}
                ],
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
