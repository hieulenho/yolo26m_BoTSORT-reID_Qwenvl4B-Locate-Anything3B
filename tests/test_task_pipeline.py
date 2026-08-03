from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from football_tracking.adaptive_tracking import semantic_queue as semantic_queue_module
from football_tracking.adaptive_tracking.fast_appearance import (
    FastVisualAttributeStore,
)
from football_tracking.adaptive_tracking.fast_semantics import (
    FastSemanticProposalStore,
)
from football_tracking.adaptive_tracking.semantic_fusion import TrackSemanticEvidence
from football_tracking.adaptive_tracking.semantic_queue import (
    SemanticCacheView,
    SemanticEventQueue,
    _enforce_event_taxonomy,
    _event_prompt,
    _group_event_prompt,
    _replace_file_with_retry,
)
from football_tracking.data.schemas import BoundingBoxXYXY
from football_tracking.detection.preprocessing import FramePreprocessor
from football_tracking.task_pipeline.builder import write_task_runtime
from football_tracking.task_pipeline.config import (
    TaskPipelineConfigError,
    load_task_pipeline_config,
)
from football_tracking.tracking.class_stabilizer import TrackClassStabilizer
from football_tracking.tracking.pipeline import load_tracking_config
from football_tracking.tracking.schemas import TrackerDetection, TrackOutput
from football_tracking.tracking.tracker_factory import load_tracker_runtime_config


def _track(
    *,
    frame_index: int = 1,
    track_id: int = 1,
    class_id: int = 0,
    class_name: str = "person",
    confidence: float = 0.9,
) -> TrackOutput:
    return TrackOutput.from_xyxy(
        frame_index=frame_index,
        sequence_name="test",
        track_id=track_id,
        bbox_xyxy=BoundingBoxXYXY(16, 16, 112, 112),
        confidence=confidence,
        class_id=class_id,
        class_name=class_name,
    )


def test_generic_task_builds_a_stream_tracking_config(tmp_path: Path) -> None:
    task = load_task_pipeline_config("configs/tasks/generic_coco_realtime.yaml")
    paths = write_task_runtime(
        config=task,
        output_dir=tmp_path / "runtime",
        output_video=tmp_path / "tracked.mp4",
        device="cuda",
        overwrite=False,
    )
    tracking = load_tracking_config(paths["tracking_config"])

    assert task.semantic.provider == "qwen"
    assert task.semantic.quantization == "8bit"
    assert tracking.source_type == "stream"
    assert tracking.tracker_name == "tracktrack"
    assert tracking.preprocessing_mode == "auto_low_light"
    assert tracking.class_ids is None
    tracker = load_tracker_runtime_config(task.tracker.name, task.tracker.config_path)
    # Production association remains class-agnostic; temporal class stabilization
    # handles detector label flicker after the identity has been preserved.
    assert tracker["class_gate"] is False
    assert tracker["class_aware"] is False


def test_dense_traffic_task_uses_bounded_live_semantics() -> None:
    task = load_task_pipeline_config("configs/tasks/traffic_objects.yaml")

    assert task.semantic.label_mode == "closed"
    assert task.semantic.enable_fine_labels is True
    assert set(task.semantic.allowed_labels) >= {"car", "truck", "unknown"}
    assert "SUV" in task.semantic.fine_label_taxonomy["car"]
    assert "tractor-trailer" in task.semantic.fine_label_taxonomy["truck"]
    assert task.semantic.label_aliases["person"] == "pedestrian"
    assert task.semantic.fine_label_aliases["semi truck"] == "tractor-trailer"
    assert task.semantic.fine_unknown_threshold == pytest.approx(0.72)
    assert task.semantic.fast_label_hint_threshold == pytest.approx(0.65)
    assert task.semantic.fast_color_hint_threshold == pytest.approx(0.55)
    assert task.semantic.events_per_frame == 1
    assert task.semantic.max_pending_events == 12
    assert task.semantic.minimum_track_age_frames == 15
    assert task.semantic.max_evidence_images == 3
    assert task.semantic.evidence_interval_frames == 12
    assert task.semantic.evidence_collection_delay_seconds == pytest.approx(0.75)
    assert task.semantic.evidence_panel_width == 512
    assert task.semantic.evidence_panel_height == 384
    assert task.semantic.evidence_context_fraction == pytest.approx(0.55)
    assert task.semantic.crop_padding == pytest.approx(0.15)
    assert task.semantic.crop_size == 256


def test_traffic_quality_task_uses_measured_accuracy_profile() -> None:
    task = load_task_pipeline_config("configs/tasks/traffic_objects_quality.yaml")

    assert task.detector.checkpoint == "yolo26s.pt"
    assert task.detector.imgsz == 768
    assert task.tracker.name == "tracktrack_reid"
    assert task.semantic.max_pending_events == 512
    assert task.semantic.max_evidence_images == 2
    assert task.semantic.evidence_context_fraction == pytest.approx(0.45)
    assert task.semantic.crop_padding == pytest.approx(0.25)
    assert task.semantic.crop_size == 384


def test_runtime_preprocessing_override_is_recorded(tmp_path: Path) -> None:
    task = load_task_pipeline_config("configs/tasks/generic_coco_realtime.yaml")
    paths = write_task_runtime(
        config=task,
        output_dir=tmp_path / "runtime",
        output_video=tmp_path / "tracked.mp4",
        device="cuda",
        overwrite=False,
        preprocessing_mode="none",
    )

    tracking = load_tracking_config(paths["tracking_config"])
    resolved = json.loads(paths["resolved_task"].read_text(encoding="utf-8"))
    assert tracking.preprocessing_mode == "none"
    assert resolved["detector"]["preprocessing"]["mode"] == "none"


def test_task_config_rejects_a_model_id_that_worker_would_not_run(
    tmp_path: Path,
) -> None:
    source = Path("configs/tasks/generic_coco_realtime.yaml")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["semantic"]["model_id"] = "Qwen/another-model"
    config_path = tmp_path / "wrong_model.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(TaskPipelineConfigError, match="semantic model is fixed"):
        load_task_pipeline_config(config_path)


def test_runtime_tracker_override_is_recorded(tmp_path: Path) -> None:
    task = load_task_pipeline_config("configs/tasks/wildlife_birds.yaml")
    override = Path(
        "configs/benchmarks/tracker_profiles/tracktrack_open_t015.yaml"
    )
    paths = write_task_runtime(
        config=task,
        output_dir=tmp_path / "runtime",
        output_video=tmp_path / "tracked.mp4",
        device="cpu",
        overwrite=False,
        tracker_config_path=override,
    )

    resolved = json.loads(paths["resolved_task"].read_text(encoding="utf-8"))
    assert Path(resolved["tracker"]["config"]).resolve() == override.resolve()
    assert resolved["tracker"]["name"] == "tracktrack"


def test_runtime_tracker_override_records_reid_variant(tmp_path: Path) -> None:
    task = load_task_pipeline_config("configs/tasks/traffic_objects.yaml")
    override = Path("configs/trackers/tracktrack_reid_realtime.yaml")
    paths = write_task_runtime(
        config=task,
        output_dir=tmp_path / "runtime",
        output_video=tmp_path / "tracked.mp4",
        device="cpu",
        overwrite=False,
        tracker_config_path=override,
    )

    resolved = json.loads(paths["resolved_task"].read_text(encoding="utf-8"))
    tracking = load_tracking_config(paths["tracking_config"])
    assert resolved["tracker"]["name"] == "tracktrack_reid"
    assert tracking.tracker_name == "tracktrack_reid"


@pytest.mark.parametrize(
    "config_path,task_id",
    [
        ("configs/tasks/traffic_objects.yaml", "traffic_objects"),
        ("configs/tasks/classroom_roles.yaml", "classroom_roles"),
        ("configs/tasks/football_roles.yaml", "football_roles"),
        ("configs/tasks/wildlife_birds.yaml", "wildlife_birds"),
        ("configs/tasks/microscopy_cells.yaml", "microscopy_cells"),
        ("configs/tasks/open_vocabulary_example.yaml", "open_vocabulary_example"),
    ],
)
def test_supported_task_configs_are_valid(config_path: str, task_id: str) -> None:
    task = load_task_pipeline_config(config_path)

    assert task.task_id == task_id
    assert task.semantic.provider == "qwen"
    assert task.semantic.quantization == "8bit"


def test_task_rejects_a_second_or_non_8bit_semantic_model(tmp_path: Path) -> None:
    payload = yaml.safe_load(
        Path("configs/tasks/generic_coco_realtime.yaml").read_text(encoding="utf-8")
    )
    payload["semantic"]["quantization"] = "4bit"
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(TaskPipelineConfigError, match="8bit"):
        load_task_pipeline_config(path)


def test_auto_low_light_preprocessing_is_geometry_preserving() -> None:
    processor = FramePreprocessor(mode="auto_low_light", low_light_threshold=70)
    dark = np.full((80, 120, 3), 20, dtype=np.uint8)
    bright = np.full((80, 120, 3), 180, dtype=np.uint8)

    dark_result = processor.process(dark)
    bright_result = processor.process(bright)

    assert dark_result.applied is True
    assert bright_result.applied is False
    assert dark_result.frame.shape == dark.shape
    assert bright_result.frame is bright


def test_track_class_stabilizer_suppresses_one_frame_label_flip() -> None:
    stabilizer = TrackClassStabilizer(history_frames=10, switch_margin=0.20)
    first = stabilizer.update([_track(class_id=2, class_name="car")], 1)[0]
    second = stabilizer.update(
        [_track(frame_index=2, class_id=7, class_name="truck")],
        2,
    )[0]
    third = stabilizer.update(
        [_track(frame_index=3, class_id=2, class_name="car")],
        3,
    )[0]

    assert first.class_name == "car"
    assert second.class_name == "car"
    assert third.class_name == "car"
    assert stabilizer.diagnostics()["suppressed_switches"] >= 1


def test_sparse_semantic_proposals_persist_on_the_matching_track() -> None:
    store = FastSemanticProposalStore(minimum_observations=2)
    track = _track(track_id=7)
    proposal = TrackerDetection.from_xyxy(
        frame_index=1,
        sequence_name="test",
        bbox_xyxy=track.bbox_xyxy,
        confidence=0.40,
        class_id=1000,
        class_name="student",
    )

    first = store.update([track], [proposal], frame_index=1)[0]
    second = store.update([track], [proposal], frame_index=7)[0]
    retained = store.update([track], [], frame_index=8)[0]

    assert first.class_name == "person"
    assert second.class_name == "student"
    assert retained.class_name == "student"
    assert retained.metadata["base_detector_class_name"] == "person"
    assert retained.metadata["fast_semantic_source"] == "supplemental_detector"
    assert store.diagnostics()["stable_track_count"] == 1


def test_classroom_task_combines_person_recall_with_sparse_role_proposals(
    tmp_path: Path,
) -> None:
    task = load_task_pipeline_config("configs/tasks/classroom_roles.yaml")
    paths = write_task_runtime(
        config=task,
        output_dir=tmp_path / "runtime",
        output_video=tmp_path / "classroom.mp4",
        device="cuda",
        overwrite=False,
    )
    generated = yaml.safe_load(
        paths["tracking_config"].read_text(encoding="utf-8")
    )
    supplemental = generated["model"]["supplemental_detectors"][0]

    assert task.detector.class_ids == (0, 1000, 1001, 1002, 1003)
    assert task.detector.tracker_class_ids == (0,)
    assert supplemental["backend"] == "ultralytics_yoloe"
    assert supplemental["every_n_frames"] == 6
    assert supplemental["output_class_ids"] == [1000, 1001, 1002, 1003]
    assert supplemental["compatible_tracker_class_ids"] == [0]
    assert generated["detector"]["source_class_names"] == {
        "1000": "student",
        "1001": "teacher",
        "1002": "classroom_staff",
        "1003": "visitor",
    }


def test_sparse_semantic_proposal_respects_base_class_compatibility() -> None:
    store = FastSemanticProposalStore(minimum_observations=1)
    person = _track(track_id=7, class_id=0, class_name="person")
    proposal = TrackerDetection.from_xyxy(
        frame_index=1,
        sequence_name="test",
        bbox_xyxy=person.bbox_xyxy,
        confidence=0.9,
        class_id=1100,
        class_name="sedan",
        metadata={"compatible_tracker_class_ids": [2]},
    )

    result = store.update([person], [proposal], frame_index=1)[0]

    assert result.class_name == "person"
    assert store.diagnostics()["matched_proposals"] == 0


def test_fast_vehicle_color_requires_temporal_consensus() -> None:
    store = FastVisualAttributeStore(
        sample_interval_frames=1,
        minimum_observations=3,
    )
    frame = np.zeros((140, 140, 3), dtype=np.uint8)
    frame[16:112, 16:112] = (0, 0, 220)
    car = _track(class_id=2, class_name="car")

    first = store.update(frame, [car], frame_index=1)[0]
    second = store.update(frame, [car], frame_index=2)[0]
    third = store.update(frame, [car], frame_index=3)[0]

    assert first.class_name == "car"
    assert second.class_name == "car"
    assert third.class_name == "red car"
    assert third.metadata["fast_visual_color"] == "red"
    assert store.diagnostics()["stable_track_count"] == 1


def test_semantic_queue_replaces_a_pending_crop_with_better_temporal_evidence(
    tmp_path: Path,
) -> None:
    queue = SemanticEventQueue(
        tmp_path / "queue",
        context_id="task-test",
        semantic_task={
            "task_id": "test",
            "label_mode": "closed",
            "allowed_labels": ["student", "teacher", "unknown"],
            "max_evidence_images": 3,
            "minimum_crop_quality": 0.0,
            "replacement_quality_margin": 0.01,
        },
    )
    flat = np.full((128, 128, 3), 80, dtype=np.uint8)
    detailed = flat.copy()
    cv2.line(detailed, (0, 0), (127, 127), (255, 255, 255), 4)
    cv2.line(detailed, (127, 0), (0, 127), (0, 0, 0), 4)

    first = queue.enqueue(
        frame=flat,
        frame_index=1,
        track=_track(frame_index=1),
        reason="unknown_track",
        minimum_frame_gap=1,
    )
    second = queue.enqueue(
        frame=detailed,
        frame_index=2,
        track=_track(frame_index=2),
        reason="unknown_track",
        minimum_frame_gap=1,
    )
    event = json.loads(second.read_text(encoding="utf-8"))

    assert first is not None
    assert second is not None
    assert queue.pending_count == 1
    assert queue.replaced_pending == 1
    assert event["schema_version"] == "2.0"
    assert event["semantic_task"]["task_id"] == "test"
    assert event["evidence_frame_indices"] == [1, 2]
    assert event["evidence_layout"] == "panel"
    assert len(event["qwen_image_paths"]) == 1
    panel_path = Path(event["qwen_image_paths"][0])
    panel = cv2.imread(str(panel_path))
    assert panel_path.is_file()
    assert panel is not None
    assert panel.shape[:2] == (384, 512)


def test_semantic_queue_keeps_a_second_temporal_crop_without_quality_gain(
    tmp_path: Path,
) -> None:
    queue = SemanticEventQueue(
        tmp_path / "queue",
        context_id="task-test",
        semantic_task={
            "max_evidence_images": 2,
            "minimum_crop_quality": 0.0,
            "replacement_quality_margin": 0.50,
        },
    )
    frame = np.full((128, 128, 3), 80, dtype=np.uint8)

    first = queue.enqueue(
        frame=frame,
        frame_index=1,
        track=_track(frame_index=1),
        reason="unknown_track",
        minimum_frame_gap=1,
    )
    second = queue.enqueue(
        frame=frame,
        frame_index=2,
        track=_track(frame_index=2),
        reason="unknown_track",
        minimum_frame_gap=1,
    )

    assert first is not None
    assert second is not None
    event = json.loads(second.read_text(encoding="utf-8"))
    assert event["evidence_frame_indices"] == [1, 2]
    assert queue.replaced_pending == 1


def test_semantic_event_keeps_base_detector_and_fast_proposal_separate(
    tmp_path: Path,
) -> None:
    queue = SemanticEventQueue(tmp_path / "queue", context_id="classroom")
    track = replace(
        _track(track_id=7, class_id=1000, class_name="student"),
        metadata={
            "base_detector_class_id": 0,
            "base_detector_class_name": "person",
            "fast_semantic_label": "student",
            "fast_semantic_confidence": 0.8,
            "fast_semantic_source": "supplemental_detector",
        },
    )
    event_path = queue.enqueue(
        frame=np.full((128, 128, 3), 100, dtype=np.uint8),
        frame_index=20,
        track=track,
        reason="unknown_track",
    )

    assert event_path is not None
    event = json.loads(event_path.read_text(encoding="utf-8"))
    assert event["detector_class_name"] == "person"
    assert event["fast_semantic_proposal"]["class_label"] == "student"


def test_queue_file_replace_retries_a_windows_sharing_violation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.json"
    target = tmp_path / "target.json"
    source.write_text("{}", encoding="utf-8")
    original_replace = Path.replace
    calls = 0

    def flaky_replace(path: Path, destination: Path) -> Path:
        nonlocal calls
        if path == source and calls == 0:
            calls += 1
            raise PermissionError(32, "simulated sharing violation")
        calls += 1
        return original_replace(path, destination)

    monkeypatch.setattr(Path, "replace", flaky_replace)
    _replace_file_with_retry(
        source,
        target,
        attempts=3,
        initial_delay_seconds=0.0,
    )

    assert calls == 2
    assert target.is_file()


def test_task_semantics_hold_events_briefly_for_temporal_evidence() -> None:
    task = load_task_pipeline_config("configs/tasks/traffic_objects.yaml")

    assert task.semantic.evidence_interval_frames == 12
    assert task.semantic.evidence_collection_delay_seconds == pytest.approx(0.75)
    payload = task.semantic_event_payload()
    assert payload["evidence_interval_frames"] == 12
    assert payload["evidence_collection_delay_seconds"] == pytest.approx(0.75)


def test_semantic_queue_applies_backpressure_before_creating_crops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = SemanticEventQueue(
        tmp_path / "queue",
        context_id="traffic",
        max_pending_events=1,
    )
    frame = np.full((128, 128, 3), 100, dtype=np.uint8)
    assert queue.enqueue(
        frame=frame,
        frame_index=1,
        track=_track(track_id=1),
        reason="unknown_track",
        minimum_frame_gap=30,
    )

    def fail_if_crop_runs(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("crop generation must not run while the queue is full")

    monkeypatch.setattr(semantic_queue_module, "_track_crop", fail_if_crop_runs)
    assert (
        queue.enqueue(
            frame=frame,
            frame_index=2,
            track=_track(frame_index=2, track_id=2),
            reason="unknown_track",
            minimum_frame_gap=30,
        )
        is None
    )
    assert queue.dropped_full == 1


def test_semantic_queue_keeps_processing_tracks_in_pending_display_state(
    tmp_path: Path,
) -> None:
    queue = SemanticEventQueue(tmp_path / "queue", context_id="traffic")
    processing = queue.processing_dir / "f000000010_t0000007.json"
    processing.write_text("{}", encoding="utf-8")

    assert queue.pending_track_ids == {7}


def test_semantic_queue_does_not_duplicate_a_track_while_it_is_processing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = SemanticEventQueue(tmp_path / "queue", context_id="traffic")
    frame = np.full((128, 128, 3), 100, dtype=np.uint8)
    processing = queue.processing_dir / "f000000010_t0000007.json"
    processing.write_text("{}", encoding="utf-8")

    def fail_if_crop_runs(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("a processing track must not create duplicate evidence")

    monkeypatch.setattr(semantic_queue_module, "_track_crop", fail_if_crop_runs)
    assert (
        queue.enqueue(
            frame=frame,
            frame_index=100,
            track=_track(frame_index=100, track_id=7),
            reason="unknown_track",
            minimum_frame_gap=30,
        )
        is None
    )
    assert not list(queue.pending_dir.glob("*.json"))


def test_semantic_queue_capacity_includes_processing_tracks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = SemanticEventQueue(
        tmp_path / "queue",
        context_id="traffic",
        max_pending_events=1,
    )
    frame = np.full((128, 128, 3), 100, dtype=np.uint8)
    processing = queue.processing_dir / "f000000010_t0000007.json"
    processing.write_text("{}", encoding="utf-8")

    def fail_if_crop_runs(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("processing work must count toward queue capacity")

    monkeypatch.setattr(semantic_queue_module, "_track_crop", fail_if_crop_runs)
    assert (
        queue.enqueue(
            frame=frame,
            frame_index=100,
            track=_track(frame_index=100, track_id=8),
            reason="unknown_track",
            minimum_frame_gap=30,
        )
        is None
    )
    assert queue.dropped_full == 1


def test_closed_semantic_prompt_uses_a_compact_response_schema() -> None:
    prompt = _event_prompt(
        {
            "track_id": 7,
            "frame_index": 20,
            "detector_class_name": "car",
            "evidence_frame_indices": [10, 20],
            "semantic_task": {
                "task_id": "traffic",
                "task_name": "Traffic",
                "label_mode": "closed",
                "allowed_labels": ["car", "truck", "unknown"],
                "attributes": ["color"],
                "unknown_threshold": 0.75,
            },
        }
    )

    assert '"class_label":"..."' in prompt
    assert '"evidence_frames":[10, 20]' in prompt
    assert "observations" not in prompt
    assert '"fine_label":' not in prompt


def test_closed_semantic_prompt_can_request_an_open_fine_subtype() -> None:
    prompt = _event_prompt(
        {
            "track_id": 7,
            "frame_index": 20,
            "detector_class_name": "car",
            "evidence_frame_indices": [10, 20],
            "semantic_task": {
                "task_id": "traffic",
                "task_name": "Traffic",
                "label_mode": "closed",
                "enable_fine_labels": True,
                "allowed_labels": ["car", "truck", "unknown"],
                "attributes": ["color"],
                "unknown_threshold": 0.75,
            },
        }
    )

    assert '"fine_label":"..."' in prompt
    assert '"fine_label_type":"subtype"' in prompt
    assert "taxonomy_path" not in prompt


def test_semantic_cache_exposes_waiting_pending_unknown_and_accepted_states(
    tmp_path: Path,
) -> None:
    path = tmp_path / "semantic.json"
    path.write_text(
        json.dumps(
            {
                "tracks": [
                    {"track_id": 2, "accepted": False, "class_label": "unknown"},
                    {
                        "track_id": 3,
                        "accepted": True,
                        "class_label": "teacher",
                        "display_label": "teacher",
                        "confidence": 0.9,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    cache = SemanticCacheView(path)
    assert cache.refresh()
    tracks = [_track(track_id=value) for value in (1, 2, 3, 4)]
    decorated = cache.decorate(
        tracks,
        pending_track_ids={1},
        semantic_enabled=True,
    )

    assert decorated[0].class_name == "person"
    assert decorated[0].metadata["semantic_status"] == "pending"
    assert decorated[1].class_name == "person | unknown"
    assert decorated[2].class_name == "person | teacher"
    assert decorated[3].class_name == "person"
    assert decorated[3].metadata["semantic_status"] == "waiting"


def test_semantic_cache_renders_vehicle_color_and_fine_subtype(
    tmp_path: Path,
) -> None:
    path = tmp_path / "semantic.json"
    path.write_text(
        json.dumps(
            {
                "tracks": [
                    {
                        "track_id": 1,
                        "accepted": True,
                        "class_label": "car",
                        "detector_class_name": "car",
                        "fine_label": "sedan",
                        "fine_accepted": True,
                        "attributes": {"color": "silver"},
                        "confidence": 0.91,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    cache = SemanticCacheView(path)
    assert cache.refresh()

    decorated = cache.decorate(
        [_track(class_id=2, class_name="car")],
        semantic_enabled=True,
    )

    assert decorated[0].class_name == "car | silver sedan"


def test_group_prompt_requests_compact_color_and_distinct_fine_label() -> None:
    task = {
        "task_id": "traffic",
        "label_mode": "closed",
        "enable_fine_labels": True,
        "allowed_labels": ["car", "truck", "unknown"],
        "fine_label_taxonomy": {"car": ["sedan", "SUV"]},
        "attributes": ["color"],
    }
    events = [
        {
            "track_id": track_id,
            "frame_index": 10 + track_id,
            "evidence_frame_indices": [10 + track_id, 20 + track_id],
            "detector_class_name": "car",
            "semantic_task": task,
        }
        for track_id in (1, 2)
    ]

    prompt = _group_event_prompt(events)

    assert '"class":"..."' in prompt
    assert '"subtype":"..."' in prompt
    assert '"color":"..."' in prompt
    assert '"q":0.0' in prompt
    assert '"frames"' not in prompt.split("return JSON only:", 1)[-1]
    assert "Never place a subtype in class" in prompt
    assert "silver" in prompt
    assert '"SUV"' in prompt


def test_closed_taxonomy_repairs_vehicle_subtype_and_color_field_swaps() -> None:
    rows = _enforce_event_taxonomy(
        [
            TrackSemanticEvidence(
                track_id=23,
                class_label="SUV",
                fine_label="red",
                confidence=0.8,
                fine_confidence=0.8,
                fine_label_type="subtype",
                attributes={"color": "red"},
                source="qwen",
            )
        ],
        {
            "semantic_task": {
                "label_mode": "closed",
                "allowed_labels": ["car", "truck", "unknown"],
                "fine_label_taxonomy": {
                    "car": ["sedan", "SUV", "hatchback"],
                    "truck": ["pickup"],
                },
                "attributes": ["color"],
            }
        },
    )

    assert len(rows) == 1
    assert rows[0].class_label == "car"
    assert rows[0].fine_label == "suv"
    assert rows[0].fine_confidence == pytest.approx(0.8)
    assert rows[0].attributes == {"color": "red"}


def test_closed_taxonomy_normalizes_vehicle_subtype_aliases() -> None:
    rows = _enforce_event_taxonomy(
        [
            TrackSemanticEvidence(
                track_id=24,
                class_label="semi truck",
                fine_label="unknown",
                confidence=0.9,
                fine_confidence=0.0,
                source="qwen",
            )
        ],
        {
            "semantic_task": {
                "label_mode": "closed",
                "allowed_labels": ["car", "truck", "unknown"],
                "fine_label_taxonomy": {
                    "car": ["sedan", "SUV"],
                    "truck": ["box truck", "tractor-trailer"],
                },
                "fine_label_aliases": {
                    "semi truck": "tractor-trailer",
                },
                "attributes": ["color"],
            }
        },
    )

    assert len(rows) == 1
    assert rows[0].class_label == "truck"
    assert rows[0].fine_label == "tractor-trailer"
    assert rows[0].fine_confidence == pytest.approx(0.9)


def test_closed_taxonomy_normalizes_base_label_aliases() -> None:
    rows = _enforce_event_taxonomy(
        [
            TrackSemanticEvidence(
                track_id=25,
                class_label="person",
                fine_label="adult pedestrian",
                confidence=0.92,
                fine_confidence=0.9,
                source="qwen",
            )
        ],
        {
            "semantic_task": {
                "label_mode": "closed",
                "allowed_labels": ["pedestrian", "car", "unknown"],
                "label_aliases": {"person": "pedestrian"},
                "fine_label_taxonomy": {
                    "pedestrian": ["adult pedestrian"],
                    "car": ["sedan"],
                },
                "attributes": [],
            }
        },
    )

    assert len(rows) == 1
    assert rows[0].class_label == "pedestrian"
    assert rows[0].fine_label == "adult pedestrian"


def test_task_rejects_unknown_fine_label_alias_target(tmp_path: Path) -> None:
    payload = yaml.safe_load(
        Path("configs/tasks/traffic_objects.yaml").read_text(encoding="utf-8")
    )
    payload["semantic"]["fine_label_aliases"]["lorry"] = "unknown truck kind"
    path = tmp_path / "invalid_alias.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(TaskPipelineConfigError, match="aliases target"):
        load_task_pipeline_config(path)


def test_task_rejects_unknown_base_label_alias_target(tmp_path: Path) -> None:
    payload = yaml.safe_load(
        Path("configs/tasks/traffic_objects.yaml").read_text(encoding="utf-8")
    )
    payload["semantic"]["label_aliases"]["person"] = "road user"
    path = tmp_path / "invalid_base_alias.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(TaskPipelineConfigError, match="label_aliases target"):
        load_task_pipeline_config(path)


def test_webcam_requirements_exclude_research_only_dependencies() -> None:
    webcam = Path("requirements/webcam.txt").read_text(encoding="utf-8").lower()
    runtime = Path("requirements/runtime.txt").read_text(encoding="utf-8").lower()
    base = Path("requirements/base.txt").read_text(encoding="utf-8").lower()

    assert "-r runtime.txt" in webcam
    assert "-r qwen.txt" in webcam
    assert "trackeval" not in webcam + runtime
    assert "pandas" not in webcam + runtime
    assert "matplotlib" not in webcam + runtime
    assert "-r runtime.txt" in base
    assert "trackeval" in base
