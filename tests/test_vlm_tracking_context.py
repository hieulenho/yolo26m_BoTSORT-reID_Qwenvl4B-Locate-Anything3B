from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from football_tracking.vlm.config import DEFAULT_QWEN4B_MODEL_ID, load_vlm_tracking_config
from football_tracking.vlm.qwen_runner import _build_user_content
from football_tracking.vlm.tracking_context import (
    MotTrackRow,
    VideoInfo,
    _build_model_batches,
    _merge_qwen_batch_results,
    _select_representative_track_rows,
    read_mot_tracks,
    run_vlm_analysis,
)


def _write_video(path: Path, frame_count: int = 5) -> Path:
    import cv2  # type: ignore[import-not-found]

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        5.0,
        (64, 48),
    )
    try:
        for index in range(frame_count):
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            frame[:, :, 1] = 20 + index * 20
            writer.write(frame)
    finally:
        writer.release()
    return path


def _write_tracks(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "1,1,5,6,12,18,0.900000,1,1.000000",
                "2,1,8,6,12,18,0.910000,1,1.000000",
                "3,1,11,6,12,18,0.920000,1,1.000000",
                "1,2,35,12,10,14,0.800000,1,1.000000",
                "3,2,36,12,10,14,0.820000,1,1.000000",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_config(root: Path, video: Path, tracks: Path, metadata: Path) -> Path:
    config = root / "vlm.yaml"
    config.write_text(
        f"""
input:
  source_video: {video.as_posix()}
  tracked_video: {video.as_posix()}
  tracks: {tracks.as_posix()}
  metadata: {metadata.as_posix()}
output:
  dir: {(root / "outputs" / "vlm").as_posix()}
  keyframes_dir: keyframes
  crops_dir: crops
sampling:
  keyframe_interval_seconds: 0.2
  max_keyframes: 2
  max_tracks: 2
  max_crops_per_track: 1
  crop_padding: 0.1
  crop_output_size: 128
model:
  model_id: {DEFAULT_QWEN4B_MODEL_ID}
  run_model: false
prompt:
  task: Hay tom tat tracking.
runtime:
  overwrite: true
""".strip(),
        encoding="utf-8",
    )
    return config


def test_vlm_config_loads_qwen4b_defaults(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FOOTBALL_TRACKING_ROOT", str(tmp_path))
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    video = _write_video(tmp_path / "video.mp4")
    tracks = _write_tracks(tmp_path / "tracks.txt")
    metadata = tmp_path / "tracks.metadata.json"
    metadata.write_text("{}", encoding="utf-8")
    config_path = _write_config(tmp_path, video, tracks, metadata)

    config = load_vlm_tracking_config(config_path)

    assert config.model_id == DEFAULT_QWEN4B_MODEL_ID
    assert config.run_model is False
    assert config.max_keyframes == 2
    assert config.crop_output_size == 128


def test_read_mot_tracks_sorts_rows(tmp_path) -> None:
    tracks = tmp_path / "tracks.txt"
    tracks.write_text(
        "2,2,1,2,3,4,0.8,1,1\n1,1,1,2,3,4,0.9,1,1\n",
        encoding="utf-8",
    )

    rows = read_mot_tracks(tracks)

    assert [(row.frame_index, row.track_id) for row in rows] == [(1, 1), (2, 2)]


def test_run_vlm_analysis_writes_context_keyframes_and_crops(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FOOTBALL_TRACKING_ROOT", str(tmp_path))
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    video = _write_video(tmp_path / "video.mp4")
    tracks = _write_tracks(tmp_path / "tracks.txt")
    metadata = tmp_path / "tracks.metadata.json"
    metadata.write_text(json.dumps({"tracker": "botsort_reid"}), encoding="utf-8")
    config_path = _write_config(tmp_path, video, tracks, metadata)

    result = run_vlm_analysis(config_path)

    context_path = Path(result["paths"]["context_json"])
    prompt_path = Path(result["paths"]["prompt_md"])
    context = json.loads(context_path.read_text(encoding="utf-8"))

    assert result["summary"]["track_count"] == 2
    assert result["summary"]["keyframe_count"] == 2
    assert result["summary"]["crop_count"] == 2
    assert context["tracking_summary"]["track_observation_count"] == 5
    assert context["tracking_metadata"]["tracker"] == "botsort_reid"
    assert context["tracks"][0]["duration_seconds"] == 0.6
    assert context["tracks"][1]["gap_count"] == 1
    assert context["tracking_diagnostics"]["fragmented_tracks"][0]["track_id"] == 2
    assert context["tracking_diagnostics"]["stable_long_tracks"][0]["track_id"] == 1
    assert context_path.is_file()
    assert prompt_path.is_file()
    prompt_text = prompt_path.read_text(encoding="utf-8")
    assert "Tracking VLM Analysis Task" in prompt_text
    assert "tracking_diagnostics" in prompt_text
    assert '"track_predictions"' in prompt_text
    assert "referee_black" in prompt_text
    assert "Do not infer a semantic label from track duration" in prompt_text
    assert '"obs"' in prompt_text
    assert '"dur_s"' in prompt_text
    assert len(prompt_text) < 5000
    assert list((tmp_path / "outputs" / "vlm" / "keyframes").glob("*.jpg"))
    crop_paths = list((tmp_path / "outputs" / "vlm" / "crops").glob("track_*/*.jpg"))
    assert crop_paths
    import cv2  # type: ignore[import-not-found]

    crop = cv2.imread(str(crop_paths[0]))
    assert crop.shape[:2] == (128, 128)


def test_run_vlm_analysis_dry_run_does_not_write_outputs(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FOOTBALL_TRACKING_ROOT", str(tmp_path))
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    video = _write_video(tmp_path / "video.mp4")
    tracks = _write_tracks(tmp_path / "tracks.txt")
    metadata = tmp_path / "tracks.metadata.json"
    metadata.write_text("{}", encoding="utf-8")
    config_path = _write_config(tmp_path, video, tracks, metadata)

    result = run_vlm_analysis(config_path, dry_run=True)

    assert result["dry_run"] is True
    assert not (tmp_path / "outputs" / "vlm" / "vlm_context.json").exists()


def test_explicit_track_ids_override_automatic_ranking(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FOOTBALL_TRACKING_ROOT", str(tmp_path))
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    video = _write_video(tmp_path / "video.mp4")
    tracks = _write_tracks(tmp_path / "tracks.txt")
    metadata = tmp_path / "tracks.metadata.json"
    metadata.write_text("{}", encoding="utf-8")
    config_path = _write_config(tmp_path, video, tracks, metadata)

    result = run_vlm_analysis(
        config_path,
        overrides={
            "track_ids": "2",
            "max_tracks": 1,
            "output_dir": tmp_path / "outputs" / "explicit",
        },
    )

    context = json.loads(Path(result["paths"]["context_json"]).read_text(encoding="utf-8"))
    assert [row["track_id"] for row in context["tracks"]] == [2]
    assert result["summary"]["selected_track_count"] == 1


def test_model_batches_cover_every_selected_track_with_bounded_images(tmp_path) -> None:
    context = {"tracks": [{"track_id": track_id} for track_id in range(1, 6)]}
    keyframes = [{"path": str(tmp_path / "keyframe.jpg")}]
    crops = [
        {
            "track_id": track_id,
            "path": str(tmp_path / f"track_{track_id}_{crop_index}.jpg"),
        }
        for track_id in range(1, 6)
        for crop_index in range(2)
    ]

    batches = _build_model_batches(context, keyframes, crops, max_images=5)

    assert len(batches) == 3
    assert {track_id for batch in batches for track_id in batch["track_ids"]} == set(
        range(1, 6)
    )
    assert all(len(batch["image_paths"]) <= 5 for batch in batches)
    assert all(len(batch["image_labels"]) == len(batch["image_paths"]) for batch in batches)
    assert any("track ID 1" in label for label in batches[0]["image_labels"])


def test_qwen_images_are_interleaved_with_track_labels(tmp_path: Path) -> None:
    image_paths = [tmp_path / "keyframe.jpg", tmp_path / "crop.jpg"]

    content = _build_user_content(
        "Return JSON.",
        image_paths,
        ["Global keyframe.", "Appearance crop for track ID 7."],
    )

    assert [item["type"] for item in content] == [
        "text",
        "image",
        "text",
        "image",
        "text",
    ]
    assert content[2]["text"] == "Appearance crop for track ID 7."
    assert content[-1]["text"] == "Return JSON."


def test_batch_merge_marks_unreturned_tracks_unknown() -> None:
    jobs = [
        {
            "batch_id": "batch_001",
            "track_ids": [1, 2],
            "image_paths": [Path("frame.jpg")],
            "prompt_path": "prompt.md",
        }
    ]
    raw = {
        "status": "ok",
        "batches": [
            {
                "batch_id": "batch_001",
                "image_count": 1,
                "answer": json.dumps(
                    {
                        "track_predictions": [
                            {
                                "track_id": 1,
                                "class_label": "car",
                                "confidence": 0.9,
                            }
                        ]
                    }
                ),
            }
        ],
    }

    merged = _merge_qwen_batch_results(raw, jobs)

    predictions = {row["track_id"]: row for row in merged["answer"]["track_predictions"]}
    assert merged["status"] == "partial"
    assert predictions[1]["class_label"] == "car"
    assert predictions[2]["class_label"] == "unknown"
    assert merged["coverage"]["missing_track_ids"] == [2]


def test_representative_crops_are_temporally_diverse_and_quality_ranked() -> None:
    rows = [
        MotTrackRow(frame, 1, 5, 5, width, 20, confidence)
        for frame, width, confidence in (
            (1, 8, 0.5),
            (2, 16, 0.9),
            (3, 7, 0.4),
            (4, 10, 0.6),
            (5, 18, 0.95),
            (6, 9, 0.5),
        )
    ]

    selected = _select_representative_track_rows(
        rows,
        limit=2,
        video_info=VideoInfo(width=64, height=48, fps=5.0, frame_count=6),
    )

    assert [row.frame_index for row in selected] == [2, 5]
