from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from football_tracking.vlm.config import DEFAULT_QWEN4B_MODEL_ID, load_vlm_tracking_config
from football_tracking.vlm.tracking_context import read_mot_tracks, run_vlm_analysis


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
    assert list((tmp_path / "outputs" / "vlm" / "crops").glob("track_*/*.jpg"))


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
