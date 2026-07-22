from pathlib import Path

from football_tracking.data.manifest import build_manifest_entries, write_dataset_manifest
from football_tracking.data.schemas import SplitManifest
from tests.test_yolo_conversion import _load_sequences


def test_manifest_counts_frames_objects_and_tracks_by_sequence(tmp_path: Path) -> None:
    sequences = _load_sequences()
    split = SplitManifest(1, "sequence", ["sequence_001"], ["sequence_002"], ["sequence_003"])

    payload = write_dataset_manifest(
        sequences,
        split,
        output_dir=tmp_path,
        dataset_name="fixture",
        adapter="soccernet",
        seed=1,
        config_path=Path("configs/legacy/football/data_test.yaml"),
        class_mapping_path=Path("configs/legacy/football/class_mapping.yaml"),
        yolo_output_dir=tmp_path / "yolo",
        mot_output_dir=tmp_path / "mot",
    )

    assert payload["total_frames"] == 9
    assert payload["total_objects"] == 9
    assert payload["total_unique_tracks_by_sequence"] == 4
    assert "python_version" in payload
    first_sequence = payload["sequences"][0]
    assert "empty_frame_count" in first_sequence
    assert "ignored_object_count" in first_sequence
    assert "clipped_box_count" in first_sequence
    assert "invalid_box_count" in first_sequence
    assert (tmp_path / "dataset_manifest.json").is_file()
    assert (tmp_path / "dataset_manifest.csv").is_file()


def test_manifest_entries_do_not_merge_same_track_id_across_sequences(tmp_path: Path) -> None:
    sequences = _load_sequences()
    split = SplitManifest(1, "sequence", ["sequence_001"], ["sequence_002"], ["sequence_003"])

    entries = build_manifest_entries(sequences, split, tmp_path / "yolo", tmp_path / "mot")
    counts = {entry.sequence_name: entry.unique_track_count for entry in entries}

    assert counts["sequence_001"] == 2
    assert counts["sequence_002"] == 1
    assert all(hasattr(entry, "empty_frame_count") for entry in entries)
