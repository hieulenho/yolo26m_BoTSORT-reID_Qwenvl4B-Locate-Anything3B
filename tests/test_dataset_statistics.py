from football_tracking.data.schemas import SplitManifest
from football_tracking.data.statistics import compute_dataset_statistics
from tests.test_yolo_conversion import _load_sequences


def test_dataset_statistics_counts_fixture_frames_objects_and_tracks() -> None:
    sequences = _load_sequences()
    split = SplitManifest(1, "sequence", ["sequence_001"], ["sequence_002"], ["sequence_003"])

    stats = compute_dataset_statistics(sequences, split)

    assert stats["totals"]["sequence_count"] == 3
    assert stats["totals"]["frame_count"] == 9
    assert stats["totals"]["player_box_count"] == 8
    assert stats["totals"]["unique_tracks_global_sequence_track_id"] == 4
    assert stats["track_statistics"]["duplicate_frame_track_count"] == 1
    assert stats["bbox_statistics"]["invalid_box_count"] == 1


def test_dataset_statistics_does_not_merge_same_track_id_across_sequences() -> None:
    sequences = _load_sequences()
    split = SplitManifest(1, "sequence", ["sequence_001", "sequence_002"], [], ["sequence_003"])

    stats = compute_dataset_statistics(sequences, split)
    tracks = {(record["sequence_name"], record["track_id"]) for record in stats["tracks"]}

    assert ("sequence_001", 1) in tracks
    assert ("sequence_002", 1) in tracks
    assert len(tracks) == 4


def test_dataset_statistics_split_counts_and_empty_dataset() -> None:
    sequences = _load_sequences()
    split = SplitManifest(1, "sequence", ["sequence_001"], ["sequence_002"], ["sequence_003"])

    stats = compute_dataset_statistics(sequences, split)
    per_split = {record["split"]: record for record in stats["per_split"]}

    assert per_split["train"]["sequence_count"] == 1
    assert per_split["val"]["object_count"] == 2
    assert per_split["test"]["track_count"] == 1

    empty = compute_dataset_statistics([], None)
    assert empty["totals"]["sequence_count"] == 0
    assert empty["bbox_statistics"]["width"]["mean"] is None
