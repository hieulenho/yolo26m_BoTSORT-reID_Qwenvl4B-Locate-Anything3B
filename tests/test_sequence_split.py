import json
from pathlib import Path

from football_tracking.data.schemas import SequenceInfo
from football_tracking.data.split_sequences import split_sequences


def _sequence(name: str) -> SequenceInfo:
    return SequenceInfo(
        name=name,
        source_path=Path(name),
        frames_dir=Path(name) / "frames",
        annotations_path=Path(name) / "annotations.json",
        fps=25,
        width=64,
        height=48,
        frame_count=3,
        annotations=[],
        metadata={},
    )


def test_split_is_deterministic_by_seed() -> None:
    sequences = [_sequence(f"sequence_{index:03d}") for index in range(1, 6)]

    first = split_sequences(sequences, 0.6, 0.2, 0.2, seed=42)
    second = split_sequences(sequences, 0.6, 0.2, 0.2, seed=42)

    assert first == second


def test_split_has_no_leakage_and_no_missing_sequences() -> None:
    sequences = [_sequence(f"sequence_{index:03d}") for index in range(1, 4)]
    split = split_sequences(sequences, 0.34, 0.33, 0.33, seed=7)
    all_names = split.train + split.val + split.test

    assert sorted(all_names) == [sequence.name for sequence in sequences]
    assert not (set(split.train) & set(split.val))
    assert not (set(split.train) & set(split.test))
    assert not (set(split.val) & set(split.test))


def test_predefined_split_is_respected(tmp_path: Path) -> None:
    path = tmp_path / "splits.json"
    path.write_text(
        json.dumps({"train": ["sequence_001"], "val": ["sequence_002"], "test": ["sequence_003"]}),
        encoding="utf-8",
    )
    sequences = [_sequence(f"sequence_{index:03d}") for index in range(1, 4)]

    split = split_sequences(sequences, 0.34, 0.33, 0.33, seed=99, predefined_split_file=path)

    assert split.train == ["sequence_001"]
    assert split.val == ["sequence_002"]
    assert split.test == ["sequence_003"]
