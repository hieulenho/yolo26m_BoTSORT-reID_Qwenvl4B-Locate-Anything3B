from pathlib import Path

import yaml

from football_tracking.data.class_mapping import apply_mapping_to_object, load_class_mapping
from football_tracking.data.convert_yolo import convert_to_yolo
from football_tracking.data.schemas import FrameAnnotation, SequenceInfo, SplitManifest
from football_tracking.data.soccernet_adapter import SoccerNetAdapter

FIXTURE_ROOT = Path("tests/fixtures/mini_tracking_dataset")


def _load_sequences() -> list[SequenceInfo]:
    adapter = SoccerNetAdapter()
    mapping = load_class_mapping("configs/class_mapping.yaml")
    sequences: list[SequenceInfo] = []
    for candidate in adapter.discover_sequences(FIXTURE_ROOT):
        sequence = adapter.load_sequence(candidate.source_path)
        frames = [
            FrameAnnotation(
                sequence_name=frame.sequence_name,
                frame_index=frame.frame_index,
                image_path=frame.image_path,
                width=frame.width,
                height=frame.height,
                objects=[apply_mapping_to_object(obj, mapping) for obj in frame.objects],
            )
            for frame in sequence.annotations
        ]
        sequences.append(
            SequenceInfo(
                name=sequence.name,
                source_path=sequence.source_path,
                frames_dir=sequence.frames_dir,
                annotations_path=sequence.annotations_path,
                fps=sequence.fps,
                width=sequence.width,
                height=sequence.height,
                frame_count=sequence.frame_count,
                annotations=frames,
                metadata=sequence.metadata,
            )
        )
    return sequences


def test_yolo_conversion_outputs_expected_files_and_coordinates(tmp_path: Path) -> None:
    sequences = _load_sequences()
    split = SplitManifest(
        seed=1,
        strategy="sequence",
        train=["sequence_001", "sequence_002", "sequence_003"],
        val=[],
        test=[],
    )

    convert_to_yolo(sequences, split, tmp_path, {0: "player"}, overwrite=True)

    label = tmp_path / "labels" / "train" / "sequence_001_000001.txt"
    lines = label.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "0 0.250000 0.437500 0.187500 0.541667"
    assert all(not line.startswith("99 ") for line in lines)


def test_yolo_writes_empty_label_for_frame_without_player(tmp_path: Path) -> None:
    sequences = _load_sequences()
    split = SplitManifest(1, "sequence", ["sequence_003"], [], [])

    convert_to_yolo([sequences[2]], split, tmp_path, {0: "player"}, overwrite=True)

    label = tmp_path / "labels" / "train" / "sequence_003_000001.txt"
    assert label.is_file()
    assert label.read_text(encoding="utf-8") == ""


def test_yolo_dataset_yaml_and_unique_image_names(tmp_path: Path) -> None:
    sequences = _load_sequences()
    split = SplitManifest(1, "sequence", ["sequence_001", "sequence_002"], [], [])

    convert_to_yolo(sequences[:2], split, tmp_path, {0: "player"}, overwrite=True)

    dataset_yaml = yaml.safe_load((tmp_path / "dataset.yaml").read_text(encoding="utf-8"))
    assert dataset_yaml["nc"] == 1
    assert dataset_yaml["names"][0] == "player"
    assert (tmp_path / "images" / "train" / "sequence_001_000001.ppm").exists()
    assert (tmp_path / "images" / "train" / "sequence_002_000001.ppm").exists()
