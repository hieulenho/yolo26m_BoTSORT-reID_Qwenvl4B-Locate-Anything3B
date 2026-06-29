"""Dataset discovery helpers."""

from __future__ import annotations

import json
from pathlib import Path

from football_tracking.data.schemas import SequenceCandidate


class DatasetDiscoveryError(RuntimeError):
    """Raised when raw dataset discovery fails."""


FRAME_DIR_NAMES = ("frames", "img1", "images")
ANNOTATION_FILE_NAMES = ("annotations.json", "annotation.json", "gt.json")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


def _find_frames_dir(sequence_dir: Path) -> Path | None:
    for name in FRAME_DIR_NAMES:
        candidate = sequence_dir / name
        if candidate.is_dir() and any(path.is_file() for path in candidate.iterdir()):
            return candidate
    image_files = [
        path
        for path in sequence_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".ppm"}
    ]
    return sequence_dir if image_files else None


def _find_annotation_file(sequence_dir: Path) -> Path | None:
    for name in ANNOTATION_FILE_NAMES:
        candidate = sequence_dir / name
        if candidate.is_file():
            return candidate
    gt_txt = sequence_dir / "gt" / "gt.txt"
    return gt_txt if gt_txt.is_file() else None


def _find_video_file(sequence_dir: Path) -> Path | None:
    for path in sequence_dir.iterdir():
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            return path
    return None


def _read_metadata(sequence_dir: Path) -> tuple[Path | None, dict[str, object]]:
    metadata_path = sequence_dir / "metadata.json"
    if not metadata_path.is_file():
        return None, {}
    try:
        loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetDiscoveryError(
            f"Metadata file is not valid JSON: {metadata_path}: {exc}"
        ) from exc
    if not isinstance(loaded, dict):
        raise DatasetDiscoveryError(f"Metadata file must contain a JSON object: {metadata_path}")
    return metadata_path, loaded


def discover_sequence_candidates(raw_dir: str | Path) -> list[SequenceCandidate]:
    root = Path(raw_dir)
    if not root.is_dir():
        raise DatasetDiscoveryError(f"Raw dataset directory does not exist: {root}")

    sequence_dirs = [path for path in sorted(root.iterdir()) if path.is_dir()]
    if not sequence_dirs:
        raise DatasetDiscoveryError(
            f"No sequence directories found in raw dataset directory: {root}"
        )

    candidates: list[SequenceCandidate] = []
    errors: list[str] = []
    for sequence_dir in sequence_dirs:
        frames_dir = _find_frames_dir(sequence_dir)
        annotations_path = _find_annotation_file(sequence_dir)
        metadata_path, metadata = _read_metadata(sequence_dir)
        if frames_dir is None:
            errors.append(f"{sequence_dir.name}: missing frame directory or images")
            continue
        if annotations_path is None:
            errors.append(f"{sequence_dir.name}: missing annotation file")
            continue
        try:
            annotations_path.read_text(encoding="utf-8")
        except OSError as exc:
            errors.append(f"{sequence_dir.name}: annotation file is not readable: {exc}")
            continue

        seqinfo_path = sequence_dir / "seqinfo.ini"
        candidates.append(
            SequenceCandidate(
                name=sequence_dir.name,
                source_path=sequence_dir,
                frames_dir=frames_dir,
                annotations_path=annotations_path,
                video_path=_find_video_file(sequence_dir),
                metadata_path=metadata_path,
                seqinfo_path=seqinfo_path if seqinfo_path.is_file() else None,
                metadata=metadata,
            )
        )

    if not candidates:
        details = "; ".join(errors) if errors else "no recognizable sequences"
        raise DatasetDiscoveryError(f"No valid sequence candidates found in {root}: {details}")

    return candidates
