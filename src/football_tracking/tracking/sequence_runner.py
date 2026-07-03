"""Frame sources for video files and MOT-style SportsMOT sequences."""

from __future__ import annotations

import re
from collections.abc import Iterator
from configparser import ConfigParser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class SequenceRunnerError(RuntimeError):
    """Raised when a video or MOT sequence cannot be read."""


@dataclass(frozen=True)
class FrameItem:
    frame_index: int
    image_path: Path | None
    image: Any


@dataclass(frozen=True)
class SequenceSource:
    name: str
    source_path: Path
    source_type: str
    fps: float
    width: int
    height: int
    frame_count: int | None
    frames_dir: Path | None = None
    seqinfo_path: Path | None = None
    video_path: Path | None = None
    warnings: list[str] = field(default_factory=list)


def _numeric_sort_key(path: Path) -> tuple[int, int | str]:
    match = re.search(r"(\d+)$", path.stem)
    if match:
        return 0, int(match.group(1))
    return 1, path.name


def read_seqinfo(path: Path) -> dict[str, Any]:
    parser = ConfigParser()
    parser.optionxform = str
    parser.read(path, encoding="utf-8")
    if "Sequence" not in parser:
        raise SequenceRunnerError(f"seqinfo.ini missing [Sequence]: {path}")
    section = parser["Sequence"]
    required = ("name", "frameRate", "seqLength", "imWidth", "imHeight", "imExt", "imDir")
    missing = [key for key in required if key not in section]
    if missing:
        raise SequenceRunnerError(f"seqinfo.ini missing keys {missing}: {path}")
    return {
        "name": str(section["name"]),
        "frameRate": float(section["frameRate"]),
        "seqLength": int(section["seqLength"]),
        "imWidth": int(section["imWidth"]),
        "imHeight": int(section["imHeight"]),
        "imExt": str(section["imExt"]),
        "imDir": str(section["imDir"]),
    }


def mot_sequence_source(sequence_dir: Path) -> SequenceSource:
    seqinfo_path = sequence_dir / "seqinfo.ini"
    if not seqinfo_path.is_file():
        raise SequenceRunnerError(f"Missing seqinfo.ini: {seqinfo_path}")
    info = read_seqinfo(seqinfo_path)
    frames_dir = sequence_dir / str(info["imDir"])
    if not frames_dir.is_dir():
        raise SequenceRunnerError(f"Missing img directory: {frames_dir}")
    image_paths = sorted(frames_dir.glob(f"*{info['imExt']}"), key=_numeric_sort_key)
    warnings: list[str] = []
    if len(image_paths) != int(info["seqLength"]):
        warnings.append(
            f"Image count {len(image_paths)} does not match seqLength {info['seqLength']}."
        )
    return SequenceSource(
        name=str(info["name"]),
        source_path=sequence_dir,
        source_type="mot_sequence",
        fps=float(info["frameRate"]),
        width=int(info["imWidth"]),
        height=int(info["imHeight"]),
        frame_count=int(info["seqLength"]),
        frames_dir=frames_dir,
        seqinfo_path=seqinfo_path,
        warnings=warnings,
    )


def iter_mot_frames(source: SequenceSource, max_frames: int | None = None) -> Iterator[FrameItem]:
    if source.frames_dir is None:
        raise SequenceRunnerError("MOT source has no frames_dir.")
    info = read_seqinfo(source.seqinfo_path) if source.seqinfo_path else {}
    extension = str(info.get("imExt", ".jpg"))
    image_paths = sorted(source.frames_dir.glob(f"*{extension}"), key=_numeric_sort_key)
    if max_frames is not None:
        image_paths = image_paths[:max_frames]
    try:
        import cv2  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        raise SequenceRunnerError(f"OpenCV is required to read sequence frames: {exc}") from exc
    for position, image_path in enumerate(image_paths, 1):
        frame_index = int(image_path.stem) if image_path.stem.isdigit() else position
        image = cv2.imread(str(image_path))
        if image is None:
            raise SequenceRunnerError(f"OpenCV could not read image: {image_path}")
        yield FrameItem(frame_index=frame_index, image_path=image_path, image=image)


def video_source(path: Path, name: str | None = None) -> SequenceSource:
    if not path.is_file():
        raise SequenceRunnerError(f"Video file does not exist: {path}")
    try:
        import cv2  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        raise SequenceRunnerError(f"OpenCV is required to read video: {exc}") from exc
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise SequenceRunnerError(f"OpenCV could not open video: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        capture.release()
    if width <= 0 or height <= 0:
        raise SequenceRunnerError(f"Video has invalid dimensions: {path}")
    return SequenceSource(
        name=name or path.stem,
        source_path=path,
        source_type="video",
        fps=fps,
        width=width,
        height=height,
        frame_count=frame_count if frame_count > 0 else None,
        video_path=path,
    )


def iter_video_frames(
    source: SequenceSource,
    start_frame: int = 1,
    max_frames: int | None = None,
) -> Iterator[FrameItem]:
    if source.video_path is None:
        raise SequenceRunnerError("Video source has no video_path.")
    try:
        import cv2  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        raise SequenceRunnerError(f"OpenCV is required to read video: {exc}") from exc
    capture = cv2.VideoCapture(str(source.video_path))
    try:
        if not capture.isOpened():
            raise SequenceRunnerError(f"OpenCV could not open video: {source.video_path}")
        if start_frame > 1:
            capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)
        emitted = 0
        frame_index = start_frame
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            yield FrameItem(frame_index=frame_index, image_path=None, image=frame)
            emitted += 1
            frame_index += 1
            if max_frames is not None and emitted >= max_frames:
                break
    finally:
        capture.release()


def read_seqmap(path: Path) -> list[str]:
    if not path.is_file():
        raise SequenceRunnerError(f"Seqmap does not exist: {path}")
    names: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.lower() == "name":
            continue
        names.append(Path(line.replace("\\", "/")).name)
    return names


def discover_mot_sequences(
    mot_root: Path,
    split: str,
    seqmap: Path | None = None,
    max_sequences: int | None = None,
) -> list[SequenceSource]:
    split_dir = mot_root / split
    if not split_dir.is_dir():
        raise SequenceRunnerError(f"Missing MOT split directory: {split_dir}")
    names = (
        read_seqmap(seqmap)
        if seqmap is not None
        else [item.name for item in sorted(split_dir.iterdir()) if item.is_dir()]
    )
    sources: list[SequenceSource] = []
    for name in names:
        source = mot_sequence_source(split_dir / name)
        sources.append(source)
        if max_sequences is not None and len(sources) >= max_sequences:
            break
    return sources


def iter_source_frames(
    source: SequenceSource,
    start_frame: int = 1,
    max_frames: int | None = None,
) -> Iterator[FrameItem]:
    if source.source_type == "video":
        yield from iter_video_frames(source, start_frame=start_frame, max_frames=max_frames)
    elif source.source_type == "mot_sequence":
        yield from iter_mot_frames(source, max_frames=max_frames)
    else:
        raise SequenceRunnerError(f"Unsupported source type: {source.source_type}")
