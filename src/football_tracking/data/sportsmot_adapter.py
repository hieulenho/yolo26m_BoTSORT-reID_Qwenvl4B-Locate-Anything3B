"""SportsMOT football-only preparation helpers."""

from __future__ import annotations

import csv
import json
import math
import random
import re
import shutil
import sys
from configparser import ConfigParser
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from football_tracking.data.convert_yolo import convert_to_yolo
from football_tracking.data.schemas import (
    BoundingBoxXYXY,
    FrameAnnotation,
    ObjectAnnotation,
    SequenceInfo,
    SplitManifest,
    ValidationIssue,
    ValidationReport,
)
from football_tracking.paths import get_project_root, resolve_project_path

FOOTBALL_LIST_CANDIDATES = (
    Path("splits_txt/football.txt"),
    Path("football.txt"),
    Path("metadata/football.txt"),
)
LOCAL_SPLIT_SEED = 42


class SportsMotError(RuntimeError):
    """Raised when SportsMOT data cannot be prepared safely."""


@dataclass(frozen=True)
class SportsMotConfig:
    project_root: Path
    config_path: Path
    raw_dir: Path
    interim_dir: Path
    yolo_output_dir: Path
    yolo_smoke_output_dir: Path
    mot_output_dir: Path
    seed: int
    local_val_ratio: float
    overwrite: bool
    dry_run: bool
    prefer_symlink: bool
    decimal_places: int
    smoke_max_train_sequences: int
    smoke_max_val_sequences: int
    smoke_max_train_frames: int
    smoke_max_val_frames: int


@dataclass(frozen=True)
class SequenceRecord:
    name: str
    official_split: str
    sequence_dir: Path


@dataclass(frozen=True)
class SeqInfo:
    name: str
    frame_rate: float
    seq_length: int
    image_width: int
    image_height: int
    image_extension: str
    image_dir_name: str


def _mapping(value: Any, section: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SportsMotError(f"{section} must be a mapping.")
    return value


def _resolve_config_path_value(value: str | Path, project_root: Path) -> Path:
    raw = Path(value)
    if raw.is_absolute():
        return raw.resolve()
    return resolve_project_path(raw, project_root=project_root)


def load_sportsmot_config(config_path: str | Path) -> SportsMotConfig:
    project_root = get_project_root()
    resolved_config = _resolve_config_path_value(config_path, project_root)
    if not resolved_config.is_file():
        raise SportsMotError(f"SportsMOT config does not exist: {resolved_config}")
    raw = yaml.safe_load(resolved_config.read_text(encoding="utf-8"))
    root = _mapping(raw, "config root")
    dataset = _mapping(root.get("dataset"), "dataset")
    split = _mapping(root.get("split", {}), "split")
    yolo = _mapping(root.get("yolo"), "yolo")
    mot = _mapping(root.get("mot"), "mot")
    smoke = _mapping(root.get("smoke", {}), "smoke")
    runtime = _mapping(root.get("runtime", {}), "runtime")

    return SportsMotConfig(
        project_root=project_root,
        config_path=resolved_config,
        raw_dir=_resolve_config_path_value(str(dataset["raw_dir"]), project_root),
        interim_dir=_resolve_config_path_value(str(dataset["interim_dir"]), project_root),
        yolo_output_dir=_resolve_config_path_value(str(yolo["output_dir"]), project_root),
        yolo_smoke_output_dir=_resolve_config_path_value(
            str(yolo.get("smoke_output_dir", "data/yolo/sportsmot_football_smoke")),
            project_root,
        ),
        mot_output_dir=_resolve_config_path_value(str(mot["output_dir"]), project_root),
        seed=int(split.get("seed", LOCAL_SPLIT_SEED)),
        local_val_ratio=float(split.get("local_val_ratio", 0.20)),
        overwrite=bool(runtime.get("overwrite", False)),
        dry_run=bool(runtime.get("dry_run", False)),
        prefer_symlink=bool(yolo.get("prefer_symlink", True)),
        decimal_places=int(yolo.get("decimal_places", 6)),
        smoke_max_train_sequences=int(smoke.get("max_train_sequences", 2)),
        smoke_max_val_sequences=int(smoke.get("max_val_sequences", 1)),
        smoke_max_train_frames=int(smoke.get("max_train_frames", 100)),
        smoke_max_val_frames=int(smoke.get("max_val_frames", 50)),
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _issue_to_dict(issue: ValidationIssue) -> dict[str, Any]:
    return {
        "severity": issue.severity,
        "code": issue.code,
        "message": issue.message,
        "sequence_name": issue.sequence_name,
        "frame_index": issue.frame_index,
        "track_id": issue.track_id,
        "path": str(issue.path) if issue.path is not None else None,
    }


def write_validation_report(report: ValidationReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")


def find_sportsmot_root(raw_dir: Path) -> Path:
    candidates = [
        raw_dir,
        raw_dir / "sportsmot",
        raw_dir / "SportsMOT",
        raw_dir / "dataset",
    ]
    candidates.extend(item for item in raw_dir.glob("*") if item.is_dir())
    candidates.extend(raw_dir.glob("*/dataset"))
    for candidate in candidates:
        if (candidate / "train").is_dir() and (candidate / "val").is_dir():
            return candidate
    raise SportsMotError(f"Could not find SportsMOT root under {raw_dir}")


def find_football_list(root: Path) -> Path:
    for relative_path in FOOTBALL_LIST_CANDIDATES:
        candidate = root / relative_path
        if candidate.is_file():
            return candidate
    txt_files = sorted(str(path.relative_to(root)) for path in root.rglob("*.txt"))
    raise SportsMotError(
        "Could not find official football.txt. Candidate .txt files near dataset root: "
        f"{txt_files}"
    )


def read_football_sequences(root: Path) -> list[str]:
    path = find_football_list(root)
    names: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        names.append(Path(line.replace("\\", "/")).name)
    return sorted(dict.fromkeys(names))


def discover_official_sequences(root: Path, split: str) -> dict[str, Path]:
    split_dir = root / split
    if not split_dir.is_dir():
        raise SportsMotError(f"Missing official SportsMOT split directory: {split_dir}")
    return {
        item.name: item
        for item in sorted(split_dir.iterdir())
        if item.is_dir()
        and (item / "img1").is_dir()
        and (item / "seqinfo.ini").is_file()
        and (item / "gt" / "gt.txt").is_file()
    }


def parse_seqinfo(path: Path) -> SeqInfo:
    parser = ConfigParser()
    parser.optionxform = str
    parser.read(path, encoding="utf-8")
    if "Sequence" not in parser:
        raise SportsMotError(f"seqinfo.ini missing [Sequence]: {path}")
    section = parser["Sequence"]
    required = ("name", "frameRate", "seqLength", "imWidth", "imHeight", "imExt", "imDir")
    missing = [key for key in required if key not in section]
    if missing:
        raise SportsMotError(f"seqinfo.ini missing keys {missing}: {path}")
    return SeqInfo(
        name=str(section["name"]),
        frame_rate=float(section["frameRate"]),
        seq_length=int(section["seqLength"]),
        image_width=int(section["imWidth"]),
        image_height=int(section["imHeight"]),
        image_extension=str(section["imExt"]),
        image_dir_name=str(section["imDir"]),
    )


def _parse_gt_line(
    line: str,
    path: Path,
    line_number: int,
) -> tuple[int, int, float, float, float, float, float, int, float]:
    fields = [field.strip() for field in line.split(",")]
    if len(fields) != 9:
        raise SportsMotError(f"{path}:{line_number} must contain 9 MOT fields.")
    frame = int(float(fields[0]))
    track_id = int(float(fields[1]))
    left = float(fields[2])
    top = float(fields[3])
    width = float(fields[4])
    height = float(fields[5])
    confidence = float(fields[6])
    class_id = int(float(fields[7]))
    visibility = float(fields[8])
    values = (left, top, width, height, confidence, visibility)
    if any(not math.isfinite(value) for value in values):
        raise SportsMotError(f"{path}:{line_number} contains NaN or infinity.")
    return frame, track_id, left, top, width, height, confidence, class_id, visibility


def _validate_gt_row(
    row: tuple[int, int, float, float, float, float, float, int, float],
    seqinfo: SeqInfo,
    sequence_dir: Path,
    path: Path,
    line_number: int,
) -> None:
    frame, track_id, _left, _top, width, height, confidence, _class_id, visibility = row
    if frame < 1:
        raise SportsMotError(f"{path}:{line_number} frame must be >= 1.")
    if frame > seqinfo.seq_length:
        raise SportsMotError(f"{path}:{line_number} frame exceeds seqLength.")
    if track_id < 0:
        raise SportsMotError(f"{path}:{line_number} track_id must be non-negative.")
    if width <= 0 or height <= 0:
        raise SportsMotError(f"{path}:{line_number} width/height must be positive.")
    if not math.isfinite(confidence):
        raise SportsMotError(f"{path}:{line_number} confidence is invalid.")
    if visibility < 0.0 or visibility > 1.0:
        raise SportsMotError(f"{path}:{line_number} visibility must be in [0, 1].")
    image_path = sequence_dir / seqinfo.image_dir_name / f"{frame:06d}{seqinfo.image_extension}"
    if not image_path.is_file():
        raise SportsMotError(f"{path}:{line_number} image does not exist: {image_path}")


def validate_sequence(record: SequenceRecord) -> ValidationReport:
    issues: list[ValidationIssue] = []
    seqinfo_path = record.sequence_dir / "seqinfo.ini"
    gt_path = record.sequence_dir / "gt" / "gt.txt"
    try:
        seqinfo = parse_seqinfo(seqinfo_path)
        seen: set[tuple[int, int]] = set()
        for line_number, line in enumerate(gt_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = _parse_gt_line(line, gt_path, line_number)
            _validate_gt_row(row, seqinfo, record.sequence_dir, gt_path, line_number)
            key = (row[0], row[1])
            if key in seen:
                raise SportsMotError(
                    f"{gt_path}:{line_number} duplicate frame-track pair: {key}"
                )
            seen.add(key)
    except Exception as exc:  # noqa: BLE001
        issues.append(
            ValidationIssue(
                severity="ERROR",
                code="sportsmot_sequence_invalid",
                message=str(exc),
                sequence_name=record.name,
                path=record.sequence_dir,
            )
        )
    return ValidationReport(issues)


def validate_records(records: list[SequenceRecord]) -> ValidationReport:
    report = ValidationReport([])
    for record in records:
        report.extend(validate_sequence(record))
    return report


def football_records(root: Path) -> tuple[list[SequenceRecord], dict[str, Any]]:
    football = set(read_football_sequences(root))
    official_train = discover_official_sequences(root, "train")
    official_val = discover_official_sequences(root, "val")
    train_names = sorted(football & set(official_train))
    val_names = sorted(football & set(official_val))
    missing = sorted(football - set(official_train) - set(official_val))
    records = [
        *(SequenceRecord(name, "train", official_train[name]) for name in train_names),
        *(SequenceRecord(name, "val", official_val[name]) for name in val_names),
    ]
    summary = {
        "football_sequences": sorted(football),
        "official_train_football": train_names,
        "official_val_football": val_names,
        "missing": missing,
        "non_football_train": sorted(set(official_train) - football),
        "non_football_val": sorted(set(official_val) - football),
    }
    return records, summary


def _group_name(sequence_name: str) -> tuple[str, str | None]:
    match = re.match(r"(.+)_c\d+$", sequence_name)
    if match:
        return match.group(1), None
    return sequence_name, f"Could not infer source-video group for {sequence_name}."


def create_local_split(
    records: list[SequenceRecord],
    seed: int,
    local_val_ratio: float,
) -> tuple[SplitManifest, dict[str, list[str]], list[str]]:
    train_pool = sorted(record.name for record in records if record.official_split == "train")
    test = sorted(record.name for record in records if record.official_split == "val")
    groups: dict[str, list[str]] = {}
    warnings: list[str] = []
    for name in train_pool:
        group, warning = _group_name(name)
        groups.setdefault(group, []).append(name)
        if warning:
            warnings.append(warning)
    group_names = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(group_names)
    target_val_sequences = int(round(len(train_pool) * local_val_ratio))
    if len(train_pool) > 1:
        target_val_sequences = max(1, target_val_sequences)
    val_groups: set[str] = set()
    val_count = 0
    for group in group_names:
        if val_count >= target_val_sequences:
            break
        val_groups.add(group)
        val_count += len(groups[group])
    val = sorted(name for group in val_groups for name in groups[group])
    train = sorted(name for name in train_pool if name not in val)
    return SplitManifest(seed, "grouped_sequence_split", train, val, test), groups, warnings


def load_sequence(record: SequenceRecord) -> SequenceInfo:
    seqinfo = parse_seqinfo(record.sequence_dir / "seqinfo.ini")
    gt_path = record.sequence_dir / "gt" / "gt.txt"
    rows_by_frame: dict[int, list[ObjectAnnotation]] = {
        frame: [] for frame in range(1, seqinfo.seq_length + 1)
    }
    for line_number, line in enumerate(gt_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = _parse_gt_line(line, gt_path, line_number)
        _validate_gt_row(row, seqinfo, record.sequence_dir, gt_path, line_number)
        frame, track_id, left, top, width, height, confidence, class_id, visibility = row
        rows_by_frame[frame].append(
            ObjectAnnotation(
                frame_index=frame,
                track_id=track_id,
                source_class=str(class_id),
                target_class="player",
                target_class_id=0,
                bbox_xyxy=BoundingBoxXYXY(left, top, left + width, top + height),
                confidence=confidence,
                visibility=visibility,
                metadata={"mot_class": class_id, "official_split": record.official_split},
            )
        )
    frames = [
        FrameAnnotation(
            sequence_name=record.name,
            frame_index=frame,
            image_path=record.sequence_dir
            / seqinfo.image_dir_name
            / f"{frame:06d}{seqinfo.image_extension}",
            width=seqinfo.image_width,
            height=seqinfo.image_height,
            objects=rows_by_frame.get(frame, []),
        )
        for frame in range(1, seqinfo.seq_length + 1)
    ]
    return SequenceInfo(
        name=record.name,
        source_path=record.sequence_dir,
        frames_dir=record.sequence_dir / seqinfo.image_dir_name,
        video_path=None,
        annotations_path=gt_path,
        fps=seqinfo.frame_rate,
        width=seqinfo.image_width,
        height=seqinfo.image_height,
        frame_count=seqinfo.seq_length,
        annotations=frames,
        metadata={"official_split": record.official_split},
    )


def _sequence_stats(sequence: SequenceInfo) -> dict[str, Any]:
    objects = [obj for frame in sequence.annotations for obj in frame.objects]
    tracks = {obj.track_id for obj in objects}
    return {
        "sequence_name": sequence.name,
        "official_split": sequence.metadata.get("official_split"),
        "frame_count": sequence.frame_count,
        "box_count": len(objects),
        "track_count": len(tracks),
        "source_path": str(sequence.source_path),
    }


def _split_stats(sequences: list[SequenceInfo], split_manifest: SplitManifest) -> dict[str, Any]:
    stats: dict[str, dict[str, Any]] = {}
    for split_name, names in split_manifest.as_mapping().items():
        selected = [sequence for sequence in sequences if sequence.name in names]
        objects = [
            obj
            for sequence in selected
            for frame in sequence.annotations
            for obj in frame.objects
        ]
        tracks = {
            (sequence.name, obj.track_id)
            for sequence in selected
            for frame in sequence.annotations
            for obj in frame.objects
        }
        widths = [obj.bbox_xyxy.x2 - obj.bbox_xyxy.x1 for obj in objects]
        heights = [obj.bbox_xyxy.y2 - obj.bbox_xyxy.y1 for obj in objects]
        stats[split_name] = {
            "sequences": len(selected),
            "frames": sum(sequence.frame_count for sequence in selected),
            "boxes": len(objects),
            "tracks": len(tracks),
            "empty_frames": sum(
                1 for sequence in selected for frame in sequence.annotations if not frame.objects
            ),
            "box_width_min": min(widths) if widths else None,
            "box_width_max": max(widths) if widths else None,
            "box_height_min": min(heights) if heights else None,
            "box_height_max": max(heights) if heights else None,
        }
    return stats


def _write_per_sequence_csv(sequences: list[SequenceInfo], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sequence_name",
                "official_split",
                "frame_count",
                "box_count",
                "track_count",
                "source_path",
            ],
        )
        writer.writeheader()
        for sequence in sorted(sequences, key=lambda item: item.name):
            writer.writerow(_sequence_stats(sequence))


def _link_or_copy(source: Path, destination: Path, prefer_symlink: bool) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    if prefer_symlink:
        try:
            destination.symlink_to(source)
            return "symlink"
        except OSError:
            pass
    try:
        destination.hardlink_to(source)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def write_mot_football_dataset(
    records: list[SequenceRecord],
    split_manifest: SplitManifest,
    output_dir: Path,
    overwrite: bool,
    prefer_symlink: bool,
) -> dict[str, Any]:
    if output_dir.exists() and not overwrite:
        meaningful = [child for child in output_dir.rglob("*") if child.name != ".gitkeep"]
        if meaningful:
            raise SportsMotError(f"MOT output already exists and overwrite=false: {output_dir}")
    if output_dir.exists() and overwrite:
        for relative_path in ("train", "val", "test", "seqmaps", "manifest.json"):
            path = output_dir / relative_path
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    record_by_name = {record.name: record for record in records}
    link_methods: dict[str, int] = {}
    seqmaps: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    sources: dict[str, str] = {}
    for split_name, names in split_manifest.as_mapping().items():
        for name in names:
            record = record_by_name[name]
            seqmaps[split_name].append(name)
            destination = output_dir / split_name / name
            (destination / "gt").mkdir(parents=True, exist_ok=True)
            (destination / "img1").mkdir(parents=True, exist_ok=True)
            shutil.copy2(record.sequence_dir / "gt" / "gt.txt", destination / "gt" / "gt.txt")
            shutil.copy2(record.sequence_dir / "seqinfo.ini", destination / "seqinfo.ini")
            for image in sorted((record.sequence_dir / "img1").iterdir()):
                if image.is_file():
                    method = _link_or_copy(image, destination / "img1" / image.name, prefer_symlink)
                    link_methods[method] = link_methods.get(method, 0) + 1
            sources[name] = str(record.sequence_dir)
    seqmap_dir = output_dir / "seqmaps"
    seqmap_dir.mkdir(parents=True, exist_ok=True)
    for split_name, names in seqmaps.items():
        content = "name\n" + "\n".join(sorted(names)) + ("\n" if names else "")
        (seqmap_dir / f"{split_name}.txt").write_text(content, encoding="utf-8")
    manifest = {"seqmaps": seqmaps, "link_methods": link_methods, "sources": sources}
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def _select_evenly(frames: list[FrameAnnotation], limit: int) -> list[FrameAnnotation]:
    if len(frames) <= limit:
        return frames
    if limit <= 0:
        return []
    if limit == 1:
        return [frames[0]]
    indexes = sorted({round(index * (len(frames) - 1) / (limit - 1)) for index in range(limit)})
    return [frames[index] for index in indexes]


def _smoke_sequences(
    sequences: list[SequenceInfo],
    split_manifest: SplitManifest,
    max_train_sequences: int,
    max_val_sequences: int,
    max_train_frames: int,
    max_val_frames: int,
) -> tuple[list[SequenceInfo], SplitManifest, dict[str, Any]]:
    by_name = {sequence.name: sequence for sequence in sequences}
    train_names = split_manifest.train[:max_train_sequences]
    val_names = split_manifest.val[:max_val_sequences]
    selected: list[SequenceInfo] = []
    smoke_stats: dict[str, Any] = {"train": {}, "val": {}}
    for split_name, names, max_frames in (
        ("train", train_names, max_train_frames),
        ("val", val_names, max_val_frames),
    ):
        per_sequence_limit = max(1, max_frames // max(1, len(names))) if names else 0
        for name in names:
            sequence = by_name[name]
            frames = _select_evenly(sequence.annotations, per_sequence_limit)
            selected.append(
                SequenceInfo(
                    name=sequence.name,
                    source_path=sequence.source_path,
                    frames_dir=sequence.frames_dir,
                    video_path=sequence.video_path,
                    annotations_path=sequence.annotations_path,
                    fps=sequence.fps,
                    width=sequence.width,
                    height=sequence.height,
                    frame_count=len(frames),
                    annotations=frames,
                    metadata=sequence.metadata,
                )
            )
            smoke_stats[split_name][name] = [frame.frame_index for frame in frames]
    return (
        selected,
        SplitManifest(
            split_manifest.seed,
            "sportsmot_smoke_sequence_subset",
            train_names,
            val_names,
            [],
        ),
        smoke_stats,
    )


def _write_dataset_manifest(
    config: SportsMotConfig,
    root: Path,
    sequences: list[SequenceInfo],
    split_manifest: SplitManifest,
    football_summary: dict[str, Any],
    validation_report: ValidationReport,
    groups: dict[str, list[str]],
    warnings: list[str],
) -> dict[str, Any]:
    payload = {
        "dataset": "SportsMOT football",
        "raw_root": str(root),
        "yolo_root": str(config.yolo_output_dir),
        "mot_root": str(config.mot_output_dir),
        "seed": split_manifest.seed,
        "strategy": split_manifest.strategy,
        "football_summary": football_summary,
        "splits": split_manifest.as_mapping(),
        "groups": groups,
        "warnings": warnings,
        "validation": validation_report.to_dict()["summary"],
        "created_at": datetime.now(UTC).isoformat(),
        "python_version": sys.version,
    }
    _write_json(config.interim_dir / "dataset_manifest.json", payload)
    return payload


def prepare_sportsmot(
    config_path: str | Path,
    overwrite: bool | None = None,
    dry_run: bool | None = None,
) -> dict[str, Any]:
    config = load_sportsmot_config(config_path)
    if overwrite is not None:
        config = SportsMotConfig(**{**config.__dict__, "overwrite": overwrite})
    if dry_run is not None:
        config = SportsMotConfig(**{**config.__dict__, "dry_run": dry_run})
    root = find_sportsmot_root(config.raw_dir)
    records, football_summary = football_records(root)
    if not records:
        raise SportsMotError("No football sequences found in SportsMOT train/val.")
    validation_report = validate_records(records)
    write_validation_report(
        validation_report,
        config.project_root / "outputs/metrics/sportsmot_download_validation.json",
    )
    if validation_report.has_errors:
        raise SportsMotError(
            "SportsMOT validation failed. See outputs/metrics/sportsmot_download_validation.json."
        )
    sequences = [load_sequence(record) for record in records]
    split_manifest, groups, split_warnings = create_local_split(
        records,
        seed=config.seed,
        local_val_ratio=config.local_val_ratio,
    )
    if not split_manifest.train or not split_manifest.val:
        raise SportsMotError("Local SportsMOT train/val split is empty.")
    if config.dry_run:
        return {
            "dry_run": True,
            "root": str(root),
            "football_summary": football_summary,
            "splits": split_manifest.as_mapping(),
        }
    config.interim_dir.mkdir(parents=True, exist_ok=True)
    _write_json(config.interim_dir / "football_sequences.json", football_summary)
    _write_json(
        config.interim_dir / "splits.json",
        {
            "seed": config.seed,
            "strategy": "grouped_sequence_split",
            "source": {
                "train_pool": "SportsMOT official train football",
                "test": "SportsMOT official val football",
            },
            **split_manifest.as_mapping(),
            "groups": groups,
            "warnings": split_warnings,
        },
    )
    yolo_stats = convert_to_yolo(
        sequences,
        split_manifest,
        output_dir=config.yolo_output_dir,
        class_names={0: "player"},
        decimal_places=config.decimal_places,
        copy_images=False,
        prefer_symlink=config.prefer_symlink,
        clip_boxes=True,
        overwrite=config.overwrite,
        dry_run=False,
    )
    (config.yolo_output_dir / "splits.json").write_text(
        json.dumps(split_manifest.as_mapping(), indent=2),
        encoding="utf-8",
    )
    mot_manifest = write_mot_football_dataset(
        records,
        split_manifest,
        config.mot_output_dir,
        overwrite=config.overwrite,
        prefer_symlink=config.prefer_symlink,
    )
    smoke_sequences, smoke_split, smoke_stats = _smoke_sequences(
        sequences,
        split_manifest,
        max_train_sequences=config.smoke_max_train_sequences,
        max_val_sequences=config.smoke_max_val_sequences,
        max_train_frames=config.smoke_max_train_frames,
        max_val_frames=config.smoke_max_val_frames,
    )
    smoke_yolo_stats = convert_to_yolo(
        smoke_sequences,
        smoke_split,
        output_dir=config.yolo_smoke_output_dir,
        class_names={0: "player"},
        decimal_places=config.decimal_places,
        copy_images=False,
        prefer_symlink=config.prefer_symlink,
        clip_boxes=True,
        overwrite=True,
        dry_run=False,
    )
    _write_json(
        config.interim_dir / "smoke_splits.json",
        {"splits": smoke_split.as_mapping(), "frames": smoke_stats},
    )
    dataset_manifest = _write_dataset_manifest(
        config,
        root,
        sequences,
        split_manifest,
        football_summary,
        validation_report,
        groups,
        split_warnings,
    )
    audit = {
        "dataset": "SportsMOT football",
        "raw_root": str(root),
        "yolo_root": str(config.yolo_output_dir),
        "mot_root": str(config.mot_output_dir),
        "created_at": datetime.now(UTC).isoformat(),
        "python_version": sys.version,
        "split_seed": config.seed,
        "split_stats": _split_stats(sequences, split_manifest),
        "football_sequence_count": len(sequences),
        "invalid_box_count": 0,
        "clipped_box_count": 0,
        "duplicate_count": 0,
    }
    _write_json(config.project_root / "outputs/metrics/sportsmot_football_audit.json", audit)
    _write_per_sequence_csv(
        sequences,
        config.project_root / "outputs/metrics/sportsmot_football_per_sequence.csv",
    )
    return {
        "root": str(root),
        "football_sequence_count": len(sequences),
        "splits": split_manifest.as_mapping(),
        "yolo": yolo_stats,
        "smoke_yolo": smoke_yolo_stats,
        "mot": mot_manifest,
        "manifest": dataset_manifest,
        "audit": audit,
    }


def validate_sportsmot(config_path: str | Path) -> ValidationReport:
    config = load_sportsmot_config(config_path)
    root = find_sportsmot_root(config.raw_dir)
    records, _summary = football_records(root)
    report = validate_records(records)
    write_validation_report(
        report,
        config.project_root / "outputs/metrics/sportsmot_download_validation.json",
    )
    return report


def audit_sportsmot(config_path: str | Path) -> dict[str, Any]:
    config = load_sportsmot_config(config_path)
    manifest_path = config.project_root / "outputs/metrics/sportsmot_football_audit.json"
    if manifest_path.is_file():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    prepared = prepare_sportsmot(config_path, overwrite=False, dry_run=False)
    return prepared["audit"]
