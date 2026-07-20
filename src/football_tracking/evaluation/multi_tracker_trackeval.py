"""Official TrackEval integration point for tracker experiments."""

from __future__ import annotations

import configparser
import importlib.util
import json
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from football_tracking.evaluation.experiment_metrics import empty_tracking_metrics


@dataclass(frozen=True)
class TrackEvalTrackerResult:
    tracker_name: str
    available: bool
    metrics: dict[str, float | int | None]
    per_sequence: dict[str, dict[str, float | int | None]] = field(default_factory=dict)
    command: list[str] = field(default_factory=list)
    version: str | None = None
    reason: str | None = None
    raw_output_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tracker_name": self.tracker_name,
            "available": self.available,
            "metrics": self.metrics,
            "per_sequence": self.per_sequence,
            "command": self.command,
            "version": self.version,
            "reason": self.reason,
            "raw_output_path": str(self.raw_output_path) if self.raw_output_path else None,
        }


def trackeval_available() -> bool:
    return importlib.util.find_spec("trackeval") is not None


def trackeval_version() -> str | None:
    try:
        import trackeval  # type: ignore[import-not-found]

        return getattr(trackeval, "__version__", None) or getattr(trackeval, "__file__", None)
    except Exception:  # noqa: BLE001
        return None


def evaluate_trackers_with_trackeval(
    tracker_names: list[str],
    gt_root: Path,
    trackers_root: Path,
    split: str,
    seqmap: Path | None,
    output_root: Path,
    metrics: tuple[str, ...],
    allow_partial_sequences: bool = False,
) -> dict[str, TrackEvalTrackerResult]:
    """Run official TrackEval when installed, otherwise return explicit null metrics."""
    output_root.mkdir(parents=True, exist_ok=True)
    if not trackeval_available():
        return {
            tracker_name: _missing_trackeval_result(
                tracker_name,
                output_root,
                "Python package 'trackeval' is not installed.",
            )
            for tracker_name in tracker_names
        }
    try:
        return _run_official_trackeval(
            tracker_names=tracker_names,
            gt_root=gt_root,
            trackers_root=trackers_root,
            split=split,
            seqmap=seqmap,
            output_root=output_root,
            metrics=metrics,
            allow_partial_sequences=allow_partial_sequences,
            version=trackeval_version(),
        )
    except Exception as exc:  # noqa: BLE001
        return {
            tracker_name: _missing_trackeval_result(
                tracker_name,
                output_root,
                f"Official TrackEval failed: {exc}",
            )
            for tracker_name in tracker_names
        }


def _run_official_trackeval(
    tracker_names: list[str],
    gt_root: Path,
    trackers_root: Path,
    split: str,
    seqmap: Path | None,
    output_root: Path,
    metrics: tuple[str, ...],
    allow_partial_sequences: bool,
    version: str | None,
) -> dict[str, TrackEvalTrackerResult]:
    _patch_numpy_aliases()
    from trackeval.datasets import MotChallenge2DBox  # type: ignore[import-not-found]
    from trackeval.eval import Evaluator  # type: ignore[import-not-found]
    from trackeval.metrics import CLEAR, HOTA, Identity  # type: ignore[import-not-found]

    staged_root = output_root / "staged"
    raw_root = output_root / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    seq_names = (
        _common_tracker_seq_names(tracker_names, trackers_root, split)
        if allow_partial_sequences
        else _read_seqmap(seqmap)
    )
    frame_limits = (
        _common_prediction_frame_limits(
            tracker_names,
            trackers_root,
            split,
            seq_names,
        )
        if allow_partial_sequences
        else None
    )
    gt_folder = _trackeval_gt_folder(
        gt_root,
        split,
        seq_names,
        output_root,
        frame_limits=frame_limits,
    )
    results: dict[str, TrackEvalTrackerResult] = {}
    for tracker_name in tracker_names:
        _stage_tracker_files(tracker_name, trackers_root, staged_root, split, seq_names)
        staged_seqmap = _write_staged_seqmap(raw_root / f"{tracker_name}_seqmap.txt", seq_names)
        command = _recorded_command(
            gt_folder,
            staged_root,
            tracker_name,
            split,
            staged_seqmap,
            metrics,
        )
        eval_config = Evaluator.get_default_eval_config()
        eval_config.update(
            {
                "USE_PARALLEL": False,
                "PRINT_RESULTS": False,
                "PRINT_CONFIG": False,
                "TIME_PROGRESS": False,
                "OUTPUT_SUMMARY": True,
                "OUTPUT_DETAILED": True,
                "PLOT_CURVES": False,
                "BREAK_ON_ERROR": True,
                "LOG_ON_ERROR": str(raw_root / "trackeval_errors.log"),
            }
        )
        dataset_config = MotChallenge2DBox.get_default_dataset_config()
        dataset_config.update(
            {
                "GT_FOLDER": str(gt_folder),
                "TRACKERS_FOLDER": str(staged_root),
                "OUTPUT_FOLDER": str(output_root / "official"),
                "TRACKERS_TO_EVAL": [tracker_name],
                "CLASSES_TO_EVAL": ["pedestrian"],
                "BENCHMARK": "SportsMOT-football",
                "SPLIT_TO_EVAL": split,
                "DO_PREPROC": False,
                "PRINT_CONFIG": False,
                "TRACKER_SUB_FOLDER": "data",
                "SEQMAP_FILE": str(staged_seqmap),
                "SKIP_SPLIT_FOL": True,
                "GT_LOC_FORMAT": "{gt_folder}/{seq}/gt/gt.txt",
            }
        )
        evaluator = Evaluator(eval_config)
        dataset = MotChallenge2DBox(dataset_config)
        output_res, output_msg = evaluator.evaluate(
            [dataset],
            _metric_objects(HOTA, CLEAR, Identity),
            show_progressbar=False,
        )
        tracker_res = output_res["MotChallenge2DBox"][tracker_name]
        if tracker_res is None:
            raise RuntimeError(output_msg["MotChallenge2DBox"][tracker_name])
        overall_metrics, per_sequence = _extract_metrics(tracker_res)
        raw_path = raw_root / f"{tracker_name}_trackeval_raw.json"
        raw_path.write_text(
            json.dumps(
                {
                    "created_at": datetime.now(UTC).isoformat(),
                    "tracker": tracker_name,
                    "trackeval_version": version,
                    "command": command,
                    "gt_root": str(gt_folder),
                    "trackers_root": str(trackers_root),
                    "staged_root": str(staged_root),
                    "split": split,
                    "seqmap": str(staged_seqmap),
                    "source_seqmap": str(seqmap) if seqmap else None,
                    "allow_partial_sequences": allow_partial_sequences,
                    "messages": output_msg,
                    "overall_metrics": overall_metrics,
                    "per_sequence": per_sequence,
                },
                indent=2,
                default=_json_default,
            ),
            encoding="utf-8",
        )
        results[tracker_name] = TrackEvalTrackerResult(
            tracker_name=tracker_name,
            available=True,
            metrics=overall_metrics,
            per_sequence=per_sequence,
            command=command,
            version=version,
            reason=None,
            raw_output_path=raw_path,
        )
    return results


def _missing_trackeval_result(
    tracker_name: str,
    output_root: Path,
    reason: str,
) -> TrackEvalTrackerResult:
    raw_path = output_root / f"{tracker_name}_trackeval_missing.json"
    raw_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now(UTC).isoformat(),
                "tracker": tracker_name,
                "reason": reason,
                "metrics": empty_tracking_metrics(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return TrackEvalTrackerResult(
        tracker_name=tracker_name,
        available=False,
        metrics=empty_tracking_metrics(),
        reason=reason,
        raw_output_path=raw_path,
    )


def _patch_numpy_aliases() -> None:
    import numpy as np

    if not hasattr(np, "float"):
        np.float = float  # type: ignore[attr-defined]
    if not hasattr(np, "int"):
        np.int = int  # type: ignore[attr-defined]


def _read_seqmap(seqmap: Path | None) -> list[str]:
    if seqmap is None:
        return []
    names: list[str] = []
    for raw_line in seqmap.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.lower() == "name":
            continue
        names.append(Path(line.replace("\\", "/")).name)
    return names


def _stage_tracker_files(
    tracker_name: str,
    trackers_root: Path,
    staged_root: Path,
    split: str,
    seq_names: list[str],
) -> None:
    data_dir = staged_root / tracker_name / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    source_dir = trackers_root / tracker_name / split
    names = seq_names or [path.stem for path in sorted(source_dir.glob("*.txt"))]
    for seq_name in names:
        source_path = source_dir / f"{seq_name}.txt"
        if not source_path.is_file():
            raise FileNotFoundError(f"Tracker prediction file not found: {source_path}")
        shutil.copy2(source_path, data_dir / f"{seq_name}.txt")


def _trackeval_gt_folder(
    gt_root: Path,
    split: str,
    seq_names: list[str],
    output_root: Path,
    frame_limits: dict[str, int] | None = None,
) -> Path:
    split_dir = gt_root / split
    if split_dir.is_dir() and frame_limits is None:
        return split_dir
    staged = output_root / "staged_gt" / split
    for seq_name in seq_names:
        source_dir = _find_gt_sequence_dir(gt_root, seq_name)
        dest_dir = staged / seq_name
        (dest_dir / "gt").mkdir(parents=True, exist_ok=True)
        frame_limit = frame_limits.get(seq_name) if frame_limits is not None else None
        if frame_limit is None:
            shutil.copy2(source_dir / "gt" / "gt.txt", dest_dir / "gt" / "gt.txt")
            shutil.copy2(source_dir / "seqinfo.ini", dest_dir / "seqinfo.ini")
        else:
            _write_truncated_gt(
                source_dir / "gt" / "gt.txt",
                dest_dir / "gt" / "gt.txt",
                frame_limit,
            )
            _write_partial_seqinfo(
                source_dir / "seqinfo.ini",
                dest_dir / "seqinfo.ini",
                frame_limit,
            )
    return staged


def _common_prediction_frame_limits(
    tracker_names: list[str],
    trackers_root: Path,
    split: str,
    seq_names: list[str],
) -> dict[str, int]:
    limits: dict[str, int] = {}
    for seq_name in seq_names:
        tracker_limits = {
            tracker_name: _prediction_frame_limit(
                trackers_root / tracker_name / split / f"{seq_name}.txt"
            )
            for tracker_name in tracker_names
        }
        unique_limits = set(tracker_limits.values())
        if len(unique_limits) != 1:
            raise ValueError(
                f"Partial benchmark frame limits differ for {seq_name}: "
                f"{tracker_limits}"
            )
        frame_limit = unique_limits.pop()
        if frame_limit < 1:
            raise ValueError(
                f"Partial benchmark frame limit must be positive for {seq_name}."
            )
        limits[seq_name] = frame_limit
    return limits


def _prediction_frame_limit(prediction_path: Path) -> int:
    metadata_path = prediction_path.with_suffix(".metadata.json")
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        frame_count = int(metadata.get("frame_count", 0) or 0)
        if frame_count > 0:
            return frame_count
    if not prediction_path.is_file():
        raise FileNotFoundError(f"Tracker prediction file not found: {prediction_path}")
    max_frame = 0
    for raw_line in prediction_path.read_text(encoding="utf-8").splitlines():
        columns = raw_line.strip().split(",")
        if columns and columns[0].strip():
            max_frame = max(max_frame, int(float(columns[0])))
    return max_frame


def _write_truncated_gt(source: Path, destination: Path, frame_limit: int) -> None:
    rows = []
    for raw_line in source.read_text(encoding="utf-8").splitlines():
        columns = raw_line.strip().split(",")
        if columns and columns[0].strip() and int(float(columns[0])) <= frame_limit:
            rows.append(raw_line)
    destination.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def _write_partial_seqinfo(source: Path, destination: Path, frame_limit: int) -> None:
    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read(source, encoding="utf-8")
    if not parser.has_section("Sequence"):
        raise ValueError(f"seqinfo.ini has no [Sequence] section: {source}")
    parser.set("Sequence", "seqLength", str(frame_limit))
    with destination.open("w", encoding="utf-8", newline="") as handle:
        parser.write(handle, space_around_delimiters=False)


def _find_gt_sequence_dir(gt_root: Path, seq_name: str) -> Path:
    candidates = [
        gt_root / "train" / seq_name,
        gt_root / "val" / seq_name,
        gt_root / "test" / seq_name,
        gt_root / seq_name,
    ]
    for candidate in candidates:
        if (candidate / "gt" / "gt.txt").is_file() and (candidate / "seqinfo.ini").is_file():
            return candidate
    raise FileNotFoundError(f"Ground-truth sequence not found: {seq_name}")


def _common_tracker_seq_names(
    tracker_names: list[str],
    trackers_root: Path,
    split: str,
) -> list[str]:
    sets: list[set[str]] = []
    for tracker_name in tracker_names:
        source_dir = trackers_root / tracker_name / split
        sets.append({path.stem for path in sorted(source_dir.glob("*.txt"))})
    if not sets:
        return []
    common = set.intersection(*sets)
    return sorted(common)


def _write_staged_seqmap(path: Path, seq_names: list[str]) -> Path:
    if not seq_names:
        raise ValueError("No sequence names are available for TrackEval.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("name\n" + "\n".join(seq_names) + "\n", encoding="utf-8")
    return path


def _recorded_command(
    gt_folder: Path,
    staged_root: Path,
    tracker_name: str,
    split: str,
    seqmap: Path | None,
    metrics: tuple[str, ...],
) -> list[str]:
    command = [
        "trackeval-api",
        "--GT_FOLDER",
        str(gt_folder),
        "--TRACKERS_FOLDER",
        str(staged_root),
        "--TRACKERS_TO_EVAL",
        tracker_name,
        "--SPLIT_TO_EVAL",
        split,
        "--METRICS",
        ",".join(metrics),
        "--DO_PREPROC",
        "False",
    ]
    if seqmap is not None:
        command.extend(["--SEQMAP_FILE", str(seqmap)])
    return command


def _extract_metrics(
    tracker_res: dict[str, Any],
) -> tuple[dict[str, float | int | None], dict[str, dict[str, float | int | None]]]:
    from trackeval.metrics import CLEAR, HOTA, Identity  # type: ignore[import-not-found]

    hota, clear, identity = _metric_objects(HOTA, CLEAR, Identity)
    metric_objects = {"HOTA": hota, "CLEAR": clear, "Identity": identity}
    key_map = {"CLR_FP": "FP", "CLR_FN": "FN"}

    def summarize(seq_key: str) -> dict[str, float | int | None]:
        row: dict[str, float | int | None] = empty_tracking_metrics()
        class_results = tracker_res[seq_key]["pedestrian"]
        for metric_name, metric in metric_objects.items():
            summary = metric.summary_results({"COMBINED_SEQ": class_results[metric_name]})
            for key, value in summary.items():
                target_key = key_map.get(key, key)
                if target_key in row:
                    row[target_key] = _scalar(value)
        return row

    overall = summarize("COMBINED_SEQ")
    per_sequence = {
        seq_name: summarize(seq_name) for seq_name in tracker_res if seq_name != "COMBINED_SEQ"
    }
    return overall, per_sequence


def _scalar(value: Any) -> float | int | None:
    if value is None:
        return None
    try:
        if hasattr(value, "item"):
            value = value.item()
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number.is_integer():
        return int(number)
    return number


def _metric_objects(hota_cls: Any, clear_cls: Any, identity_cls: Any) -> list[Any]:
    return [
        hota_cls({"PRINT_CONFIG": False}),
        clear_cls({"PRINT_CONFIG": False}),
        identity_cls({"PRINT_CONFIG": False}),
    ]


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return str(value)
