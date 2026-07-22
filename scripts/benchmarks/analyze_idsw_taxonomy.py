"""Run ID switch taxonomy diagnostics for one or more tracker outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from football_tracking.evaluation.idsw_taxonomy import (
    analyze_many_trackers,
    default_tracker_roots,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Break total ID switches into diagnostic failure types.",
    )
    parser.add_argument(
        "--mot-root",
        type=Path,
        default=Path("data/mot/sportsmot_football"),
        help="SportsMOT MOT root containing train/val/test sequence folders.",
    )
    parser.add_argument(
        "--seqmap",
        type=Path,
        default=Path("data/mot/sportsmot_football/seqmaps/all.txt"),
        help="Sequence map to evaluate.",
    )
    parser.add_argument(
        "--tracker",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help=(
            "Tracker prediction directory containing one MOT txt per sequence. "
            "Repeat this argument for multiple trackers. Defaults to the known "
            "project tracker outputs when omitted."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/reports/focused_pipeline/idsw_taxonomy"),
    )
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--reid-gap", type=int, default=10)
    parser.add_argument("--swap-window", type=int, default=5)
    parser.add_argument("--crowd-scale", type=float, default=1.5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    trackers = _parse_trackers(args.tracker)
    if not trackers:
        trackers = default_tracker_roots()
    if not trackers:
        raise SystemExit("No tracker directories found. Pass --tracker NAME=PATH.")

    result = analyze_many_trackers(
        trackers=trackers,
        mot_root=args.mot_root,
        seqmap=args.seqmap,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        iou_threshold=args.iou_threshold,
        reid_gap=args.reid_gap,
        swap_window=args.swap_window,
        crowd_scale=args.crowd_scale,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


def _parse_trackers(values: list[str]) -> dict[str, Path]:
    trackers: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"Invalid --tracker value, expected NAME=PATH: {value}")
        name, raw_path = value.split("=", 1)
        name = name.strip()
        path = Path(raw_path.strip())
        if not name:
            raise SystemExit(f"Tracker name is empty in: {value}")
        if not path.is_dir():
            raise SystemExit(f"Tracker directory does not exist: {path}")
        trackers[name] = path
    return trackers


if __name__ == "__main__":
    main()
