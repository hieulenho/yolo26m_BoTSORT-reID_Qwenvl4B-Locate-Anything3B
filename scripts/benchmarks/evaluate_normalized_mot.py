"""Evaluate normalized MOT predictions with the installed official TrackEval API."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from football_tracking.evaluation.multi_tracker_trackeval import (
    evaluate_trackers_with_trackeval,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument(
        "--prediction",
        type=Path,
        action="append",
        required=True,
        help="Prediction MOT file. Repeat once per --sequence in matching order.",
    )
    parser.add_argument(
        "--sequence",
        action="append",
        required=True,
        help="Sequence name. Repeat once per --prediction in matching order.",
    )
    parser.add_argument("--tracker-name", default="adaptive")
    parser.add_argument("--split", default="val")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    try:
        pairs = _prediction_pairs(args.sequence, args.prediction)
    except ValueError as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 2
    missing = [prediction for _, prediction in pairs if not prediction.is_file()]
    if missing:
        sys.stderr.write(f"Error: Prediction does not exist: {missing[0]}\n")
        return 2
    summary_path = args.output_dir / "trackeval_summary.json"
    if summary_path.exists() and not args.overwrite:
        sys.stderr.write(f"Error: Output exists and overwrite=false: {summary_path}\n")
        return 2
    tracker_dir = args.output_dir / "inputs" / args.tracker_name / args.split
    tracker_dir.mkdir(parents=True, exist_ok=True)
    for sequence, prediction in pairs:
        shutil.copy2(prediction, tracker_dir / f"{sequence}.txt")
    seqmap = args.output_dir / "seqmap.txt"
    seqmap.write_text(
        "name\n" + "\n".join(sequence for sequence, _ in pairs) + "\n",
        encoding="utf-8",
    )
    results = evaluate_trackers_with_trackeval(
        tracker_names=[args.tracker_name],
        gt_root=args.gt_root.resolve(),
        trackers_root=(args.output_dir / "inputs").resolve(),
        split=args.split,
        seqmap=seqmap.resolve(),
        output_root=(args.output_dir / "trackeval").resolve(),
        metrics=("HOTA", "CLEAR", "Identity"),
    )
    payload = results[args.tracker_name].to_dict()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["available"] else 1


def _prediction_pairs(
    sequences: list[str], predictions: list[Path]
) -> list[tuple[str, Path]]:
    if len(sequences) != len(predictions):
        raise ValueError(
            "The number of --sequence and --prediction arguments must match "
            f"({len(sequences)} != {len(predictions)})."
        )
    normalized = [sequence.strip() for sequence in sequences]
    if any(not sequence for sequence in normalized):
        raise ValueError("Sequence names must not be empty.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Sequence names must be unique.")
    return list(zip(normalized, predictions, strict=True))


if __name__ == "__main__":
    raise SystemExit(main())
