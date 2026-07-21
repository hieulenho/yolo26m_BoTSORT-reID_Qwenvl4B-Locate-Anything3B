"""Compare two saved tracker profiles on the same detector/video evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from football_tracking.benchmarking.multidomain_report import (
    compute_mot_stability_proxy,
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _row(name: str, metadata_path: Path, mot_path: Path, fps: float) -> dict[str, Any]:
    metadata = _read_json(metadata_path)
    timing = dict(metadata.get("timing", {}))
    diagnostics = dict(metadata.get("tracker_diagnostics") or {})
    proxy = compute_mot_stability_proxy(mot_path, fps)
    return {
        "profile": name,
        "tracker": metadata.get("tracker"),
        "frame_count": metadata.get("frame_count"),
        "detection_count": metadata.get("detection_count"),
        "unique_track_count": metadata.get("unique_track_count"),
        "steady_state_fps": timing.get("steady_state_fps"),
        "raw_class_switches": diagnostics.get("raw_class_switches", 0),
        "stable_class_switches": diagnostics.get("stable_class_switches", 0),
        "short_track_ratio": proxy.get("short_track_ratio"),
        "median_track_length_frames": proxy.get("median_track_length_frames"),
        "mean_track_continuity": proxy.get("mean_track_continuity"),
        "within_id_gap_events": proxy.get("within_id_gap_events"),
        "scope": proxy.get("scope"),
        "metadata": str(metadata_path.resolve()),
        "mot": str(mot_path.resolve()),
    }


def _percent_reduction(baseline: float, candidate: float) -> float | None:
    if baseline <= 0:
        return None
    return round(100.0 * (baseline - candidate) / baseline, 3)


def _write_chart(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    labels = [str(row["profile"]) for row in rows]
    figure, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    axes[0].bar(labels, [float(row["steady_state_fps"] or 0) for row in rows])
    axes[0].set_title("End-to-end FPS")
    axes[1].bar(labels, [int(row["unique_track_count"] or 0) for row in rows])
    axes[1].set_title("Unique predicted IDs")
    axes[2].bar(labels, [100 * float(row["short_track_ratio"] or 0) for row in rows])
    axes[2].set_title("Tracks shorter than 1 s (%)")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Traffic tracker profile ablation (same detections)")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-name", default="realtime_ocsort")
    parser.add_argument("--baseline-metadata", type=Path, required=True)
    parser.add_argument("--baseline-mot", type=Path, required=True)
    parser.add_argument("--candidate-name", default="realtime_stable_tracktrack")
    parser.add_argument("--candidate-metadata", type=Path, required=True)
    parser.add_argument("--candidate-mot", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_json = args.output_dir / "tracking_profile_ablation.json"
    if output_json.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output_json}")
    rows = [
        _row(args.baseline_name, args.baseline_metadata, args.baseline_mot, args.fps),
        _row(args.candidate_name, args.candidate_metadata, args.candidate_mot, args.fps),
    ]
    baseline, candidate = rows
    comparison = {
        "unique_id_reduction_percent": _percent_reduction(
            float(baseline["unique_track_count"]), float(candidate["unique_track_count"])
        ),
        "short_track_reduction_percent": _percent_reduction(
            float(baseline["short_track_ratio"]), float(candidate["short_track_ratio"])
        ),
        "stable_class_switch_reduction_percent": _percent_reduction(
            float(baseline["stable_class_switches"]),
            float(candidate["stable_class_switches"]),
        ),
        "fps_change_percent": round(
            100.0
            * (float(candidate["steady_state_fps"]) - float(baseline["steady_state_fps"]))
            / float(baseline["steady_state_fps"]),
            3,
        ),
        "identity_metric_note": (
            "Raw-video continuity proxies are not IDSW. Use the SportsMOT GT table for "
            "official identity comparison."
        ),
    }
    payload = {"schema_version": 1, "rows": rows, "comparison": comparison}
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_path = args.output_dir / "tracking_profile_ablation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    chart_path = args.output_dir / "tracking_profile_ablation.png"
    _write_chart(chart_path, rows)
    markdown = args.output_dir / "tracking_profile_ablation.md"
    markdown.write_text(
        "\n".join(
            [
                "# Traffic tracking profile ablation",
                "",
                "Both rows use the same 35.0-second video, dynamic vocabulary, YOLO26n at "
                "640 px, and detector outputs.",
                "",
                "| Profile | FPS | IDs | Short tracks | Median length | Stable class changes |",
                "|---|---:|---:|---:|---:|---:|",
                *[
                    f"| {row['profile']} | {float(row['steady_state_fps']):.2f} | "
                    f"{row['unique_track_count']} | {100 * float(row['short_track_ratio']):.1f}% | "
                    f"{float(row['median_track_length_frames']):.1f} | "
                    f"{row['stable_class_switches']} |"
                    for row in rows
                ],
                "",
                comparison["identity_metric_note"],
                "",
                "![Profile ablation](tracking_profile_ablation.png)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"status": "ok", "comparison": comparison}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
