"""Comparison report for measured realtime tracking runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import fmean
from typing import Any


class RealtimeReportError(RuntimeError):
    """Raised when measured realtime artifacts cannot be compared."""


def build_realtime_report(
    runs: list[tuple[str, str | Path]],
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    if not runs:
        raise RealtimeReportError("At least one realtime run is required.")
    rows = [_load_run(name, Path(path).resolve()) for name, path in runs]
    root = Path(output_dir).resolve()
    paths = {
        "json": root / "realtime_benchmark.json",
        "csv": root / "realtime_benchmark.csv",
        "markdown": root / "realtime_benchmark.md",
    }
    if not overwrite:
        existing = [path for path in paths.values() if path.exists()]
        if existing:
            raise RealtimeReportError(f"Realtime report exists: {existing[0]}")
    root.mkdir(parents=True, exist_ok=True)
    best = max(rows, key=lambda row: float(row["source_progress_fps"] or 0.0))
    payload = {
        "schema_version": 1,
        "run_count": len(rows),
        "hardware": rows[0]["hardware"],
        "best_source_progress_run": best["name"],
        "mean_processing_fps": fmean(float(row["processing_fps"] or 0.0) for row in rows),
        "runs": rows,
        "notes": [
            "processing_fps measures processed frames only",
            "source_progress_fps includes intentionally dropped late frames",
            "frame dropping bounds live latency but is not appropriate for offline evaluation",
        ],
    }
    paths["json"].write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(paths["csv"], rows)
    paths["markdown"].write_text(_markdown(payload), encoding="utf-8")
    figures = _figures(rows, root / "figures")
    return {
        "status": "ok",
        "summary": payload,
        "paths": {name: str(path) for name, path in paths.items()},
        "figures": [str(path) for path in figures],
    }


def _load_run(name: str, path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RealtimeReportError(f"Realtime metrics do not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("mode") != "realtime":
        raise RealtimeReportError(f"Not a realtime metrics artifact: {path}")
    timing = value.get("timing", {})
    resources = value.get("resources", {})
    cuda = value.get("cuda_memory", {})
    hardware = value.get("hardware", {})
    return {
        "name": str(name),
        "path": str(path),
        "frames_processed": int(value.get("frames", 0)),
        "source_frames_consumed": int(value.get("source_frames_consumed", value.get("frames", 0))),
        "dropped_late_frames": int(value.get("dropped_late_frames", 0)),
        "drop_rate": float(value.get("late_frame_drop_rate", 0.0)),
        "processing_fps": timing.get("processing_fps"),
        "steady_state_fps": timing.get("steady_state_processing_fps"),
        "source_progress_fps": timing.get("source_progress_fps", timing.get("end_to_end_fps")),
        "source_fps": timing.get("source_fps"),
        "p50_latency_ms": timing.get("frame_latency_ms_p50"),
        "p95_latency_ms": timing.get("frame_latency_ms_p95"),
        "p99_latency_ms": timing.get("frame_latency_ms_p99"),
        "startup_seconds": timing.get("startup_seconds"),
        "detector_fps": timing.get("detector_fps"),
        "tracker_fps": timing.get("tracker_fps"),
        "peak_ram_gb": _gb(resources.get("peak_process_rss_bytes")),
        "peak_vram_gb": _gb(cuda.get("peak_allocated_bytes")),
        "hardware": {
            "gpu": hardware.get("gpu_name"),
            "gpu_memory_gb": _gb(hardware.get("gpu_memory_total_bytes")),
            "cpu": hardware.get("processor"),
            "logical_cpu_count": hardware.get("logical_cpu_count"),
        },
    }


def _gb(value: Any) -> float | None:
    return round(float(value) / (1024**3), 4) if value is not None else None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [key for key in rows[0] if key not in {"path", "hardware"}]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)


def _markdown(payload: dict[str, Any]) -> str:
    hardware = payload["hardware"]
    lines = [
        "# Realtime Benchmark",
        "",
        f"Hardware: {hardware.get('gpu')} "
        f"({float(hardware.get('gpu_memory_gb') or 0):.1f} GB VRAM), "
        f"{hardware.get('logical_cpu_count')} logical CPU threads.",
        "",
        "| Run | Process FPS | Source progress FPS | p95 | Drop | Startup | RAM | VRAM |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["runs"]:
        lines.append(
            f"| {row['name']} | {float(row['processing_fps'] or 0):.2f} | "
            f"{float(row['source_progress_fps'] or 0):.2f} | "
            f"{float(row['p95_latency_ms'] or 0):.1f} ms | "
            f"{100 * float(row['drop_rate'] or 0):.1f}% | "
            f"{float(row['startup_seconds'] or 0):.1f}s | "
            f"{float(row['peak_ram_gb'] or 0):.2f} GB | "
            f"{float(row['peak_vram_gb'] or 0):.2f} GB |"
        )
    lines.extend(
        [
            "",
            "The bounded-latency run drops late input frames instead of accumulating camera lag.",
            "Use the no-drop profile for offline accuracy evaluation.",
            "",
            "![FPS](figures/realtime_fps.png)",
            "",
            "![Latency](figures/realtime_latency_drop.png)",
            "",
        ]
    )
    return "\n".join(lines)


def _figures(rows: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = [row["name"] for row in rows]
    x = list(range(len(rows)))
    figure, axis = plt.subplots(figsize=(9, 4.8))
    axis.bar(
        [index - 0.2 for index in x],
        [row["processing_fps"] or 0 for row in rows],
        0.4,
        label="Processed",
    )
    axis.bar(
        [index + 0.2 for index in x],
        [row["source_progress_fps"] or 0 for row in rows],
        0.4,
        label="Source progress",
    )
    axis.axhline(30, color="black", linestyle="--", linewidth=1, label="30 FPS source")
    axis.set_xticks(x, labels, rotation=15, ha="right")
    axis.set_ylabel("FPS")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    fps_path = output_dir / "realtime_fps.png"
    figure.savefig(fps_path, dpi=180)
    plt.close(figure)

    figure, latency_axis = plt.subplots(figsize=(9, 4.8))
    latency_axis.bar(x, [row["p95_latency_ms"] or 0 for row in rows], color="#2878B5")
    latency_axis.set_xticks(x, labels, rotation=15, ha="right")
    latency_axis.set_ylabel("p95 latency (ms)")
    latency_axis.grid(axis="y", alpha=0.25)
    drop_axis = latency_axis.twinx()
    drop_axis.plot(
        x,
        [100 * float(row["drop_rate"] or 0) for row in rows],
        color="#D95319",
        marker="o",
    )
    drop_axis.set_ylabel("Late-frame drop (%)")
    figure.tight_layout()
    latency_path = output_dir / "realtime_latency_drop.png"
    figure.savefig(latency_path, dpi=180)
    plt.close(figure)
    return [fps_path, latency_path]


__all__ = ["RealtimeReportError", "build_realtime_report"]
