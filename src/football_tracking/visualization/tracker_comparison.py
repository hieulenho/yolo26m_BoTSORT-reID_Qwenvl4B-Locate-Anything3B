"""Matplotlib plots for tracker comparison metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

METRIC_FIGURES = {
    "HOTA": "hota_comparison.png",
    "DetA": "deta_comparison.png",
    "AssA": "assa_comparison.png",
    "IDF1": "idf1_comparison.png",
    "MOTA": "mota_comparison.png",
    "IDSW": "idsw_comparison.png",
    "Frag": "frag_comparison.png",
    "tracker_fps": "tracker_fps_comparison.png",
}


def _numeric(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def write_tracker_comparison_figures(
    overall_rows: list[dict[str, Any]],
    figures_dir: Path,
) -> list[str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for metric_name, filename in METRIC_FIGURES.items():
        path = _write_bar(overall_rows, metric_name, figures_dir / filename)
        if path is not None:
            written.append(str(path))
    fp_fn = _write_grouped(overall_rows, ("FP", "FN"), figures_dir / "fp_fn_comparison.png")
    if fp_fn is not None:
        written.append(str(fp_fn))
    for metric_name, filename in {
        "HOTA": "speed_vs_hota.png",
        "IDF1": "speed_vs_idf1.png",
    }.items():
        path = _write_scatter(overall_rows, "tracker_fps", metric_name, figures_dir / filename)
        if path is not None:
            written.append(str(path))
    return written


def _write_bar(rows: list[dict[str, Any]], metric_name: str, path: Path) -> Path | None:
    values = [(row.get("tracker"), _numeric(row.get(metric_name))) for row in rows]
    values = [(name, value) for name, value in values if name and value is not None]
    if not values:
        return None
    import matplotlib  # type: ignore[import-not-found]

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt  # type: ignore[import-not-found]

    names = [str(name) for name, _value in values]
    numbers = [float(value) for _name, value in values]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, numbers, color=["#2f6f73", "#8c5a2b"][: len(names)])
    ax.set_title(metric_name)
    ax.set_ylabel(metric_name)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _write_grouped(
    rows: list[dict[str, Any]],
    metric_names: tuple[str, str],
    path: Path,
) -> Path | None:
    trackers = [str(row.get("tracker")) for row in rows]
    first = [_numeric(row.get(metric_names[0])) for row in rows]
    second = [_numeric(row.get(metric_names[1])) for row in rows]
    if not trackers or any(value is None for value in [*first, *second]):
        return None
    import matplotlib  # type: ignore[import-not-found]

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt  # type: ignore[import-not-found]

    positions = list(range(len(trackers)))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([pos - width / 2 for pos in positions], first, width, label=metric_names[0])
    ax.bar([pos + width / 2 for pos in positions], second, width, label=metric_names[1])
    ax.set_xticks(positions, trackers)
    ax.set_ylabel("count")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _write_scatter(
    rows: list[dict[str, Any]],
    x_metric: str,
    y_metric: str,
    path: Path,
) -> Path | None:
    values = [
        (str(row.get("tracker")), _numeric(row.get(x_metric)), _numeric(row.get(y_metric)))
        for row in rows
    ]
    values = [
        (name, x_value, y_value)
        for name, x_value, y_value in values
        if x_value is not None and y_value is not None
    ]
    if not values:
        return None
    import matplotlib  # type: ignore[import-not-found]

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt  # type: ignore[import-not-found]

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, x_value, y_value in values:
        ax.scatter([x_value], [y_value], label=name, s=60)
        ax.annotate(name, (x_value, y_value), textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel(x_metric)
    ax.set_ylabel(y_metric)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
