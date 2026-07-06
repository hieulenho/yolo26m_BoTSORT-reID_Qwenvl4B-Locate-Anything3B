"""Reproducible chart generation from saved language metric artifacts."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def write_chart_data(
    *,
    ablation_rows: list[dict[str, Any]],
    output_dir: str | Path,
    overwrite: bool = False,
) -> dict[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "ablation_target_f1_chart_data.csv"
    if csv_path.exists() and not overwrite:
        return {"ablation_target_f1_csv": str(csv_path)}
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["variant_id", "micro_target_f1"])
        writer.writeheader()
        for row in ablation_rows:
            writer.writerow(
                {
                    "variant_id": row.get("variant_id"),
                    "micro_target_f1": row.get("micro_target_f1"),
                }
            )
    png = _try_write_bar(ablation_rows, root / "ablation_target_f1.png")
    paths = {"ablation_target_f1_csv": str(csv_path)}
    if png is not None:
        paths["ablation_target_f1_png"] = str(png)
    return paths


def _try_write_bar(rows: list[dict[str, Any]], path: Path) -> Path | None:
    values = [
        (str(row.get("variant_id")), row.get("micro_target_f1"))
        for row in rows
        if row.get("micro_target_f1") is not None
    ]
    if not values:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return None
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar([name for name, _value in values], [float(value) for _name, value in values])
    axis.set_xlabel("Variant")
    axis.set_ylabel("Micro Target F1")
    axis.set_ylim(0.0, 1.0)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path
