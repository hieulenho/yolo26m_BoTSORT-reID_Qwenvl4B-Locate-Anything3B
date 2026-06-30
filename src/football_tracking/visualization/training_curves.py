"""Training curve rendering from Ultralytics results.csv."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)

COLUMN_ALIASES = {
    "train_box_loss.png": ["train/box_loss", "train_box_loss"],
    "train_cls_loss.png": ["train/cls_loss", "train_cls_loss"],
    "train_dfl_loss.png": ["train/dfl_loss", "train_dfl_loss"],
    "validation_box_loss.png": ["val/box_loss", "val_box_loss"],
    "validation_cls_loss.png": ["val/cls_loss", "val_cls_loss"],
    "precision_curve_over_epochs.png": ["metrics/precision(B)", "precision"],
    "recall_curve_over_epochs.png": ["metrics/recall(B)", "recall"],
    "map50_over_epochs.png": ["metrics/mAP50(B)", "map50"],
    "map50_95_over_epochs.png": ["metrics/mAP50-95(B)", "map50_95"],
    "learning_rate_over_epochs.png": ["lr/pg0", "lr0", "learning_rate"],
}


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def render_training_curves(results_csv: Path, output_dir: Path) -> list[Path]:
    if not results_csv.is_file():
        LOGGER.warning("results.csv is missing: %s", results_csv)
        return []
    rows = _load_rows(results_csv)
    if not rows:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    written: list[Path] = []
    epochs = list(range(1, len(rows) + 1))
    for filename, aliases in COLUMN_ALIASES.items():
        column = next((alias for alias in aliases if alias in rows[0]), None)
        if column is None:
            LOGGER.warning("No column found for %s", filename)
            continue
        values = []
        for row in rows:
            try:
                values.append(float(row[column]))
            except (TypeError, ValueError):
                values.append(float("nan"))
        path = output_dir / filename
        plt.figure()
        plt.plot(epochs, values)
        plt.title(filename.removesuffix(".png").replace("_", " "))
        plt.xlabel("epoch")
        plt.ylabel(column)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        written.append(path)
    return written
