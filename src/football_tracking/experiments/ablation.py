"""One-factor-at-a-time tracker ablation planning."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from football_tracking.paths import get_project_root, resolve_project_path
from football_tracking.reporting.ablation_report import write_ablation_report


class AblationError(RuntimeError):
    """Raised when ablation config is invalid."""


@dataclass(frozen=True)
class AblationExperiment:
    experiment_id: str
    tracker: str
    parameter: str
    value: Any
    status: str

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _resolve(path: str | Path, project_root: Path) -> Path:
    candidate = Path(path)
    return (
        candidate.resolve()
        if candidate.is_absolute()
        else resolve_project_path(candidate, project_root)
    )


def _experiment_id(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def generate_ablation_plan(config_path: str | Path) -> list[AblationExperiment]:
    project_root = get_project_root()
    path = _resolve(config_path, project_root)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise AblationError("Ablation config root must be a mapping.")
    ablation = raw.get("ablation", {})
    strategy = raw.get("strategy", {})
    max_experiments = int(strategy.get("max_experiments", 20))
    include_baseline = bool(strategy.get("include_baseline", True))
    experiments: list[AblationExperiment] = []
    if include_baseline:
        payload = {"tracker": "shared", "parameter": "baseline", "value": "baseline"}
        experiments.append(
            AblationExperiment(_experiment_id(payload), "shared", "baseline", "baseline", "pending")
        )
    for tracker_name in ("sort", "deepsort", "shared"):
        tracker_values = ablation.get(tracker_name, {}) or {}
        for parameter, values in tracker_values.items():
            for value in values:
                payload = {"tracker": tracker_name, "parameter": parameter, "value": value}
                experiments.append(
                    AblationExperiment(
                        _experiment_id(payload),
                        tracker_name,
                        str(parameter),
                        value,
                        "pending",
                    )
                )
                if len(experiments) >= max_experiments:
                    return experiments
    return experiments


def run_tracker_ablation(
    config_path: str | Path,
    dry_run: bool = False,
    max_experiments: int | None = None,
) -> dict[str, Any]:
    experiments = generate_ablation_plan(config_path)
    if max_experiments is not None:
        experiments = experiments[:max_experiments]
    rows = [experiment.to_dict() for experiment in experiments]
    if dry_run:
        return {
            "dry_run": True,
            "experiment_count": len(rows),
            "experiments": rows,
            "note": "Ablation plan generated; no files were written.",
        }
    project_root = get_project_root()
    manifest_path = project_root / "outputs/experiments/ablation/manifest.json"
    metrics_json = project_root / "outputs/metrics/experiments/ablation_results.json"
    metrics_csv = project_root / "outputs/metrics/experiments/ablation_results.csv"
    report_path = project_root / "outputs/metrics/experiments/tracker_ablation_report.md"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_json.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"experiments": rows}, indent=2), encoding="utf-8")
    metrics_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with metrics_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["experiment_id", "tracker", "parameter", "value", "status"],
        )
        writer.writeheader()
        writer.writerows(rows)
    write_ablation_report(rows, report_path)
    return {
        "dry_run": dry_run,
        "experiment_count": len(rows),
        "experiments": rows,
        "paths": {
            "manifest": str(manifest_path),
            "json": str(metrics_json),
            "csv": str(metrics_csv),
            "report": str(report_path),
        },
        "note": "Ablation plan generated. Tracker execution is intentionally explicit.",
    }
