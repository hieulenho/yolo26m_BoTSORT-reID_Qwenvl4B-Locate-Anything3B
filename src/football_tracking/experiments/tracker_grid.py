"""Generate tracker grid-search experiment configs without running heavy jobs."""

from __future__ import annotations

import copy
import csv
import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from football_tracking.paths import get_project_root, resolve_project_path


class TrackerGridError(RuntimeError):
    """Raised when tracker grid-search planning fails."""


@dataclass(frozen=True)
class TrackerGridVariant:
    experiment_id: str
    name: str
    parameter_values: dict[str, Any]
    tracker_config: Path
    compare_config: Path
    command: str

    def to_dict(self, project_root: Path) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "parameter_values": self.parameter_values,
            "tracker_config": _relative(self.tracker_config, project_root),
            "compare_config": _relative(self.compare_config, project_root),
            "command": self.command,
        }


def _mapping(value: Any, section: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TrackerGridError(f"{section} must be a mapping.")
    return value


def _resolve(value: Any, project_root: Path, section: str) -> Path:
    if not isinstance(value, str | Path) or not str(value).strip():
        raise TrackerGridError(f"{section} must be a non-empty path string.")
    path = Path(value)
    return path.resolve() if path.is_absolute() else resolve_project_path(path, project_root)


def _load_yaml(path: Path, section: str) -> dict[str, Any]:
    if not path.is_file():
        raise TrackerGridError(f"{section} does not exist: {path}")
    return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), section)


def _relative(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _experiment_id(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def _set_nested(payload: dict[str, Any], dotted_path: str, value: Any) -> None:
    parts = dotted_path.split(".")
    if len(parts) < 2:
        raise TrackerGridError(f"Grid parameter must use dot path syntax: {dotted_path}")
    current: dict[str, Any] = payload
    for part in parts[:-1]:
        child = current.setdefault(part, {})
        if not isinstance(child, dict):
            raise TrackerGridError(f"Cannot set nested parameter through scalar: {dotted_path}")
        current = child
    current[parts[-1]] = value


def _parameter_combinations(parameters: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not parameters:
        return []
    keys = list(parameters)
    values = []
    for key in keys:
        parameter_values = parameters[key]
        if not isinstance(parameter_values, list) or not parameter_values:
            raise TrackerGridError(f"parameters.{key} must be a non-empty list.")
        values.append(parameter_values)
    return [dict(zip(keys, item, strict=True)) for item in itertools.product(*values)]


def plan_tracker_grid(
    config_path: str | Path,
    *,
    dry_run: bool = False,
    max_experiments: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    _pending_payloads.clear()
    project_root = get_project_root()
    path = _resolve(config_path, project_root, "config")
    raw = _load_yaml(path, "tracker grid config")
    grid = _mapping(raw.get("grid"), "grid")
    strategy = _mapping(raw.get("strategy", {}), "strategy")
    runtime = _mapping(raw.get("runtime", {}), "runtime")
    raw_parameters = _mapping(raw.get("parameters", {}), "parameters")
    parameters = {str(key): values for key, values in raw_parameters.items()}
    name = str(grid.get("name", "tracker_grid"))
    tracker_name = str(grid.get("tracker_name", "botsort_reid")).lower()
    base_tracker_config = _resolve(
        grid.get("base_tracker_config"),
        project_root,
        "grid.base_tracker_config",
    )
    base_compare_config = _resolve(
        grid.get("base_compare_config"),
        project_root,
        "grid.base_compare_config",
    )
    output_root = _resolve(
        grid.get("output_root", f"outputs/experiments/tracker_grid/{name}"),
        project_root,
        "grid.output_root",
    )
    base_tracker_payload = _load_yaml(base_tracker_config, "grid.base_tracker_config")
    base_compare_payload = _load_yaml(base_compare_config, "grid.base_compare_config")
    combinations = _parameter_combinations(parameters)
    include_baseline = bool(strategy.get("include_baseline", True))
    variants_payloads = []
    if include_baseline:
        variants_payloads.append(("baseline", {}))
    variants_payloads.extend(
        (f"variant_{index:03d}", values) for index, values in enumerate(combinations, start=1)
    )
    max_count = max_experiments or grid.get("max_experiments")
    if max_count is not None:
        variants_payloads = variants_payloads[: int(max_count)]
    variants = [
        _variant(
            name,
            tracker_name,
            base_tracker_payload,
            base_compare_payload,
            output_root,
            variant_name,
            parameter_values,
            project_root,
            str(
                runtime.get(
                    "command_prefix",
                    r".\.venv\Scripts\python.exe -m football_tracking.cli",
                )
            ),
        )
        for variant_name, parameter_values in variants_payloads
    ]
    if not dry_run:
        _write_grid_outputs(output_root, variants, project_root, overwrite)
    return {
        "dry_run": dry_run,
        "grid": name,
        "tracker": tracker_name,
        "variant_count": len(variants),
        "output_root": _relative(output_root, project_root),
        "variants": [variant.to_dict(project_root) for variant in variants],
        "paths": {
            "manifest": _relative(output_root / "manifest.json", project_root),
            "csv": _relative(output_root / "manifest.csv", project_root),
            "run_all": _relative(output_root / "run_all.ps1", project_root),
        },
    }


def _variant(
    grid_name: str,
    tracker_name: str,
    base_tracker_payload: dict[str, Any],
    base_compare_payload: dict[str, Any],
    output_root: Path,
    variant_name: str,
    parameter_values: dict[str, Any],
    project_root: Path,
    command_prefix: str,
) -> TrackerGridVariant:
    payload = {"grid": grid_name, "tracker": tracker_name, "values": parameter_values}
    experiment_id = _experiment_id(payload)
    tracker_payload = copy.deepcopy(base_tracker_payload)
    for parameter, value in parameter_values.items():
        _set_nested(tracker_payload, parameter, value)
    tracker_config = output_root / "tracker_configs" / f"{variant_name}_{experiment_id}.yaml"
    compare_config = output_root / "compare_configs" / f"{variant_name}_{experiment_id}.yaml"
    compare_payload = _compare_payload(
        base_compare_payload,
        grid_name,
        tracker_name,
        tracker_config,
        output_root,
        variant_name,
        experiment_id,
        project_root,
    )
    command = (
        f"{command_prefix} compare-trackers --config "
        f"{_relative(compare_config, project_root)} --overwrite --debug"
    )
    _pending_payloads[tracker_config] = tracker_payload
    _pending_payloads[compare_config] = compare_payload
    return TrackerGridVariant(
        experiment_id=experiment_id,
        name=variant_name,
        parameter_values=parameter_values,
        tracker_config=tracker_config,
        compare_config=compare_config,
        command=command,
    )


_pending_payloads: dict[Path, dict[str, Any]] = {}


def _compare_payload(
    base: dict[str, Any],
    grid_name: str,
    tracker_name: str,
    tracker_config: Path,
    output_root: Path,
    variant_name: str,
    experiment_id: str,
    project_root: Path,
) -> dict[str, Any]:
    payload = copy.deepcopy(base)
    payload.setdefault("experiment", {})
    payload["experiment"]["name"] = f"{grid_name}_{variant_name}_{experiment_id}"
    payload["trackers"] = [
        {
            "name": tracker_name,
            "config": _relative(tracker_config, project_root),
        }
    ]
    output = payload.setdefault("output", {})
    namespace = f"{variant_name}_{experiment_id}"
    output["root"] = _relative(output_root / "runs" / namespace, project_root)
    output["tracks_root"] = _relative(output_root / "tracks" / namespace, project_root)
    output["metrics_root"] = _relative(output_root / "metrics" / namespace, project_root)
    output["figures_root"] = _relative(output_root / "figures" / namespace, project_root)
    output["videos_root"] = _relative(output_root / "videos" / namespace, project_root)
    return payload


def _write_grid_outputs(
    output_root: Path,
    variants: list[TrackerGridVariant],
    project_root: Path,
    overwrite: bool,
) -> None:
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise TrackerGridError(
            f"Grid output directory is not empty and overwrite=false: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    for path in sorted(_pending_payloads):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(_pending_payloads[path], sort_keys=False, allow_unicode=False),
            encoding="utf-8",
        )
    rows = [variant.to_dict(project_root) for variant in variants]
    (output_root / "manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (output_root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "experiment_id",
            "name",
            "parameter_values",
            "tracker_config",
            "compare_config",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{key: row[key] for key in fieldnames if key != "parameter_values"},
                    "parameter_values": json.dumps(row["parameter_values"], sort_keys=True),
                }
            )
    commands = ["$env:FOOTBALL_TRACKING_PROGRESS=\"1\""]
    commands.extend(variant.command for variant in variants)
    (output_root / "run_all.ps1").write_text("\n".join(commands) + "\n", encoding="utf-8")
