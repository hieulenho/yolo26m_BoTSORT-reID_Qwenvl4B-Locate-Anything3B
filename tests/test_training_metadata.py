import json
from pathlib import Path

from football_tracking.detection.experiment import ExperimentManifest


def test_experiment_manifest_serializes_runtime_metadata(tmp_path: Path) -> None:
    manifest = ExperimentManifest(
        experiment_name="demo",
        run_dir=tmp_path / "run",
        project_root=Path.cwd(),
        payload={"dataset_yaml": "dataset.yaml"},
    )

    path = manifest.finish("completed")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["experiment_name"] == "demo"
    assert payload["status"] == "completed"
    assert "python_version" in payload
