from pathlib import Path

import yaml

from football_tracking.detection.evaluate import evaluate_detector


class _Box:
    mp = 0.1
    mr = 0.2
    map50 = 0.3
    map = 0.4
    map75 = 0.5


class _Result:
    box = _Box()


class _Model:
    def val(self, **_kwargs: object) -> _Result:
        return _Result()


def test_detector_evaluation_writes_val_metrics(tmp_path: Path) -> None:
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"checkpoint")
    dataset = tmp_path / "dataset.yaml"
    dataset.write_text(
        yaml.safe_dump(
            {
                "path": str(tmp_path),
                "val": "images/val",
                "names": {0: "player"},
                "nc": 1,
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "eval.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "model": {"weights": str(weights)},
                "dataset": {"data_yaml": str(dataset), "split": "val"},
                "evaluation": {
                    "imgsz": 640,
                    "batch": 1,
                    "conf": 0.001,
                    "iou": 0.7,
                    "max_det": 10,
                    "device": "cpu",
                },
                "output": {
                    "project": str(tmp_path / "eval"),
                    "run_name": "val",
                    "metrics_dir": str(tmp_path / "metrics"),
                    "figures_dir": str(tmp_path / "figures"),
                },
                "runtime": {"overwrite": True},
            }
        ),
        encoding="utf-8",
    )

    result = evaluate_detector(config, model=_Model())

    assert result["metrics"]["map50"] == 0.3
    assert Path(result["paths"]["json"]).is_file()
