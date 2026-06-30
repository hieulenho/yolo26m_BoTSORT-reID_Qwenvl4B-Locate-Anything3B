from pathlib import Path

from football_tracking.detection.trainer import YOLOv8Trainer
from football_tracking.detection.training_config import load_training_config
from tests.test_training_config import _config
from tests.test_training_preflight import _valid_dataset


class _FakeModel:
    def train(self, **kwargs: object) -> str:
        run_dir = Path(str(kwargs["project"])) / str(kwargs["name"])
        weights = run_dir / "weights"
        weights.mkdir(parents=True, exist_ok=True)
        (weights / "best.pt").write_bytes(b"best")
        (weights / "last.pt").write_bytes(b"last")
        (run_dir / "results.csv").write_text("epoch,metrics/mAP50(B)\n1,0.1\n", encoding="utf-8")
        return "trained"


def test_training_smoke_with_fake_model_writes_artifacts(tmp_path: Path) -> None:
    _valid_dataset(tmp_path)
    config = load_training_config(_config(tmp_path))
    trainer = YOLOv8Trainer(config, model_factory=lambda _weights: _FakeModel())

    result = trainer.train()

    assert result["artifacts"]["best_checkpoint"].is_file()
    assert (config.run_dir / "experiment_manifest.json").is_file()
