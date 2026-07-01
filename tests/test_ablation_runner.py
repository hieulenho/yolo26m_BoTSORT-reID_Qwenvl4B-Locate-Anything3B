from __future__ import annotations

from football_tracking.experiments.ablation import generate_ablation_plan, run_tracker_ablation


def test_ablation_plan_is_deterministic() -> None:
    first = generate_ablation_plan("configs/tracker_ablation.yaml")[:3]
    second = generate_ablation_plan("configs/tracker_ablation.yaml")[:3]
    assert [item.experiment_id for item in first] == [item.experiment_id for item in second]


def test_ablation_dry_run_does_not_write_files(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FOOTBALL_TRACKING_ROOT", str(tmp_path))
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    config = tmp_path / "ablation.yaml"
    config.write_text(
        """
base_experiment:
  config: compare.yaml
ablation:
  sort:
    max_age: [15]
strategy:
  mode: one_factor_at_a_time
  max_experiments: 2
  include_baseline: true
runtime:
  resume: true
""".strip(),
        encoding="utf-8",
    )

    result = run_tracker_ablation(config, dry_run=True)

    assert result["dry_run"] is True
    assert not (tmp_path / "outputs" / "experiments" / "ablation" / "manifest.json").exists()
