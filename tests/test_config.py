from pathlib import Path

import pytest

from football_tracking.config import ConfigError, load_config
from football_tracking.paths import get_project_root


def test_loads_base_config_and_resolves_paths() -> None:
    root = get_project_root()

    config = load_config()

    assert config.project["name"] == "football-player-tracking"
    assert config.paths["data_dir"] == root / "data"
    assert config.paths["logs_dir"] == root / "outputs" / "logs"


def test_missing_config_raises_clear_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="does not exist"):
        load_config(tmp_path / "missing.yaml", project_root=tmp_path)


def test_invalid_yaml_raises_clear_error(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("project: [broken\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="Invalid YAML"):
        load_config(config_path, project_root=tmp_path)


def test_missing_required_section_raises_clear_error(tmp_path: Path) -> None:
    config_path = tmp_path / "missing_section.yaml"
    config_path.write_text(
        """
project:
  name: test
paths:
  data_dir: data
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="missing required section"):
        load_config(config_path, project_root=tmp_path)
