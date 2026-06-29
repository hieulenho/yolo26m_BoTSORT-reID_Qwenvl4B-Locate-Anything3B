from pathlib import Path

import pytest

from football_tracking.paths import (
    ProjectPathError,
    ensure_output_directories,
    get_project_root,
    resolve_project_path,
)


def test_get_project_root_returns_directory_containing_pyproject() -> None:
    root = get_project_root()

    assert (root / "pyproject.toml").is_file()


def test_resolve_project_path_rejects_paths_outside_project() -> None:
    with pytest.raises(ProjectPathError):
        resolve_project_path(Path("..") / "outside")


def test_ensure_output_directories_creates_expected_directories(tmp_path: Path) -> None:
    project_root = tmp_path
    (project_root / "pyproject.toml").write_text("[project]\nname = 'tmp'\n", encoding="utf-8")

    created = ensure_output_directories(project_root=project_root)

    assert project_root / "outputs" in created
    assert (project_root / "outputs" / "logs").is_dir()
    assert all(path.is_relative_to(project_root) for path in created)
