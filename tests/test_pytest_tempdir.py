from pathlib import Path


def test_pytest_tmp_path_stays_inside_project(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    expected_root = project_root / "outputs" / "pytest_tmp"

    assert tmp_path.resolve().is_relative_to(expected_root.resolve())
