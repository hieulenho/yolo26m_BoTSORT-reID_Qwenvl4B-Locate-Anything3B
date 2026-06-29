from __future__ import annotations

from pathlib import Path

from football_tracking.utils import environment
from football_tracking.utils.environment import (
    PythonVersionResult,
    validate_python_version,
)


def test_python_312_is_accepted() -> None:
    result = validate_python_version((3, 12, 10), executable="python312.exe")

    assert result.is_supported is True
    assert result.status == "OK"
    assert result.major == 3
    assert result.minor == 12


def test_python_311_is_rejected() -> None:
    result = validate_python_version((3, 11, 9), executable="python311.exe")

    assert result.is_supported is False
    assert result.status == "FAILED"


def test_python_313_is_rejected() -> None:
    result = validate_python_version((3, 13, 0), executable="python313.exe")

    assert result.is_supported is False
    assert result.status == "FAILED"


def test_python_version_error_message_contains_current_and_expected_versions() -> None:
    result = validate_python_version((3, 11, 8), executable="python311.exe")

    assert "3.11.8" in result.message
    assert "3.12.x" in result.message


def test_doctor_fails_when_python_version_is_not_supported(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        environment,
        "validate_python_version",
        lambda: PythonVersionResult(
            version="3.11.8",
            executable="python311.exe",
            major=3,
            minor=11,
            micro=8,
            is_supported=False,
            message=(
                "Python version 3.11.8 is not supported. "
                "Expected Python 3.12.x. "
                "Recreate the virtual environment with: py -3.12 -m venv .venv"
            ),
            status="FAILED",
        ),
    )

    report = environment.run_doctor()

    assert report.status == "FAILED"
    assert report.exit_code != 0
    assert any(
        check.name == "Python version" and check.status == "FAILED" for check in report.checks
    )


def test_setup_script_uses_python_312_launcher_without_fallback() -> None:
    script = Path("scripts/setup_env.ps1").read_text(encoding="utf-8")

    assert "py -3.12 -m venv .venv" in script
    assert '"-3.12", "-m", "venv"' in script
    assert "-3.11" not in script
    assert "-3.13" not in script
    assert "Get-Command python" not in script
