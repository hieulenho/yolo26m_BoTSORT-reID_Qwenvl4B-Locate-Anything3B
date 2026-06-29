from __future__ import annotations

from types import SimpleNamespace

import pytest

from football_tracking.utils import environment
from football_tracking.utils.environment import (
    collect_environment_info,
    collect_torch_info,
    run_doctor,
)


class _FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return False


def test_collect_torch_info_does_not_crash_without_cuda() -> None:
    fake_torch = SimpleNamespace(
        __version__="test",
        cuda=_FakeCuda(),
        version=SimpleNamespace(cuda=None),
    )

    checks, info = collect_torch_info(fake_torch)

    assert info["cuda_available"] is False
    assert any(check.name == "CUDA" and check.status == "WARNING" for check in checks)


def test_doctor_does_not_crash_when_optional_dependency_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import_module = environment.importlib.import_module

    def fake_import_module(name: str):
        if name == "ultralytics":
            raise ModuleNotFoundError("No module named 'ultralytics'")
        return real_import_module(name)

    monkeypatch.setattr(environment.importlib, "import_module", fake_import_module)

    report = run_doctor()

    assert any(check.name == "Ultralytics" and check.status == "WARNING" for check in report.checks)


def test_doctor_checks_output_directory_writable() -> None:
    report = run_doctor()

    assert report.info["output"]["writable"] is True


def test_collect_environment_info_returns_stable_dictionary() -> None:
    info = collect_environment_info()

    assert isinstance(info, dict)
    assert "status" in info
    assert "checks" in info
    assert "info" in info
