"""Repository-wide pytest settings for Windows-safe temporary directories."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent
PYTEST_TEMP_ROOT = PROJECT_ROOT / "outputs" / "pytest_tmp"


def configure_local_pytest_temp() -> Path:
    PYTEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    temp_path = str(PYTEST_TEMP_ROOT)
    os.environ["TMP"] = temp_path
    os.environ["TEMP"] = temp_path
    os.environ["TMPDIR"] = temp_path
    tempfile.tempdir = temp_path
    return PYTEST_TEMP_ROOT


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    temp_root = configure_local_pytest_temp()
    if config.option.basetemp is None:
        config.option.basetemp = str(temp_root / f"run-{os.getpid()}")
