"""Force local temp directories when Python is launched from this repository."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_TEMP = PROJECT_ROOT / "outputs" / "pytest_tmp"

try:
    LOCAL_TEMP.mkdir(parents=True, exist_ok=True)
except OSError:
    pass
else:
    temp_path = str(LOCAL_TEMP)
    os.environ["TMP"] = temp_path
    os.environ["TEMP"] = temp_path
    os.environ["TMPDIR"] = temp_path
    tempfile.tempdir = temp_path
