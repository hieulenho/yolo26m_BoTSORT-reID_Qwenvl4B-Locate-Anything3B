"""Environment and dependency checks for the doctor command."""

from __future__ import annotations

import importlib
import platform
import sys
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from football_tracking.config import ConfigError, load_config
from football_tracking.paths import ProjectPathError, ensure_output_directories, get_project_root

Status = Literal["OK", "WARNING", "FAILED"]
EXPECTED_PYTHON_MAJOR = 3
EXPECTED_PYTHON_MINOR = 12
EXPECTED_PYTHON_LABEL = "3.12.x"


@dataclass(frozen=True)
class CheckResult:
    status: Status
    name: str
    message: str
    critical: bool = False


@dataclass(frozen=True)
class PythonVersionResult:
    version: str
    executable: str
    major: int
    minor: int
    micro: int
    is_supported: bool
    message: str
    expected: str = EXPECTED_PYTHON_LABEL
    status: Status = "OK"


@dataclass(frozen=True)
class DoctorReport:
    checks: tuple[CheckResult, ...]
    info: dict[str, Any]

    @property
    def counts(self) -> Counter[str]:
        return Counter(check.status for check in self.checks)

    @property
    def status(self) -> Status:
        if self.counts["FAILED"]:
            return "FAILED"
        if self.counts["WARNING"]:
            return "WARNING"
        return "OK"

    @property
    def exit_code(self) -> int:
        return 1 if any(check.status == "FAILED" and check.critical for check in self.checks) else 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "exit_code": self.exit_code,
            "summary": dict(self.counts),
            "checks": [asdict(check) for check in self.checks],
            "info": self.info,
        }


def _module_version(module: Any) -> str:
    return str(getattr(module, "__version__", "unknown"))


def _version_parts(version_info: Any) -> tuple[int, int, int]:
    return int(version_info[0]), int(version_info[1]), int(version_info[2])


def validate_python_version(
    version_info: Any | None = None,
    executable: str | None = None,
) -> PythonVersionResult:
    """Validate that the active interpreter is exactly Python 3.12.x."""

    current_version_info = sys.version_info if version_info is None else version_info
    major, minor, micro = _version_parts(current_version_info)
    version = f"{major}.{minor}.{micro}"
    current_executable = sys.executable if executable is None else executable
    is_supported = major == EXPECTED_PYTHON_MAJOR and minor == EXPECTED_PYTHON_MINOR

    if is_supported:
        return PythonVersionResult(
            version=version,
            executable=current_executable,
            major=major,
            minor=minor,
            micro=micro,
            is_supported=True,
            message=(
                f"Python version {version} is supported. Expected Python {EXPECTED_PYTHON_LABEL}."
            ),
            status="OK",
        )

    return PythonVersionResult(
        version=version,
        executable=current_executable,
        major=major,
        minor=minor,
        micro=micro,
        is_supported=False,
        message=(
            f"Python version {version} is not supported. "
            f"Expected Python {EXPECTED_PYTHON_LABEL}. "
            "Recreate the virtual environment with: py -3.12 -m venv .venv"
        ),
        status="FAILED",
    )


def _check_import(display_name: str, import_name: str) -> tuple[CheckResult, dict[str, Any]]:
    try:
        module = importlib.import_module(import_name)
    except Exception as exc:  # noqa: BLE001 - doctor reports import failures without crashing.
        return (
            CheckResult(
                "WARNING",
                display_name,
                f"{display_name} is not importable: {exc}",
            ),
            {"available": False, "error": str(exc), "version": None},
        )

    version = _module_version(module)
    return (
        CheckResult("OK", display_name, f"{display_name}: {version}"),
        {"available": True, "error": None, "version": version},
    )


def collect_torch_info(torch_module: Any | None = None) -> tuple[list[CheckResult], dict[str, Any]]:
    checks: list[CheckResult] = []

    if torch_module is None:
        try:
            torch_module = importlib.import_module("torch")
        except Exception as exc:  # noqa: BLE001 - torch is optional for doctor.
            return (
                [CheckResult("WARNING", "PyTorch", f"PyTorch is not importable: {exc}")],
                {
                    "available": False,
                    "error": str(exc),
                    "version": None,
                    "cuda_available": False,
                    "cuda_version": None,
                    "gpu_count": 0,
                    "gpu_names": [],
                },
            )

    torch_version = _module_version(torch_module)
    cuda = getattr(torch_module, "cuda", None)
    version_info = getattr(torch_module, "version", None)
    cuda_version = getattr(version_info, "cuda", None)

    try:
        cuda_available = bool(cuda.is_available()) if cuda is not None else False
    except Exception as exc:  # noqa: BLE001
        checks.append(CheckResult("WARNING", "CUDA", f"Could not query CUDA availability: {exc}"))
        cuda_available = False

    try:
        gpu_count = int(cuda.device_count()) if cuda_available and cuda is not None else 0
    except Exception as exc:  # noqa: BLE001
        checks.append(CheckResult("WARNING", "GPU count", f"Could not query GPU count: {exc}"))
        gpu_count = 0

    gpu_names: list[str] = []
    if cuda_available and cuda is not None:
        for index in range(gpu_count):
            try:
                gpu_names.append(str(cuda.get_device_name(index)))
            except Exception as exc:  # noqa: BLE001
                checks.append(
                    CheckResult("WARNING", "GPU name", f"Could not query GPU {index}: {exc}")
                )

    checks.append(CheckResult("OK", "PyTorch", f"PyTorch: {torch_version}"))
    if cuda_available:
        checks.append(CheckResult("OK", "CUDA", f"CUDA is available: {cuda_version or 'unknown'}"))
    else:
        checks.append(CheckResult("WARNING", "CUDA", "CUDA is not available"))

    if gpu_count > 0:
        checks.append(CheckResult("OK", "GPU", f"{gpu_count} GPU(s): {', '.join(gpu_names)}"))
    else:
        checks.append(CheckResult("WARNING", "GPU", "No GPU detected"))

    return (
        checks,
        {
            "available": True,
            "error": None,
            "version": torch_version,
            "cuda_available": cuda_available,
            "cuda_version": cuda_version,
            "gpu_count": gpu_count,
            "gpu_names": gpu_names,
        },
    )


def _check_python_executable_location(
    project_root: Path,
    executable: str,
) -> tuple[CheckResult, dict[str, Any]]:
    venv_dir = (project_root / ".venv").resolve()
    executable_path = Path(executable).resolve()
    is_in_venv = executable_path.is_relative_to(venv_dir)

    if is_in_venv:
        return (
            CheckResult(
                "OK",
                "Python virtual environment",
                f"Python executable is inside: {venv_dir}",
            ),
            {
                "expected_venv": str(venv_dir),
                "executable": str(executable_path),
                "is_inside_project_venv": True,
            },
        )

    return (
        CheckResult(
            "WARNING",
            "Python virtual environment",
            f"Python executable is not inside project .venv: {executable_path}",
        ),
        {
            "expected_venv": str(venv_dir),
            "executable": str(executable_path),
            "is_inside_project_venv": False,
        },
    )


def _check_output_writable(project_root: Path) -> tuple[CheckResult, dict[str, Any]]:
    try:
        ensure_output_directories(project_root=project_root)
        output_dir = project_root / "outputs"
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_dir,
            prefix=".doctor_",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_file.write("doctor write test\n")
            temp_path = Path(temp_file.name)
        temp_path.unlink(missing_ok=True)
    except Exception as exc:  # noqa: BLE001
        return (
            CheckResult(
                "FAILED",
                "Output directory",
                f"Output directory is not writable: {exc}",
                critical=True,
            ),
            {"writable": False, "error": str(exc)},
        )

    return (
        CheckResult("OK", "Output directory", f"Output directory is writable: {output_dir}"),
        {"writable": True, "error": None, "path": str(output_dir)},
    )


def run_doctor(config_path: str | Path | None = None) -> DoctorReport:
    checks: list[CheckResult] = []
    python_result = validate_python_version()
    info: dict[str, Any] = {
        "python_executable": python_result.executable,
        "python_version": python_result.version,
        "operating_system": platform.platform(),
        "current_working_directory": str(Path.cwd()),
        "python": asdict(python_result),
    }

    try:
        project_root = get_project_root()
        checks.append(CheckResult("OK", "Project root", f"Project root: {project_root}"))
        info["project_root"] = str(project_root)
    except ProjectPathError as exc:
        project_root = Path.cwd().resolve()
        checks.append(CheckResult("FAILED", "Project root", str(exc), critical=True))
        info["project_root"] = None
        info["project_root_error"] = str(exc)

    checks.append(
        CheckResult(
            python_result.status,
            "Python version",
            f"Python version: {python_result.version}. {python_result.message}",
            critical=not python_result.is_supported,
        )
    )
    checks.append(
        CheckResult("OK", "Python executable", f"Python executable: {python_result.executable}")
    )
    executable_check, executable_info = _check_python_executable_location(
        project_root,
        python_result.executable,
    )
    checks.append(executable_check)
    info["python"]["virtual_environment"] = executable_info

    checks.append(CheckResult("OK", "Operating system", f"Operating system: {platform.platform()}"))
    checks.append(
        CheckResult("OK", "Current working directory", f"Current working directory: {Path.cwd()}")
    )

    torch_checks, torch_info = collect_torch_info()
    checks.extend(torch_checks)
    info["torch"] = torch_info

    dependency_checks = {
        "ultralytics": _check_import("Ultralytics", "ultralytics"),
        "opencv": _check_import("OpenCV", "cv2"),
        "numpy": _check_import("NumPy", "numpy"),
        "deep_sort_realtime": _check_import("DeepSORT realtime", "deep_sort_realtime"),
    }
    for key, (check, dependency_info) in dependency_checks.items():
        checks.append(check)
        info[key] = dependency_info

    try:
        config = load_config(config_path=config_path, project_root=project_root)
    except ConfigError as exc:
        checks.append(CheckResult("FAILED", "Config", f"Config could not be loaded: {exc}", True))
        info["config"] = {"readable": False, "error": str(exc)}
    else:
        checks.append(CheckResult("OK", "Config", f"Config loaded: {config.config_path}"))
        info["config"] = {
            "readable": True,
            "error": None,
            "path": str(config.config_path),
            "resolved_paths": {key: str(value) for key, value in config.paths.items()},
        }

    output_check, output_info = _check_output_writable(project_root)
    checks.append(output_check)
    info["output"] = output_info
    return DoctorReport(tuple(checks), info)


def collect_environment_info(config_path: str | Path | None = None) -> dict[str, Any]:
    """Return a stable dictionary representation of the doctor report."""

    return run_doctor(config_path=config_path).to_dict()


def format_doctor_report(report: DoctorReport) -> str:
    lines = [f"[{check.status}] {check.message}" for check in report.checks]
    counts = report.counts
    lines.extend(
        [
            "",
            "Summary:",
            f"OK: {counts['OK']}",
            f"Warnings: {counts['WARNING']}",
            f"Failed: {counts['FAILED']}",
            f"Status: {report.status}",
        ]
    )
    return "\n".join(lines)
