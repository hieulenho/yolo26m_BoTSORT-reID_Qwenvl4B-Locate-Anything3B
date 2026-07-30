# Script Guide

Run supported commands from the repository root in PowerShell.

## Primary Entry Points

| Script | Purpose |
|---|---|
| `setup_webcam.ps1` | create/verify the Python 3.12 detector, tracker, and Qwen runtime |
| `run_task_webcam.ps1` | probe a local camera and run one TaskConfig |
| `run_task_realtime.ps1` | run one TaskConfig on a camera, stream, or video |
| `run_tests.ps1` | execute the repository test gate |

Example:

```powershell
.\scripts\run_task_webcam.ps1 `
  -TaskConfig configs\tasks\generic_coco_realtime.yaml `
  -CameraIndex 0 `
  -SemanticWorkerMode live `
  -Overwrite
```

Semantic modes:

- `live`: foreground tracking and a persistent asynchronous Qwen worker;
- `deferred`: queue evidence during tracking and run Qwen after capture;
- `disabled`: detector/tracker only, used for objective foreground benchmarks.

`setup_webcam.ps1` installs the lightweight `requirements/runtime.txt` and Qwen dependencies.
The broader `requirements/base.txt` is reserved for benchmark and report generation.

## Runtime Helpers

`scripts/runtime/` contains Python processes called by the PowerShell entrypoints. They are
implementation details, but remain directly testable.

## Benchmarks

`scripts/benchmarks/` contains GT conversion, detector/tracker evaluation, repeated runtime
measurement, semantic evaluation, IDSW diagnostics, and report generation. The current report
entrypoint is:

```powershell
.\.venv\Scripts\python.exe scripts\benchmarks\build_task_research_report.py `
  --config configs\benchmarks\task_research_report.yaml `
  --overwrite
```

## Legacy

`scripts/legacy/` and the older adaptive wrappers are retained only to reproduce historical
LocateAnything/Qwen A/B/C artifacts. New deployments must use `run_task_realtime.ps1` or
`run_task_webcam.ps1`. See the root `commands.txt` for tested command lines.
