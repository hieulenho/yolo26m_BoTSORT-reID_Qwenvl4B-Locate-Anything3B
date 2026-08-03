# Script Guide

Run supported commands from the repository root in PowerShell.

## Primary Entry Points

| Script | Purpose |
|---|---|
| `setup_webcam.ps1` | create/verify the Python 3.12 detector, tracker, and Qwen runtime |
| `save_camera_credential.ps1` | save a Yoosee RTSP password once with Windows DPAPI |
| `run_task_webcam.ps1` | select a local or RTSP camera by logical CameraIndex and run one TaskConfig |
| `run_camera.ps1` | compatibility alias for `run_task_webcam.ps1` |
| `run_ip_camera.ps1` | probe a Yoosee/ONVIF RTSP stream and run one TaskConfig |
| `run_task_realtime.ps1` | run one TaskConfig on a camera, stream, or video |
| `run_tests.ps1` | execute the repository test gate |

Example:

```powershell
.\scripts\run_camera.ps1 -ListOnly

.\scripts\run_camera.ps1 `
  -CameraIndex 1 `
  -TaskConfig configs\tasks\generic_coco_realtime.yaml `
  -SemanticWorkerMode live `
  -Overwrite
```

`run_task_webcam.ps1` and its `run_camera.ps1` alias number all readable local webcams first,
followed by RTSP hosts discovered
on bounded private subnets. The numbering is a convenience inventory, not a Windows hardware
ID, so list it again after adding a camera or changing networks.

The lower-level local-camera command remains available:

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

The focused UA-DETRAC tracking/semantic quality report is generated with:

```powershell
.\.venv\Scripts\python.exe scripts\benchmarks\build_traffic_quality_report.py `
  --config configs\benchmarks\traffic_quality_report.yaml `
  --overwrite
```

## Legacy

`scripts/legacy/` and the older adaptive wrappers are retained only to reproduce historical
A/B/C artifacts. New deployments must use `run_task_realtime.ps1` or `run_task_webcam.ps1`.
The supported Locate -> Qwen quality path is the `-LocateFirst` deferred mode of
`run_task_realtime.ps1`. See the root `commands.txt` for tested command lines.
