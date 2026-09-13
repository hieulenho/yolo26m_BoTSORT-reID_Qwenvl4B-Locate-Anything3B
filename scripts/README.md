# Script guide

Run supported commands from the repository root in PowerShell.

| Script | Purpose |
| --- | --- |
| `setup_webcam.ps1` | Install/verify Python 3.12 CPU or CUDA runtime |
| `run_traffic.ps1` | Default traffic webcam/video entry point |
| `run_task_realtime.ps1` | Advanced TaskConfig execution, live or deferred |
| `run_task_webcam.ps1` | Choose a local/RTSP camera by logical inventory index |
| `run_camera.ps1` | Alias for task webcam discovery |
| `run_ip_camera.ps1` | Authenticated RTSP camera connection |
| `save_camera_credential.ps1` | Store camera credentials using Windows DPAPI |
| `runtime/download_models.py` | Prefetch public YOLO and optional Qwen weights |
| `run_tests.ps1` | Run the regression checks |

```powershell
.\scripts\run_traffic.ps1 -Source 0
.\scripts\run_traffic.ps1 -Source "C:\Videos\traffic.mp4"
```

`live` keeps a Qwen worker running beside tracking. `deferred` processes evidence
after file tracking and re-renders labels. `disabled` runs detection/tracking
only. Deferred results must not be called realtime semantic predictions.

`scripts/runtime/` contains subprocess helpers. `scripts/benchmarks/` contains
reusable MOT/semantic evaluation and traffic report tools. Read
[`docs/BENCHMARKS.md`](../docs/BENCHMARKS.md) for required data and protocols.
For the optional Locate -> Qwen path, use `-LocateFirst` with `deferred` mode.
Historical football wrappers and team-label scripts are no longer public entry points.
