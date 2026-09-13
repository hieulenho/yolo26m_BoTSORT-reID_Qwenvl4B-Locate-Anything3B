# Usage

Run commands from the repository root in PowerShell after `setup_webcam.ps1`.
The supported one-command entry point is `scripts/run_traffic.ps1`.

## Advanced video evaluation

```powershell
.\scripts\run_task_realtime.ps1 `
  -TaskConfig configs\tasks\traffic_objects_quality.yaml `
  -Source "C:\Videos\traffic.mp4" `
  -RunName traffic_quality `
  -SemanticWorkerMode deferred `
  -Device cuda `
  -DisableFrameDropping `
  -SynchronousVideoWrite `
  -NoWindow
```

Deferred mode finishes tracking, processes saved evidence, then re-renders the
file with accepted labels. This is **offline**, not a measurement of live deep
label latency. Add `-LocateFirst` only after installing
`requirements/locate_tracking.txt`. LocateAnything and Qwen load sequentially.
For the separate YOLOE proposal profile install `requirements/open_vocab.txt`.
These optional stacks are not installed for the default live profile.

## Outputs

Default location: `outputs/traffic/<RunName>/` (`outputs/task_realtime` when using
the lower-level runner directly).

| File | Meaning |
| --- | --- |
| `tracked_semantic.mp4` | Frames written by the preview/render loop |
| `tracked_semantic.txt` | Per-frame tracked boxes, IDs, confidence and base class |
| `realtime_metrics.json` | Timing, processed/captured frames, dropping and runtime diagnostics |
| `runtime/task.resolved.json` | Exact task settings used |
| `runtime/tracking.generated.yaml` | Detector and tracker configuration |
| `semantic_queue/` | Saved evidence and queued/completed jobs |
| `semantic_cache.json` | Accepted labels indexed by track |
| `semantic_memory.json` | Temporal semantic evidence |
| `semantic_worker_report.json` | Worker result and timing report |
| `semantic_worker.stderr.log` | Model warnings/errors |

Semantic files are absent in disabled mode or before the first semantic result.
A label confidence is not a measured accuracy. Track IDs can change after a
scene reset or tracking failure; they are not permanent identities of people.
With frame dropping, the saved video is a processing output and must not be
treated as a frame-complete input copy. Use timestamp/metrics for latency.

## Troubleshooting

- **No suitable Python runtime:** install 64-bit Python 3.12 and reopen the shell,
  or pass `-PythonExecutable "C:\Path To Python\python.exe"` to setup.
- **Scripts blocked:** `Set-ExecutionPolicy -Scope Process Bypass` affects only
  the current shell; no machine-wide security setting is changed.
- **CUDA unavailable:** update your NVIDIA driver, then rerun CUDA setup, or use
  `-Device cpu -SemanticWorkerMode disabled`. Do not expect CPU Qwen to be realtime.
- **OOM:** close other GPU programs, use the default lean profile (no YOLOE),
  and compare with `-SemanticWorkerMode disabled`. A bounded queue limits waiting
  work, not the model's weight/activation memory.
- **Pending labels:** inspect worker stderr/status. First use includes a model
  download and cold load. If inference is slower than new-object arrival, some
  tracks will leave before a deep label is ready.
- **Camera busy:** close Camera/Teams/OBS and allow desktop apps camera access.
  Try another `-Source` index. The camera remains local; the default pipeline does
  not upload its frames to an inference service.
- **Offline run:** prefetch with `setup_webcam.ps1 -DownloadModels` while online.
  Keep the Hugging Face cache and YOLO weights. Installing packages still needs
  network access unless you have prepared a wheel cache yourself.

Do not commit camera credentials or RTSP URLs containing passwords. Network
cameras must be configured on a reachable network before the project can read
their stream. The project cannot bypass camera authentication.
