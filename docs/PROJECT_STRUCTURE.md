# Repository layout

| Path | Responsibility |
| --- | --- |
| `scripts/setup_webcam.ps1` | Python environment, CPU/CUDA dependencies and diagnostics |
| `scripts/run_traffic.ps1` | Main webcam/video entry point |
| `scripts/run_task_realtime.ps1` | Advanced live/deferred execution and worker lifecycle |
| `configs/tasks/traffic*.yaml` | Traffic detector/tracker/taxonomy profiles |
| `configs/semantics/` | Qwen model and evidence limits |
| `configs/trackers/` | Reproducible tracker settings |
| `src/football_tracking/task_pipeline/` | Task validation and runtime configuration |
| `src/football_tracking/tracking/` | Detection association, identity state and tracking outputs |
| `src/football_tracking/adaptive_tracking/` | Capture, scheduling, semantic memory and rendering |
| `src/football_tracking/vlm/` | Qwen loading, quantization and inference |
| `src/football_tracking/locate_tracking/` | Optional grounding and shared evidence utilities |
| `scripts/benchmarks/` | Evaluation utilities |
| `docs/benchmarks/traffic_quality/` | Curated traffic measurements and figures |
| `tests/` | Regression tests and small synthetic/parser fixtures |
| `data/`, `models/`, `outputs/` | Ignored local inputs, weights and generated results |

Public football scripts, team-label benchmarks, old presentation generators and
superseded reports have been retired. They remain recoverable from Git history,
not from parallel legacy entry points in this checkout. Generic MOT conversion,
evaluation and algorithm tests are kept because traffic uses the same core logic.
