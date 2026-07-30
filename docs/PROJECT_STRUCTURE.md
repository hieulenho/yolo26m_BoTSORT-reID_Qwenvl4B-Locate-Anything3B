# Project Structure

The repository separates deployment contracts, reusable package code, reproducible evaluation,
and generated artifacts.

## Supported Runtime

```text
configs/tasks/                         task intent, detector, taxonomy, policy
configs/trackers/                      immutable tracker presets
configs/semantics/dynamic_track.yaml   one-Qwen 8-bit execution profile
scripts/run_task_realtime.ps1          video, camera, or RTSP entrypoint
scripts/run_task_webcam.ps1            local-camera probe and entrypoint
src/football_tracking/task_pipeline/   TaskConfig validation and runtime builder
src/football_tracking/tracking/        detector-to-track pipeline and adapters
src/football_tracking/adaptive_tracking/ queue, evidence, fusion, cache, realtime loop
src/football_tracking/vlm/             Qwen loader and bounded inference session
requirements/runtime.txt               lightweight detector/tracker runtime
requirements/webcam.txt                runtime plus Qwen semantic worker
```

Scripts call package services. Package modules do not import scripts or generated output.

## Evaluation

```text
configs/benchmarks/                     fixed measurement contracts
scripts/benchmarks/                     run and consolidate experiments
src/football_tracking/benchmarking/     report builders and schema validation
src/football_tracking/evaluation/       TrackEval and IDSW diagnostics
outputs/benchmarks/research_final/      local canonical report artifacts
docs/benchmarks/research_final/         lightweight publishable report, CSVs, figures
requirements/base.txt                   runtime plus TrackEval/report dependencies
tests/                                  unit, integration, and contract tests
```

## Data And Models

```text
data/       local datasets, normalized GT, review manifests
models/     promoted local checkpoints
outputs/    generated runs, caches, videos, metrics, reports
```

Large videos, datasets, model weights, caches, and ordinary run outputs are ignored by Git.
Reviewed manifests and lightweight canonical research artifacts may be versioned.

## Legacy Boundary

`scripts/legacy/`, `configs/legacy/`, `docs/legacy/`, and `src/football_tracking/locate_tracking/`
exist for historical LocateAnything/Qwen A/B/C reproduction. They are not loaded by the current
TaskConfig runtime. Removing them entirely would break reproducibility of older saved reports, so
they remain isolated rather than mixed with primary entrypoints.

## Cleanup Policy

Safe disposable artifacts include Python caches, Ruff/Pytest caches, temporary slide-generation
folders, and noncanonical smoke outputs. Preserve reviewed GT, promoted checkpoints, benchmark
inputs referenced by `configs/benchmarks/task_research_report.yaml`, and the published report.
