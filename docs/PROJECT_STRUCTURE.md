# Project Structure

The repository separates reusable source code, human-edited experiment definitions, local
model/data assets, and generated evidence.

## Package Boundaries

```text
src/football_tracking/
  adaptive_tracking/    discovery, ontology, routing, realtime loop, fusion, render
  benchmarking/         detector/tracker/semantic consolidation and final report
  data/                 SportsMOT preparation, conversion, validation, audit
  detection/            Ultralytics YOLO/YOLOE adapters and composite detector routing
  evaluation/           TrackEval integration and IDSW diagnostic taxonomy
  experiments/          reproducible shared-cache tracker experiments
  locate_tracking/      LocateAnything grounding, association, memory, and verification
  reporting/            run-level provenance and report helpers
  tracking/             tracker registry, adapters, MOT pipeline, timing
  visualization/        frame overlays and plots
  vlm/                  Qwen loading, quantization, scene discovery, prompt/context building
```

Dependency direction is intentionally one way: CLI/scripts call package services; package
modules do not import scripts or generated outputs.

## Human-Edited Files

```text
configs/adaptive_tracking.yaml       primary adaptive defaults
configs/ontology/                    canonical vocabulary registry
configs/trackers/                    immutable tracker presets
configs/benchmarks/                  benchmark contracts and source manifests
requirements/                        dependency groups
scripts/                             small set of supported PowerShell entry points
scripts/runtime/                     realtime and scene-discovery workers
scripts/benchmarks/                  metric, audit, and report builders
scripts/data/                        sample acquisition and input diagnostics
scripts/legacy/                      compatibility-only workflows
tests/                               regression and artifact-contract tests
README.md                            public project entry point
commands.txt                         complete terminal runbook
```

## Generated Artifacts

```text
outputs/adaptive_runs/               per-video discovery, plan, semantics, report
outputs/benchmarks/detection/        canonical detector comparison
outputs/benchmarks/tracking/         canonical tracking and IDSW comparison
outputs/benchmarks/semantic/         reviewed semantic A/B/C ablation
outputs/benchmarks/runtime/          route-level realtime measurements
outputs/benchmarks/realtime/         long-stream latency/drop/resource comparisons
outputs/benchmarks/final/            consolidated local report
outputs/cache/semantic_discovery/    reusable discovery cache keyed by source/config
outputs/detections/cache/            shared detections for fair tracker comparison
data/semantic_benchmark/review/      human-review CSV, provenance, and contact sheets
```

Publishable lightweight results live under `docs/benchmarks/` and
`docs/assets/benchmarks/`, so README links continue to work on GitHub even though `outputs/`
is ignored.

## Local Assets

`data/`, `models/`, root checkpoint files, videos, and model text encoders are local assets.
They are intentionally ignored when large or licensed separately. Do not commit:

```text
*.pt, *.pth, *.safetensors, mobileclip*.ts, videos, datasets, outputs, runs
```

## Legacy Compatibility

The older football-only A/B/C scripts, configs, and documents remain under `scripts/legacy/`,
`configs/legacy/football/`, and `docs/legacy/` for reproducing historical artifacts. New work
should enter through `scripts/run_adaptive_tracking.ps1` or
`scripts/run_realtime_adaptive.ps1`.
Do not mix old result folders with `outputs/benchmarks/final/`.

## Cleanup Rules

- Safe to remove: Python caches, root archive backups, `runs/`, and disposable smoke outputs.
- Preserve: `data/`, promoted checkpoints under `models/`, reviewed GT manifests, and canonical
  benchmark sources referenced by `configs/benchmarks/final_report.yaml`.
- Never delete a benchmark source before rebuilding the final report and checking
  `artifact_audit.json`.
