# Project Structure

This repository is organized around reusable detection, tracking, evaluation, and VLM stages.

## Source Code

```text
src/football_tracking/
  cli.py                    main argparse CLI
  data/                     dataset discovery, conversion, validation, audit
  detection/                YOLO detector, training, evaluation, detection cache
  tracking/                 SORT, DeepSORT, BoT-SORT/ByteTrack adapters and MOT writing
  experiments/              shared-cache tracker comparison and grid search
  evaluation/               TrackEval integration
  rendering/                annotated video rendering from MOT files
  visualization/            plotting and overlay helpers
  reporting/                Markdown/CSV/JSON report generation
  domains/                  reusable domain profile config builder
  vlm/                      Qwen VLM context creation and optional local inference
```

## Human-Edited Project Files

```text
configs/                    YAML entry points and tracker presets
docs/                       design notes and runbooks
requirements/               base/dev/VLM dependency groups
scripts/                    PowerShell wrappers for common workflows
tests/                      pytest regression suite
README.md                   primary project guide
```

## Runtime Artifacts

These paths are ignored by Git and can be regenerated.

```text
data/raw/                   downloaded datasets
data/yolo/                  YOLO-format prepared datasets
data/mot/                   MOTChallenge-format prepared datasets
models/                     promoted checkpoints
outputs/detections/cache/   per-frame detection cache
outputs/tracks/             MOT tracker outputs
outputs/videos/             rendered tracking videos
outputs/metrics/            JSON/CSV/Markdown metrics
outputs/figures/            generated figures
outputs/training/           Ultralytics training runs
runs/                       Ultralytics default run directory
```

## Naming Conventions

- `*_smoke.yaml`: small/fast plumbing checks.
- `yolov8m_*`: historical baseline configs.
- `yolo26m_*`: current football detector configs.
- `*_all.yaml`: all available SportsMOT football sequences.
- `*_hard_identity.yaml`: focused identity-stress subsets.
- `outputs/.../raw/`: raw third-party tool output, usually TrackEval.

## Safe Cleanup

It is usually safe to delete generated artifacts under `outputs/`, `runs/`, and
`configs/generated/` when you want to rerun experiments. Do not delete `data/` or `models/`
unless you are intentionally rebuilding datasets or retraining/downloading checkpoints.
