# Config Guide

Human-edited configs define deployment intent and experiment contracts. Generated runtime YAML
belongs inside a run directory and must not be edited manually.

## TaskConfig

`configs/tasks/` is the main interface. Each YAML defines:

- task ID, name, and objective;
- detector backend, checkpoint, classes, thresholds, and preprocessing;
- tracker implementation, preset, and class stabilization;
- Qwen taxonomy, requested attributes, instruction, unknown threshold, and evidence limits;
- render options.

The production semantic model is locked to `Qwen/Qwen3-VL-4B-Instruct` with 8-bit
quantization. Closed tasks must provide an explicit allowed-label taxonomy. Open tasks may emit
hierarchical labels, but still apply confidence and unknown rejection.

## Tracker Presets

| Preset | Purpose |
|---|---|
| `tracktrack_realtime.yaml` | single-class realtime default |
| `tracktrack_multiclass_realtime.yaml` | class-agnostic multiclass association plus temporal class stabilization |
| `tracktrack_open_realtime.yaml` | low-threshold open-vocabulary detections |
| `tracktrack_reid_realtime.yaml` | TrackTrack appearance-CNN ablation |
| `botsort_reid_identity_stable.yaml` | identity-focused comparison |
| `ocsort_realtime.yaml` | high-throughput motion-only comparison |

The tracker choice must be justified by a shared-detection benchmark, not FPS alone.

## Semantic Runtime

`configs/semantics/dynamic_track.yaml` contains the bounded Qwen execution profile: one track per
batch, at most two model images, 8-bit weights, and a 192-token structured answer. Task-specific
labels and instructions are copied into each atomic queue event.

## Benchmarks

`configs/benchmarks/task_research_report.yaml` identifies canonical input artifacts and output
locations. `task_runtime_suite.json` defines the repeated foreground matrix. Tracker and detector
contracts under the same directory retain fixed splits, image size, thresholds, and provenance.

## Optional And Historical Configs

YOLOE requires `requirements/open_vocab.txt`. LocateAnything and old A/B/C experiments are
legacy ablations and are not part of the production TaskConfig runtime.
