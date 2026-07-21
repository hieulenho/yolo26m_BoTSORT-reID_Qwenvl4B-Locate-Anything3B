# Config Guide

Configs are source-controlled experiment definitions. Generated configs belong under an output
run directory and should not be edited by hand.

## Primary Adaptive Config

`adaptive_tracking.yaml` defines:

- Qwen discovery model, 4-bit quantization, shot sampling, and class limit;
- ontology registry and semantic cache root;
- football, COCO, and open-vocabulary detector checkpoints;
- realtime, realtime_stable, balanced, and accuracy tracker profiles;
- event-triggered Qwen/Locate policy and unknown threshold;
- sequential GPU execution for an 8 GB device.

## Ontology

`ontology/vocabulary_registry.yaml` stores canonical names, aliases, COCO mappings, default
actions, and domain hints. It normalizes VLM output; it does not limit Qwen to only the listed
classes. Unknown names remain eligible for the YOLOE route.

## Tracker Presets

| Preset | Role |
|---|---|
| `ocsort_realtime.yaml` | default live tracker |
| `tracktrack_realtime.yaml` | balanced quality profile |
| `botsort_reid_identity_stable.yaml` | identity-focused profile |
| `deepocsort_reid_realtime.yaml` | appearance-CNN comparison |
| `fasttrack_realtime.yaml` | low-latency comparison |
| `bytetrack_fast.yaml` | non-ReID baseline |

Tracker selection is backed by `configs/benchmarks/tracking_full_report.yaml`; do not infer the
best tracker from FPS alone.

## Benchmark Contracts

```text
benchmarks/detector_sportsmot.yaml
benchmarks/tracking_sportsmot_yolo26m*.yaml
benchmarks/tracking_full_report.yaml
benchmarks/semantic_pipelines.yaml
benchmarks/semantic_ablation.yaml
benchmarks/final_report.yaml
```

These files point to immutable source artifacts and expected sequence/frame/track counts. The
final report fails instead of silently accepting missing or incompatible inputs.

## Dynamic Semantic Config

`vlm_dynamic_track_semantics.yaml` uses an open output schema. Qwen receives global keyframes,
track crops, and structured MOT metadata, then emits labels with evidence and confidence. It is
separate from scene discovery because the two stages have different image and token budgets.

## Historical Configs

Root configs for YOLOv8, fixed football A/B/C, tracker grids, and smoke tests remain for result
reproduction. New multi-domain work should start with `adaptive_tracking.yaml` and a benchmark
config under `configs/benchmarks/`.
