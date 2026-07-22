# Three-pass completion audit

Audit date: 2026-07-22

## Scope

This audit covers the adaptive multi-domain path, realtime semantic queue, tracker benchmark,
semantic ground-truth workflow, ID-switch diagnostics, configuration, and user-facing commands.
It does not turn draft model proposals into ground truth.

## Requirement coverage

| Stage | Implementation | Audit status |
|---|---|---|
| 1. Shot segmentation | `adaptive_tracking/shot_sampling.py` and online scene-cut reset | Implemented and tested |
| 2. Global keyframes | bounded per-shot keyframe selection | Implemented and tested |
| 3. Qwen scene discovery | dynamic domain, objects, actions, and semantic facets | Implemented; real-model runs remain hardware-dependent |
| 4. Vocabulary normalization | ontology aliases, attributes, COCO mapping, class limit | Implemented and tested |
| 5. Detector router | football fine-tuned YOLO, COCO YOLO, unknown-class YOLOE | Implemented and tested |
| 6. Per-frame detection | routed primary and supplemental detectors | Implemented and benchmarked |
| 7. Tracking | OC-SORT realtime, TrackTrack stable realtime, BoT-SORT ReID accuracy | Implemented and benchmarked |
| 8. Track metadata | MOT text, runtime metadata, route provenance, diagnostics | Implemented and validated |
| 9. Multi-time crops | bounded track sampling plus realtime event crops | Implemented and tested |
| 10. Deep Qwen labels | open hierarchical base/fine labels | Implemented with persistent batch session |
| 11. Temporal fusion | bounded memory, confidence margin, unknown rejection | Implemented and tested |
| 12. Render and metrics | MP4, JSON, MOT, TrackEval, runtime CSV, figures | Implemented and smoke-validated |

## Pass 1: component and experiment contracts

- Targeted tests: 57 passed.
- Ruff, PowerShell parser, and environment doctor passed.
- A clean 300-frame SportsMOT smoke benchmark ran eight trackers against GT.
- TrackEval outputs, 11 comparison figures, and five-category diagnostic IDSW outputs were created.
- A persistent Qwen session and a bounded realtime queue worker were added so one worker process
  can serve multiple batches without reloading the model for every batch.

## Pass 2: full repository validation

- Full suite: 421 passed.
- Python compile check passed.
- All 117 YAML files parsed successfully.
- Every PowerShell script parsed successfully.
- `git diff --check` passed.

## Pass 3: release verification

- Full suite rerun after fixes: 421 passed.
- Environment doctor: 15 checks passed, 0 warnings, 0 failures.
- Hardware detected: NVIDIA GeForce RTX 4060 Laptop GPU, CUDA 12.8, 8 GiB VRAM.
- Smoke artifacts validated: 8 tracker summaries, 11 non-empty figures, 8 IDSW summaries.
- Realtime `live` mode now runs until the stream stop marker; the 64-event safety bound applies
  only to the default deferred worker on a shared 8 GiB GPU.

## Measured smoke result

This result is a plumbing and regression check on one 300-frame sequence, not a replacement for
the retained 30-sequence benchmark.

| Tracker | HOTA | IDF1 | Official IDSW | Cached pipeline FPS |
|---|---:|---:|---:|---:|
| SORT | 65.526 | 67.131 | 29 | 172.807 |
| DeepSORT | 86.942 | 90.688 | 8 | 6.278 |
| ByteTrack | 76.292 | 90.660 | 4 | 101.676 |
| BoT-SORT ReID | 85.875 | 92.440 | 3 | 11.785 |
| OC-SORT | 79.439 | 95.843 | 1 | 87.487 |
| Deep OC-SORT ReID | 78.546 | 94.440 | 2 | 12.632 |
| FastTrack | 76.585 | 90.077 | 5 | 172.358 |
| TrackTrack | 89.177 | 93.587 | 3 | 48.315 |

## Honest completion boundary

- The cross-domain review packages contain 395 candidate tracks: 36 wildlife, 153 traffic, and
  206 classroom. None is human-reviewed yet, so cross-domain semantic accuracy is not reportable.
- The smoke IDSW taxonomy contains 52 heuristic events. Human review coverage is 0%; use official
  TrackEval IDSW for ranking until the event review sheet is completed.
- Webcam/RTSP latency still depends on the capture device and network. The retained long-stream
  benchmark is file replay and must not be described as a physical-camera measurement.
- On one 8 GiB GPU, detector tracking and Qwen/LocateAnything should run sequentially. The default
  realtime mode therefore defers semantics; `live` is intended for a second GPU or server.

## Current artifacts

- Smoke tracker metrics: `outputs/benchmarks/tracking/sportsmot_yolo26m/smoke/metrics/`
- Smoke IDSW diagnostics: `outputs/benchmarks/tracking/sportsmot_yolo26m/smoke/idsw_taxonomy/`
- Semantic GT progress: `outputs/reports/semantic_gt_status.json`
- Canonical retained report: `docs/benchmarks/final_experiment_report.md`

