# Five-Pass Engineering Audit

Date: 2026-07-21

Each pass ended with a focused regression run. The fifth pass reran the complete repository.

## Pass 1 - Detector Routing, Identity, And Class Stability

Reviewed dynamic vocabulary routing, generated tracker routes, per-class adapters, output ID
mapping, and frame-level class labels. The root cause of many class-triggered ID changes was one
tracker instance per class. A temporary `car -> truck` detector error therefore created another
identity.

Regular persistent classes now share one class-agnostic tracker. Only genuinely different motion
regimes, such as a small fast ball, use a separate delegate. A decayed temporal vote stabilizes
the displayed class and requires six observations plus a clear score margin before a correction.
Unit tests cover both a transient class error and a persistently wrong initial class.

## Pass 2 - Long Multi-Domain Inputs And Routing Fallbacks

Replaced short smoke clips with three licensed public videos: wildlife (37.9 s, 908 frames),
traffic (35.0 s, 1,051 frames), and classroom (84.3 s, 2,530 frames). The downloader probes every
file and rejects invalid or sub-30-second inputs.

The classroom run exposed a fallback bug: when Qwen returned only `detect` actions, the router
selected a high-confidence desk instead of people. Fallback promotion now selects persistent
entities such as students, teachers, people, vehicles, and animals before considering furniture.
All three videos now produce non-empty tracks and semantic renders.

## Pass 3 - Qwen Output, Memory, And Unknown Rejection

The original verbose Qwen schema exceeded 768 output tokens on classroom batches. The dynamic
schema now returns only the fields needed for fusion; all 3/3 classroom batches parse and all 8/8
selected tracks receive a model prediction.

Repeated offline runs also reused semantic memory, which duplicated prior evidence. `-Overwrite`
now removes that run's generated memory before fusion. Fine-grained labels use a conservative
0.95 threshold: unsupported bird species and vehicle subtypes remain visible in the audit JSON
but render as `unknown` instead of being presented as facts.

## Pass 4 - Realtime Latency, Scene Cuts, And Identity Stability

The `realtime` profile keeps OC-SORT. A new `realtime_stable` profile keeps the same YOLO26n
detector, 640-pixel input, vocabulary, and video, but uses TrackTrack. Tracker state is now reset
at shot boundaries while the global output-ID counter remains monotonic, preventing identities
from being associated across unrelated camera cuts.

| Traffic profile | E2E FPS | Predicted IDs | Tracks <1 s | Median track | Stable class changes |
|---|---:|---:|---:|---:|---:|
| realtime / OC-SORT | 32.12 | 153 | 64.1% | 11 frames | 25 |
| realtime_stable / TrackTrack | 22.84 | 87 | 31.0% | 58 frames | 20 |

These raw-video values are continuity proxies, not IDSW. The official 30-sequence SportsMOT GT
comparison remains: TrackTrack HOTA 71.058, IDSW 1,042; OC-SORT HOTA 59.379, IDSW 2,186;
BoT-SORT ReID HOTA 68.503, IDSW 895 at 11.66 cached-pipeline FPS.

The live loop now prewarms the detector, writes video asynchronously, bounds the semantic queue,
reports p50/p95/p99 latency, and can drop a late input frame instead of accumulating camera lag.
On the 35-second 30 FPS traffic stream, the no-drop profile processed 27.35 FPS at 45.1 ms p95.
The bounded-latency profile advanced through the source at 29.98 FPS with 44.0 ms p95 while
dropping 11.3% of late frames. Offline accuracy runs never enable this frame dropping.

## Pass 5 - Reports, Runtime Artifacts, And Repository Validation

The multi-domain report now records source duration/resolution, FPS, raw/stable class changes,
semantic coverage, VRAM, median track lifetime, short-track ratio, and within-ID gap events. It
labels all prediction-only continuity metrics explicitly so they cannot be confused with GT IDSW.

The semantic GT workflow now generates three-time contact sheets and one review row for every
predicted track. It covers 395 tracks and 18,814 observations across the three public videos.
Finalization rejects model proposals, draft rows, unnamed reviewers, and duplicate sample IDs;
only human-reviewed manifests can enter semantic accuracy evaluation.

```text
ruff check src scripts tests: PASS
compileall src scripts tests: PASS
pytest: 419 passed
YAML parse: 129/129 passed
PowerShell parser: 11/11 scripts passed
runtime MP4 probe: 4/4 opened, nonblank, and complete
Qwen compact batches: 9/9 parsed across three domains
```

The tested hardware was an NVIDIA GeForce RTX 4060 Laptop GPU with 8 GB VRAM, 16 GB system RAM,
PyTorch 2.11.0+cu128, Ultralytics 8.4.82, and Python 3.12.10.

## Remaining Accuracy Work

1. Complete human review of the prepared 395-track cross-domain annotation package.
2. Add MOT identity GT for the raw traffic/classroom clips before calling continuity proxies IDSW.
3. Calibrate base/fine unknown thresholds by domain instead of treating model confidence as a
   probability.
4. Run a multi-hour webcam/RTSP soak test; the current long-stream result is a 35-second file
   replay using the same bounded-latency loop.
