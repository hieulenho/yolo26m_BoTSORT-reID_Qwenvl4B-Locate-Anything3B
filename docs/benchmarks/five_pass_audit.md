# Five-Pass Engineering Audit

Date: 2026-07-20

This audit records five independent review passes. A pass was accepted only after its discovered
issues were fixed and the relevant checks were rerun.

## Pass 1 - Architecture And Data Contracts

Checked scene discovery schemas, ontology normalization, detector routing, class-ID remapping,
tracker class filtering, and generated YAML.

Found and fixed:

- football supplemental COCO classes were initially remapped to class `0`;
- detection-only classes were filtered out before rendering;
- detection-only boxes could accidentally cross the tracker boundary.

Result: source class IDs are preserved, tracker inputs are partitioned explicitly, and
detection-only objects render without a fake `track_id`.

## Pass 2 - Tracker And Ground-Truth Integrity

Checked the shared detector cache, 30-sequence/20,171-frame contract, TrackEval summaries,
per-sequence rows, tracker config hashes, and IDSW diagnostic partitions.

Result: eight trackers are comparable under the same detector cache. Official ranking uses HOTA,
AssA, IDF1, and TrackEval IDSW. The five IDSW categories are reported separately as diagnostic
heuristics and sum to 100% for each tracker.

## Pass 3 - Realtime Routes And Hardware

Checked football fine-tuned, COCO pretrained, and YOLOE open-vocabulary routes on the same RTX
4060 Laptop GPU. Each route processed and rendered 120 frames.

Found and fixed:

- the first hybrid football implementation ran both detectors every frame and reduced speed;
- caching a sampled ball box caused a visible stale/ghost box between inference frames;
- older route metrics did not record the concrete detector backend/checkpoint.

Result: the realtime football route runs the primary detector every frame and the supplemental
COCO detector every six frames (`20` calls over `120` frames). Stale geometry is never reused.
All three MP4 files probe as `120` frames, `1280x720`, `30 FPS`.

## Pass 4 - Semantic Quality And VRAM

Checked Qwen-only (A), Locate-only (B), and event-verified fusion (C) against the same 31 manually
reviewed tracks. Checked accepted count, coverage, end-to-end accuracy, selective accuracy, Macro
F1, cold wall time, and measured peak VRAM.

Result: Pipeline C has the strongest measured semantic accuracy (`64.52%`) and Macro F1
(`81.87%`). Qwen and LocateAnything execute sequentially, so peak VRAM is the maximum component
peak (`4.46 GiB`), not the sum. The report explicitly limits this conclusion to one football
video; cross-domain accuracy remains pending human GT.

## Pass 5 - Repository And Reproducibility

Checked source formatting, the complete test suite, PowerShell syntax, README links, report source
hashes, generated charts, and local video readability.

Final checks:

```text
ruff check src scripts tests: PASS
pytest: 379 passed
PowerShell parser: 11/11 scripts passed
README local links: 15/15 passed
final artifact audit: PASS with 4 declared scope warnings
runtime MP4 probe: 3/3 opened and complete
```

Cleanup removed two obsolete root archives (about 2.3 GiB), old Ultralytics `runs/` validation
outputs, Python caches, and the superseded hard-coded figure generator. Dataset files, promoted
checkpoints, reviewed GT, shared detector caches, raw tracker predictions, and canonical benchmark
sources were preserved.

## Remaining Work Before A Cross-Domain Accuracy Claim

1. Collect traffic, medical, education, and non-football sports videos with licensing metadata.
2. Annotate domain, object vocabulary, bounding boxes, identities, and semantic labels with a
   second-person review.
3. Run the same frozen router/profile on every domain and report per-domain confidence intervals.
4. Run a long webcam/RTSP soak test; the current realtime figures are 120-frame file-source runs.
