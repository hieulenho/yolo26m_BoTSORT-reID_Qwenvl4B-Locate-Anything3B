# Dense Traffic Live Semantic Benchmark

## Scope

- Hardware: NVIDIA GeForce RTX 4060 Laptop GPU, 8,188 MiB VRAM
- Driver: 596.08
- Source: UA-DETRAC `MVI_40774`, 25 FPS
- Detector/tracker task: `configs/tasks/traffic_objects.yaml`
- Semantic model: `Qwen/Qwen3-VL-4B-Instruct`, 8-bit
- Evidence: one 448 x 336 panel per semantic job

## Foreground and Queue Cost

The first comparison processes the same 300 frames without running Qwen. It isolates the
cost of queue bookkeeping, crop selection, evidence panels, rendering, and video writing.

| Profile | End-to-end FPS | Processing FPS | P95 latency | Queue ms/frame | Enqueued | Pending |
|---|---:|---:|---:|---:|---:|---:|
| Tracking only | 28.518 | 29.496 | 42.89 ms | 0.001 | 0 | 0 |
| Queue capacity 12 | 24.777 | 25.548 | 50.25 ms | 0.974 | 14 | 12 |
| Queue capacity 4 | 25.395 | 26.272 | 45.22 ms | 0.389 | 4 | 4 |

The capacity-four profile keeps up with the 25 FPS source and is now the live default.
Backpressure is applied before crop creation, and tracks already being processed count toward
the same bound.

## Concurrent Qwen Result

The second run processes all 750 frames with a live Qwen worker and queue capacity four.

| Metric | Result |
|---|---:|
| End-to-end FPS | 19.892 |
| Processing FPS | 20.596 |
| P95 frame latency | 80.56 ms |
| Keeps up with 25 FPS source | No |
| Semantic jobs completed during run | 1 |
| Semantic jobs left pending | 3 |
| Qwen cold load | 25.775 s |
| Qwen inference for one track | 19.856 s |
| Qwen peak reserved VRAM | 5.234 GB |

The worker exited cleanly after the foreground loop, removed its PID file, and released GPU
memory. The completed track was accepted as `car`; Qwen also returned visible color and motion
attributes. A separate fine-label smoke requested an open vehicle subtype but correctly kept
the subtype unknown because the distant evidence did not support a more specific label.

## Interpretation

`waiting` means a track has only its immediate detector label and has not entered the bounded
semantic queue. `pending` means its evidence is queued or currently inside Qwen. Neither state
blocks detection or tracking.

On one 8 GB GPU, a 4B VLM cannot deeply classify every short-lived road user at camera rate.
YOLO and Qwen compete for the same GPU while a semantic job is running. Use:

- `disabled` for the highest live FPS with immediate detector classes;
- `live` for sparse, delayed deep labels on persistent tracks;
- `deferred` for recorded video when all retained tracks should be processed before the final
  semantic render.

Raw artifacts:

- `outputs/benchmarks/dense_traffic_runtime/traffic_foreground_300/`
- `outputs/benchmarks/dense_traffic_runtime/traffic_queue_300/`
- `outputs/benchmarks/dense_traffic_runtime/traffic_queue_cap4_300/`
- `outputs/benchmarks/dense_traffic_runtime/traffic_live_qwen_cap4_750/`
- `outputs/benchmarks/dense_traffic_runtime/traffic_fine_label_smoke/`
