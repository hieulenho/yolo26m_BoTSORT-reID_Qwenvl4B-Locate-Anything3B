# Final experiment report

Artifact audit: **PASS** with 6 scoped limitation(s).

## Hardware

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- VRAM: 8.00 GiB
- System RAM: 15.69 GiB
- PyTorch: 2.11.0+cu128

## Detector

| Detector | Training | Precision | Recall | mAP50 | mAP50-95 | Detector FPS | E2E FPS |
|---|---|---:|---:|---:|---:|---:|---:|
| YOLO26m fine-tuned | SportsMOT train | 0.9595 | 0.9601 | 0.9793 | 0.8306 | 55.89 | 42.05 |
| YOLO26m pretrained | COCO pretrained | 0.8662 | 0.9026 | 0.8935 | 0.7361 | 53.20 | 40.41 |
| YOLOv8m pretrained | COCO pretrained | 0.8555 | 0.9139 | 0.8932 | 0.7229 | 6.65 | 6.23 |
| YOLO26n pretrained | COCO pretrained | 0.7865 | 0.8377 | 0.8401 | 0.5894 | 58.65 | 41.82 |

## Tracking

| Tracker | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | Tracker FPS | Cached FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TrackTrack | 71.058 | 83.864 | 60.273 | 91.511 | 71.341 | 1042 | 33.20 | 21.66 |
| BoT-SORT ReID (stable) | 68.503 | 80.080 | 58.643 | 88.451 | 71.352 | 895 | 13.67 | 11.66 |
| DeepSORT | 60.096 | 80.131 | 45.163 | 85.896 | 57.530 | 3724 | 20.31 | 17.06 |
| OC-SORT | 59.379 | 72.918 | 48.413 | 87.479 | 66.108 | 2186 | 511.27 | 79.40 |
| FastTracker | 58.702 | 73.450 | 46.985 | 88.129 | 64.325 | 2220 | 272.10 | 51.03 |
| ByteTrack | 58.032 | 72.524 | 46.496 | 87.021 | 64.106 | 1828 | 785.03 | 83.33 |
| Deep OC-SORT ReID | 55.949 | 69.815 | 44.899 | 81.478 | 65.038 | 2016 | 15.52 | 13.41 |
| SORT | 41.216 | 74.128 | 22.984 | 83.428 | 36.440 | 7734 | 321.21 | 122.45 |

Recommended profiles: OC-SORT for realtime, TrackTrack for balanced quality, and BoT-SORT ReID stable when minimizing official ID switches is the priority.

## Semantic A/B/C

| Pipeline | Accuracy | Macro F1 | Coverage | Fine strict | Fine candidate | Unknown F1 | Hallucination | Cold (s) | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 75.00% | 91.67% | 75.00% | 0.00% | 100.00% | 61.54% | 26.67% | 475.44 | 4.89 GiB |
| B | 45.00% | 80.43% | 75.00% | 0.00% | 0.00% | 15.38% | 46.67% | 350.87 | 4.56 GiB |
| C | 60.00% | 88.46% | 90.00% | 0.00% | 100.00% | 20.00% | 38.89% | 826.31 | 4.89 GiB |

## Realtime routes

| Route | Detector | Detections | Detect-only | E2E FPS | Detector FPS | Tracker FPS | Steady FPS | P95 latency | Startup | Peak CUDA |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Football fine-tuned | yolo26m_best.pt + yolo26n.pt | 2703 | 0 | 24.75 | 42.47 | 257.17 | 40.40 | 35.06 ms | 0.62 s | 187.1 MiB |
| COCO pretrained | yolo26n.pt | 2322 | 69 | 34.30 | 58.97 | 373.07 | 46.63 | 25.09 ms | 0.47 s | 63.6 MiB |
| Open vocabulary | yolo26n.pt + yoloe-26s-seg.pt | 2253 | 0 | 19.46 | 30.72 | 217.58 | 29.60 | 47.39 ms | 11.75 s | 165.2 MiB |

Detector scheduling:
- Football fine-tuned: `yolo26m_best.pt` every frame; supplemental = yolo26n.pt: every 6 frame(s), 20 call(s).
- COCO pretrained: `yolo26n.pt` every frame; supplemental = none.
- Open vocabulary: `yolo26n.pt` every frame; supplemental = yoloe-26s-seg.pt: every 6 frame(s), 20 call(s).

Track diagnostics on the same 120-frame source:

| Route | Frames with tracks | Coverage | Unique IDs | Median length | Tracks <30f | Tracks with gaps | Warnings |
|---|---:|---:|---:|---:|---:|---:|---:|
| Football fine-tuned | 119 / 120 | 99.17% | 37 | 61.0 | 32.4% | 12 | 0 |
| COCO pretrained | 119 / 120 | 99.17% | 50 | 21.0 | 54.0% | 25 | 2 |
| Open vocabulary | 119 / 120 | 99.17% | 50 | 21.0 | 54.0% | 25 | 2 |

## Physical webcam realtime

Each profile contains three independent 900-frame runs on the named hardware. Values are mean +/- population standard deviation.

| Profile | Runs | Process FPS | Source FPS | P95 | Drop | Startup |
|---|---:|---:|---:|---:|---:|---:|
| bounded_tracking_only | 3 | 53.41 +/- 2.10 | 30.02 +/- 0.01 | 33.5 +/- 1.2 ms | 0.33 +/- 0.09% | 8.34 +/- 0.41 s |
| bounded_semantic_deferred | 3 | 53.82 +/- 1.95 | 30.01 +/- 0.01 | 33.3 +/- 1.1 ms | 0.33 +/- 0.00% | 8.27 +/- 0.40 s |
| no_drop_semantic_deferred | 3 | 54.23 +/- 1.90 | 30.00 +/- 0.05 | 32.2 +/- 1.4 ms | 0.00 +/- 0.00% | 8.29 +/- 0.38 s |

## IDSW diagnostic taxonomy

Counts below are recomputed diagnostic events. Percentages partition each tracker's recomputed total; they do not replace official TrackEval IDSW.

| Tracker | Recomputed | Fragmentation | Identity swap | ReID failure | Association | Appearance |
|---|---:|---:|---:|---:|---:|---:|
| SORT | 9312 | 2551 (27.4%) | 990 (10.6%) | 545 (5.9%) | 1819 (19.5%) | 3407 (36.6%) |
| DeepSORT | 6097 | 742 (12.2%) | 954 (15.6%) | 475 (7.8%) | 1380 (22.6%) | 2546 (41.8%) |
| ByteTrack | 1971 | 636 (32.3%) | 497 (25.2%) | 570 (28.9%) | 89 (4.5%) | 179 (9.1%) |
| BoT-SORT ReID (stable) | 1070 | 186 (17.4%) | 363 (33.9%) | 486 (45.4%) | 13 (1.2%) | 22 (2.1%) |
| OC-SORT | 2690 | 818 (30.4%) | 678 (25.2%) | 456 (17.0%) | 258 (9.6%) | 480 (17.8%) |
| Deep OC-SORT ReID | 2571 | 504 (19.6%) | 1360 (52.9%) | 250 (9.7%) | 123 (4.8%) | 334 (13.0%) |
| FastTracker | 2413 | 731 (30.3%) | 760 (31.5%) | 565 (23.4%) | 109 (4.5%) | 248 (10.3%) |
| TrackTrack | 1302 | 252 (19.4%) | 380 (29.2%) | 493 (37.9%) | 72 (5.5%) | 105 (8.1%) |

## Scope

- Detector and tracking scores are compared against SportsMOT ground truth.
- Semantic scores use same 20 predicted tracks matched to official UA-DETRAC GT at IoU 0.5.
- Detection-only classes are rendered without a track ID; only `track` classes enter MOT.
- The retained table is UA-DETRAC traffic. A separate official AnimalTrack Zebra A/B/C matrix extends semantic evaluation to wildlife.
- IDSW taxonomy is heuristic; use the official TrackEval IDSW column for ranking.

## Figures

- `..\assets\benchmarks\adaptive_architecture.png`
- `..\assets\benchmarks\realtime_route_fps.png`
- `..\assets\benchmarks\realtime_stage_resources.png`
- `..\assets\benchmarks\tracker_quality_speed.png`
- `..\assets\benchmarks\detector_quality_speed.png`
- `..\assets\benchmarks\semantic_quality_cost.png`
- `..\assets\benchmarks\idsw_taxonomy.png`
