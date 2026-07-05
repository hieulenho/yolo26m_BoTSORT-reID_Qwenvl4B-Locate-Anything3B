# Config Guide

Configs are intentionally kept as small YAML entry points. Prefer adding a new config over
editing an existing experiment once results have been recorded.

## Recommended Current Configs

| Purpose | Config |
|---|---|
| Prepare SportsMOT football data | `sportsmot_data.yaml` |
| Train active football detector | `yolo26m_sportsmot_football_train.yaml` |
| Evaluate active football detector | `yolo26m_sportsmot_football_eval.yaml` |
| Cache detections for all SportsMOT football sequences | `detection_cache_yolo26m_all.yaml` |
| Track one video with YOLO26m + BoT-SORT ReID | `track_video_yolo26m_botsort.yaml` |
| Compare stable BoT-SORT ReID on all sequences | `compare_trackers_yolo26m_botsort_identity_stable_all.yaml` |
| Qwen VLM analysis after tracking | `vlm_qwen4b_tracking.yaml` |

## Tracker Presets

Tracker presets live in `configs/trackers/`.

| Preset | Use |
|---|---|
| `botsort_reid_identity_stable.yaml` | Fewer ID switches; preferred when ID stability matters. |
| `botsort_balanced.yaml` | Balanced recall/identity baseline. |
| `botsort_high_recall.yaml` | More permissive tracking when missed objects matter more. |
| `botsort_high_identity.yaml` | Stricter association for crowded scenes. |
| `bytetrack_fast.yaml` | Faster tracker baseline without ReID. |

The root-level `botsort_reid.yaml` is the default video-tracking preset used by
`track_video_yolo26m_botsort.yaml`.

## Legacy/Baseline Configs

Files with `yolov8m` are kept for historical baselines and regression tests. New football runs
should start from the `yolo26m_*` configs unless you intentionally want to compare old baselines.

## Generated Configs

`configs/generated/` is created by `build-domain-configs` and is ignored by Git. Regenerate these
configs from `configs/domains/*.yaml` instead of editing generated files by hand.
