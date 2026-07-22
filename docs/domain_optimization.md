# Domain Optimization Workflow

This project now treats detection and tracking as reusable building blocks:

```text
domain profile -> generated CLI configs -> detector cache -> tracker run -> TrackEval metrics
```

The older configs still work. The domain layer only generates clean wrappers around the existing
commands so each new domain can reuse the same pipeline.

## Priority 1: Domain Profiles

Domain profiles live in `configs/domains/`.

- `football.yaml`: SportsMOT football, player/person class, full MOT evaluation.
- `generic_person.yaml`: generic person video tracking.
- `vehicle.yaml`: generic COCO vehicle classes.

Generate runnable configs from a profile:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.cli build-domain-configs --domain configs\domains\football.yaml --overwrite
```

Generated files are written to `configs/generated/<domain>/`.

## Priority 2: Tracker Presets

Tracker presets live in `configs/trackers/`.

- `botsort_balanced.yaml`: default, good first choice.
- `botsort_high_recall.yaml`: tries to reduce false negatives.
- `botsort_high_identity.yaml`: stricter association for fewer ID switches.
- `bytetrack_fast.yaml`: faster fallback when ReID is too slow.

Generate configs with another preset:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.cli build-domain-configs --domain configs\domains\football.yaml --preset high_recall --overwrite
```

## Priority 3: Detector Fine-Tuning

Each domain profile points to a domain-specific checkpoint path, for example:

```text
models/detector/football/yolo26m_best.pt
models/detector/person/yolo26m_best.pt
models/detector/vehicle/yolo26m_best.pt
```

If that checkpoint does not exist, the current football profile falls back to `yolo26m.pt` for
plumbing and smoke testing. Real accuracy improvements should come from fine-tuned detector weights.

## Priority 4: Measured Tracker Tuning

Generate a BoT-SORT ReID grid-search plan:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.cli plan-tracker-grid --config configs\legacy\football\tracker_grid_botsort_reid.yaml --overwrite
```

This writes:

```text
outputs/experiments/tracker_grid/botsort_reid_recall_identity/manifest.csv
outputs/experiments/tracker_grid/botsort_reid_recall_identity/run_all.ps1
```

Run the planned experiments:

```powershell
.\outputs\experiments\tracker_grid\botsort_reid_recall_identity\run_all.ps1
```

Start with a small `--max-experiments` when iterating.

## Priority 5: Full Football Evaluation

After generating football configs:

```powershell
$env:FOOTBALL_TRACKING_PROGRESS="1"
.\.venv\Scripts\python.exe -m football_tracking.cli compare-trackers --config configs\generated\football\compare_trackers.yaml --overwrite --debug
```

Main outputs:

```text
outputs/metrics/experiments/football_botsort_reid/best_tracker_result.json
outputs/metrics/experiments/football_botsort_reid/tracker_summary.csv
outputs/metrics/experiments/football_botsort_reid/tracker_summary.json
outputs/metrics/experiments/football_botsort_reid/tracker_sequence_metrics.csv
```

Open `best_tracker_result.json` for the selected best tracker and
`tracker_summary.json` for the full configured tracker table.
