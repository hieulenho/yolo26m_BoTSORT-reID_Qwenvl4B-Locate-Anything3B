# YOLO26m Football Fine-Tuning

Priority 1 is to create a domain-specific detector checkpoint:

```text
models/detector/football/yolo26m_best.pt
```

The tracking configs already look for that checkpoint first. If it is missing, they fall back to
`yolo26m.pt`, which is useful for plumbing but not the final accuracy target.

## Configs

Training:

```text
configs/legacy/football/yolo26m_sportsmot_football_train.yaml
```

Evaluation:

```text
configs/legacy/football/yolo26m_sportsmot_football_eval.yaml
```

The training config uses:

```text
data/yolo/sportsmot_football/dataset.yaml
outputs/training/football/yolo26m
models/detector/football/yolo26m_best.pt
outputs/metrics/football/yolo26m
```

## Preflight

```powershell
.\.venv\Scripts\python.exe -m football_tracking.cli preflight-training --config configs\legacy\football\yolo26m_sportsmot_football_train.yaml
```

The current preflight result is clean:

```text
train images: 8213
val images: 2900
objects: 145228
errors: 0
warnings: 0
```

## Train

Default full training:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.cli train-detector --config configs\legacy\football\yolo26m_sportsmot_football_train.yaml
```

Or use the wrapper:

```powershell
.\scripts\train_football_detector.ps1
```

If GPU memory is tight on 8 GB VRAM, use a smaller batch:

```powershell
.\scripts\train_football_detector.ps1 -Batch 1
```

For a shorter first real run:

```powershell
.\scripts\train_football_detector.ps1 -Epochs 30 -Batch 1
```

## Resume

```powershell
.\.venv\Scripts\python.exe -m football_tracking.cli resume-detector --config configs\legacy\football\yolo26m_sportsmot_football_train.yaml --checkpoint outputs\training\football\yolo26m\weights\last.pt
```

## Evaluate

After training creates `models/detector/football/yolo26m_best.pt`:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.cli evaluate-detector --config configs\legacy\football\yolo26m_sportsmot_football_eval.yaml --overwrite
```

Metrics are written under:

```text
outputs/metrics/football/yolo26m
```

## Re-run Tracking With The Fine-Tuned Detector

Regenerate or reuse the football domain configs:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.cli build-domain-configs --domain configs\domains\football.yaml --overwrite
```

Then run tracking/evaluation:

```powershell
$env:FOOTBALL_TRACKING_PROGRESS="1"
.\.venv\Scripts\python.exe -m football_tracking.cli compare-trackers --config configs\generated\football\compare_trackers.yaml --overwrite --debug
```
