# Semantic benchmark ground truth

Copy `manifest.template.yaml`, then replace every placeholder after manually
reviewing the source video, discovery keyframes, and representative track crops.

`domain` is one human-assigned scene category. `objects` is the complete visible
class vocabulary relevant to the task, with `action` set to `track`, `detect`, or
`context`. `tracks` maps actual MOT IDs from the evaluated run to human semantic
labels. Set `ignore: true` only when a track cannot be judged from the video.

Run:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_adaptive_semantics.py `
  --manifest data\semantic_benchmark\manifest.yaml `
  --output-dir outputs\benchmarks\semantics\final `
  --overwrite
```

The evaluator refuses missing artifacts, duplicate sample IDs, duplicate track
IDs, and samples without GT. It never treats render coverage as model accuracy.
