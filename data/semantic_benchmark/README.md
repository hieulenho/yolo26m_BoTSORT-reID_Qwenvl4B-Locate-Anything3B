# Semantic benchmark ground truth

Copy `manifest.template.yaml`, then replace every placeholder after manually
reviewing the source video, discovery keyframes, and representative track crops.

`domain` is one human-assigned scene category. `objects` is the complete visible
class vocabulary relevant to the task, with `action` set to `track`, `detect`, or
`context`. `tracks` maps actual MOT IDs from the evaluated run to human semantic
labels. Set `ignore: true` only when a track cannot be judged from the video.
`detector_route` is the expected routing family: `football_finetuned`,
`coco_pretrained`, `open_vocabulary`, or `coco_open_composite`.

For the requested cross-domain study, start from
`multidomain_manifest.template.yaml` and replace all four sample placeholders.
Do not score a domain until a human has reviewed every listed object class and
track label. A model-generated label is a prediction, not ground truth.

Prepare one review package per video with `scripts/prepare_semantic_gt.py prepare`. After a
reviewer has completed every CSV row and `ground_truth_review.yaml`, run `finalize`. Merge the
reviewed video manifests into one benchmark without editing YAML by hand:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_semantic_gt.py merge `
  --manifest data\semantic_benchmark\review\wildlife_black_noddies\manifest.reviewed.yaml `
  --manifest data\semantic_benchmark\review\traffic_street\manifest.reviewed.yaml `
  --manifest data\semantic_benchmark\review\education_classroom_long\manifest.reviewed.yaml `
  --output-manifest data\semantic_benchmark\multidomain.reviewed.yaml `
  --overwrite
```

Run:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_adaptive_semantics.py `
  --manifest data\semantic_benchmark\manifest.yaml `
  --output-dir outputs\benchmarks\semantics\final `
  --overwrite
```

The evaluator refuses missing artifacts, duplicate sample IDs, duplicate track
IDs, and samples without GT. It never treats render coverage as model accuracy.
