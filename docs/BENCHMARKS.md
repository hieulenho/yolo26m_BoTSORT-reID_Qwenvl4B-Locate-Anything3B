# Traffic evaluation

## Published snapshot

The [traffic quality report](benchmarks/traffic_quality/traffic_quality_report.md)
contains a 750-frame UA-DETRAC comparison, its hardware/protocol, source hashes,
CSV tables and plots. Raw videos, weights, private annotations and full run logs
are not redistributed. The saved results belong to the measured profiles, not
automatically to every configuration in this repository.

Fine vehicle subtype and color GT are separate from parent-class GT. The reported
eight GT-matched semantic tracks are a small sample, not a general accuracy claim.
An `unknown` or `pending` output is not a correct deep-label prediction by default.

## Reproduce on authorized data

1. Install evaluation dependencies with `python -m pip install -r requirements/base.txt`
   using the project virtual environment.
2. Obtain the source frames, official annotations and ignore regions legally.
   Normalize them with `scripts/benchmarks/convert_multidomain_gt.py` (see `--help`).
   Use the UA-DETRAC adapter and preserve the frame numbering and resolution.
3. Create a normalized manifest with accessible video paths. The runner expects
   `sequences` entries containing `normalized_sequence` (or `sequence`),
   `media_path` and `frame_count`. Its GT folder must follow the converter's layout.
4. Run the same frames and detector settings for each tracker:

```powershell
.\scripts\benchmarks\run_task_gt_benchmark.ps1 `
  -TaskConfig configs\tasks\traffic_vehicles_benchmark.yaml `
  -GtManifest "data\benchmarks\normalized\traffic\normalized_gt_manifest.json" `
  -RunName traffic_tracktrack `
  -Device cuda
```

Use `-TrackerConfig` to compare `configs/trackers/bytetrack_fast.yaml`,
`ocsort_realtime.yaml`, or the ReID profiles. Do not silently reuse predictions
from another detector, source resolution, threshold or frame subset.
The runner disables semantic inference so tracker timing remains interpretable.

For semantic evaluation, match predicted tracks to independent reviewed GT with
`build_semantic_gt_from_mot.py`, then use `evaluate_adaptive_semantics.py`.
Inspect each command's `--help`; do not promote model-generated drafts to GT.

## Report separately

- Detection: precision, recall and mAP on annotated frames.
- Tracking: HOTA, MOTA, IDF1, total IDSW, FP and FN using the same GT/ignore protocol.
- Semantics: coverage, parent/subtype/color accuracy and Macro-F1 with explicit
  sample counts; unknown rejection only when known/unknown GT exists.
- Live performance: captured and processed FPS, frame drops, latency percentiles,
  semantic queue age and Qwen inference latency, including cold load separately.
- Hardware: CPU, GPU/VRAM, RAM, driver/CUDA, library versions, image size and precision.

The legacy five-way IDSW breakdown is a heuristic diagnostic. It is not the
official TrackEval total or a validated causal classification of failures.

`build_traffic_quality_report.py --config configs/benchmarks/traffic_quality_report.yaml`
rebuilds the published-style report only when its named source artifacts exist.
For new runs, point a copied contract at those runs; missing files are not scores of zero.
