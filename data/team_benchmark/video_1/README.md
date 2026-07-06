# video_1 Team Benchmark Draft

This folder is the real-video scaffold for `F:/videos/1.mp4`.

SportsMOT-style MOT files do not contain team labels, so team labels must be
manually annotated before this benchmark can support research claims.

Current verified annotation:

```text
query: the goalkeeper wearing green
track: 19
window: frames 40-320
evidence: outputs/locate_tracking/runs/video_1_locateanything_track19_window
```

Files:

```text
benchmark_manifest_draft.json
  Runnable manifest with the verified track-19 target.

track_annotation_template.csv
  Top 30 longest tracks from F:/videos/1_Tracking_qwen.txt.
  Fill team_label values that are currently __TODO__.

predictions_pipeline_a_qwen_draft.json
  Draft Pipeline A prediction artifact. Replace with real Qwen per-track
  classification output when available.

predictions_pipeline_b_locate_qwen_draft.json
  Pipeline B prediction artifact derived from the resolved LocateAnything run.
```

Run:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.locate_tracking.cli validate-team-benchmark `
  --manifest data\team_benchmark\video_1\benchmark_manifest_draft.json `
  --output outputs\team_benchmark\video_1_draft\validation.json

.\.venv\Scripts\python.exe -m football_tracking.locate_tracking.cli run-team-benchmark `
  --manifest data\team_benchmark\video_1\benchmark_manifest_draft.json `
  --predictions data\team_benchmark\video_1\predictions_pipeline_a_qwen_draft.json `
  --output-dir outputs\team_benchmark\video_1_draft\pipeline_a_qwen `
  --overwrite

.\.venv\Scripts\python.exe -m football_tracking.locate_tracking.cli run-team-benchmark `
  --manifest data\team_benchmark\video_1\benchmark_manifest_draft.json `
  --predictions data\team_benchmark\video_1\predictions_pipeline_b_locate_qwen_draft.json `
  --output-dir outputs\team_benchmark\video_1_draft\pipeline_b_locate_qwen `
  --overwrite

.\.venv\Scripts\python.exe -m football_tracking.locate_tracking.cli compare-team-benchmarks `
  --evaluation outputs\team_benchmark\video_1_draft\pipeline_a_qwen `
  --evaluation outputs\team_benchmark\video_1_draft\pipeline_b_locate_qwen `
  --output-dir outputs\team_benchmark\video_1_draft\comparison `
  --overwrite
```

Next annotation step:

1. Open `track_annotation_template.csv`.
2. For each `__TODO__`, inspect track overlays/crops and set `team_label`.
3. Promote verified rows into `benchmark_manifest_draft.json`.
4. Generate real Pipeline A/Qwen and Pipeline B/Locate prediction manifests.
5. Re-run the commands above.
