# Video 1 Team/Position Benchmark

This folder contains the focused benchmark for `F:/videos/1.mp4`.

Core files:

- `benchmark_manifest_expanded.json`: 21 reviewed tracks and 6 language/team queries.
- `track_annotation_expanded.csv`: track-level team and role labels.
- `track_annotation_template.csv`: template for adding more reviewed tracks.
- `pipeline_a_yolo26m_botsort_reid_qwen4b_expanded_bootstrap.json`: Pipeline A bootstrap predictions.
- `pipeline_c_yolo26m_botsort_reid_locateanything3b_qwen4b_expanded_bootstrap.json`: Pipeline C bootstrap predictions.

Pipeline definitions:

- A: YOLO26m + BoT-SORT ReID + Qwen3-VL 4B.
- B: YOLO26m + BoT-SORT ReID + LocateAnything 3B.
- C: YOLO26m + BoT-SORT ReID + LocateAnything 3B + Qwen3-VL 4B.

Current limitation:

- The A/C prediction manifests are bootstrap/contact-sheet artifacts for validating benchmark plumbing.
- A true Pipeline B manifest is still required before reporting a complete A/B/C comparison.
- Position labels are currently coarse (`player`, `goalkeeper`); paper-ready role labels should expand to `defender`, `midfielder`, `forward`, etc.

Run:

```powershell
.\scripts\run_team_position_benchmark.ps1 `
  -Manifest data\team_benchmark\video_1\benchmark_manifest_expanded.json `
  -PipelineA data\team_benchmark\video_1\pipeline_a_yolo26m_botsort_reid_qwen4b_expanded_bootstrap.json `
  -PipelineC data\team_benchmark\video_1\pipeline_c_yolo26m_botsort_reid_locateanything3b_qwen4b_expanded_bootstrap.json `
  -OutputDir outputs\team_benchmark\focused\video_1_available `
  -Overwrite
```
