# Qwen4B VLM Layer

This project keeps detection/tracking deterministic and uses the VLM as a reasoning layer.

```text
video
  -> YOLO detector
  -> BoT-SORT ReID tracker
  -> MOT tracks + annotated video
  -> keyframes/crops/context JSON
  -> Qwen VLM report
```

The VLM should not replace YOLO or the tracker. It consumes clean tracking artifacts and answers
questions about events, behavior, visible identities, and possible tracking mistakes.

## Install Optional VLM Dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements\vlm.txt
```

The base tracking pipeline does not require these packages. They are imported only when
`--run-model` is enabled.

## One Command: Track Then Prepare Qwen Context

This tracks the video, creates keyframes/crops/context, and stops before model inference:

```powershell
.\scripts\track_video_qwen_vlm.ps1 `
  -Source F:\videos\1.mp4 `
  -OutputVideo F:\videos\1_Tracking.mp4 `
  -Overwrite
```

To run Qwen after context preparation:

```powershell
.\scripts\track_video_qwen_vlm.ps1 `
  -Source F:\videos\1.mp4 `
  -OutputVideo F:\videos\1_Tracking.mp4 `
  -RunModel `
  -Overwrite
```

## Analyze Existing Tracking Output

If tracking has already been generated:

```powershell
.\scripts\analyze_tracking_vlm.ps1 `
  -SourceVideo F:\videos\1.mp4 `
  -TrackedVideo F:\videos\1_Tracking.mp4 `
  -Tracks F:\videos\1_Tracking.txt `
  -Metadata F:\videos\1_Tracking.metadata.json `
  -OutputDir F:\videos\1_vlm `
  -Overwrite
```

Add `-RunModel` to execute Qwen.

The runner tries the local Hugging Face cache first. If the model was already downloaded, the
pipeline can run without another Hub request. If local files are incomplete, Transformers may still
need network access to fetch missing config files.

## Main Outputs

```text
<output-dir>/
  vlm_context.json
  prompt.md
  keyframes/
  crops/
  vlm_answer.md
  vlm_answer.json
```

`vlm_context.json` is the stable artifact for downstream systems. It contains video metadata,
track summaries, tracking diagnostics, keyframe paths, crop paths, and tracking metadata.

The `tracking_diagnostics` block is generated from MOT metadata and highlights:

```text
stable_long_tracks
largest_displacement_tracks
fragmented_tracks
low_confidence_tracks
short_tracks
selected_track_ids_visible_in_keyframes
```

These are heuristic hints for Qwen and for manual review. They are not a replacement for TrackEval
metrics.
The prompt sent to Qwen uses a compact version of these diagnostics to reduce VRAM pressure; the
full `vlm_context.json` stays available for inspection.

## Model Choice

The default model id is:

```text
Qwen/Qwen3-VL-4B-Instruct
```

You can override it:

```powershell
.\scripts\analyze_tracking_vlm.ps1 `
  -SourceVideo F:\videos\1.mp4 `
  -TrackedVideo F:\videos\1_Tracking.mp4 `
  -ModelId Qwen/Qwen3-VL-4B-Instruct `
  -RunModel `
  -Overwrite
```

For an 8GB laptop GPU, start with context preparation first. Then run the model with fewer
keyframes if VRAM is tight. The current stable local setting is usually:

```powershell
.\scripts\analyze_tracking_vlm.ps1 `
  -SourceVideo F:\videos\1.mp4 `
  -TrackedVideo F:\videos\1_Tracking_qwen.mp4 `
  -Tracks F:\videos\1_Tracking_qwen.txt `
  -Metadata F:\videos\1_Tracking_qwen.metadata.json `
  -OutputDir F:\videos\1_vlm_tracking_report `
  -RunModel `
  -TorchDtype float16 `
  -MaxKeyframes 2 `
  -MaxTracks 10 `
  -MaxCropsPerTrack 1 `
  -MaxNewTokens 768 `
  -Overwrite
```

On the RTX 4060 Laptop 8GB setup, 2 keyframes is stable while 3 keyframes can run out of VRAM.
Increase `MaxNewTokens` before increasing `MaxKeyframes`.

## Smoke Check

To verify Qwen without overwriting your video-side VLM folder:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.cli analyze-tracking-vlm `
  --config configs\vlm_qwen4b_tracking.yaml `
  --source-video F:\videos\1.mp4 `
  --tracked-video F:\videos\1_Tracking_vlm.mp4 `
  --tracks F:\videos\1_Tracking_vlm.txt `
  --metadata F:\videos\1_Tracking_vlm.metadata.json `
  --output-dir outputs\vlm\qwen4b\smoke_check `
  --max-keyframes 1 `
  --max-tracks 5 `
  --max-crops-per-track 1 `
  --max-new-tokens 64 `
  --run-model `
  --overwrite
```

Expected result: `model_result.status` is `ok` and `vlm_answer.md` is written.
