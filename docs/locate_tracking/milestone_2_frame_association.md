# Milestone 2: Single-Frame Grounding-To-Track Association

## Architecture

```text
OLD PIPELINE - UNCHANGED

Video
 ->
YOLO26m
 ->
BoT-SORT ReID
 ->
MOT TXT
```

```text
NEW PARALLEL PIPELINE

Frame
+
Query
 ->
LocateAnything
 ->
GroundingResult

MOT TXT
 ->
Read-only MOT Adapter

GroundingResult
+
Frame Track Observations
 ->
Geometric Association
 ->
Track ID Candidate
```

Milestone 2 consumes existing tracker artifacts as immutable inputs.  The
tracking core, detector core, evaluation code, and Qwen VLM layer are not
modified or called by this subsystem.

## Why Artifact-Level Integration

- The old tracker remains unchanged and continues to write normal MOT files.
- MOT artifacts are read-only inputs for the language-guided branch.
- LocateAnything grounding can be developed and cached independently.
- Association logic is deterministic and testable without GPU, CUDA, internet,
  or model downloads.

## Frame Indexing

Public frame indices and MOT frame indices are 1-based.  OpenCV frame positions
are 0-based.  The conversion happens only inside
`video/frame_extractor.py`:

```text
video_position = frame_index - 1
```

No other module should perform this conversion.

## Geometry Metrics

The matcher serializes transparent geometry metrics:

- `iou`: intersection over union.
- `grounding_coverage`: intersection area divided by grounded-box area.
- `track_coverage`: intersection area divided by track-box area.
- `center_similarity`: `1 - center_distance / frame_diagonal`, clamped to `[0, 1]`.

Candidate scoring is configurable:

```text
score =
  w_iou * iou
  + w_track_coverage * track_coverage
  + w_center * center_similarity
```

Weights are normalized and recorded in every candidate.

## Ambiguity Policy

The matcher does not force top-1 when candidates are too close.  If the top score
and runner-up score differ by less than `ambiguity_margin`, the association is
reported as `ambiguous` with no selected track id.

## Commands

Match an existing grounding artifact:

```powershell
cd F:\Tracking

.\.venv\Scripts\python.exe -m football_tracking.locate_tracking.cli match-grounding-frame `
  --association-config configs\locate_tracking\frame_association.yaml `
  --grounding-result outputs\locate_tracking\grounding\frame_000120.json `
  --tracks F:\videos\1_Tracking.txt `
  --frame-index 120 `
  --frame-width 1920 `
  --frame-height 1080 `
  --output outputs\locate_tracking\queries\frame_000120\association.json `
  --overwrite
```

End-to-end single-frame query:

```powershell
cd F:\Tracking

.\.venv\Scripts\python.exe -m football_tracking.locate_tracking.cli query-track-frame `
  --grounding-config configs\locate_tracking\locateanything_grounding.yaml `
  --association-config configs\locate_tracking\frame_association.yaml `
  --source-video F:\videos\1.mp4 `
  --tracks F:\videos\1_Tracking.txt `
  --frame-index 120 `
  --query "the goalkeeper wearing green" `
  --output-dir outputs\locate_tracking\queries\goalkeeper_green_frame_120 `
  --overwrite
```

For CPU-only smoke tests, set `backend.name: mock` in a temporary grounding
config or pass `--backend mock` with a mock-compatible config.

## Output

Milestone 2 writes only under:

```text
outputs/locate_tracking/
```

It does not write into `outputs/tracks`, `outputs/metrics`, or `outputs/vlm`, and
it does not overwrite MOT files.

## Limitations

- Single frame only.
- No ReID verification.
- No appearance embeddings.
- No semantic memory.
- No reacquisition.
- No cross-frame voting.
- No uncertainty state machine.
- No raw ID remapping.
- No video-wide LocateAnything inference.

