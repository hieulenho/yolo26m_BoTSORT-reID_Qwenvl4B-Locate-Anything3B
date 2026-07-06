# Milestone 5 - Uncertainty Detection And Event-Triggered Grounding

Milestone 5 adds a read-only monitor after semantic and appearance resolution. It checks whether the current raw track is still reliable, emits uncertainty events, and prepares a grounding plan for LocateAnything. It does not merge IDs, recover a new ID, write MOT files, modify BoT-SORT, or update appearance prototypes.

## Inputs

- Source video path.
- MOT TXT tracking artifact.
- M3 semantic memory JSON.
- Optional M4 appearance verification JSON.
- Optional M4 fusion result JSON.
- `configs/locate_tracking/uncertainty_monitoring.yaml`.

## Outputs

- `target_observation_timeline.json`: per-frame observations for the current target.
- `uncertainty_assessment.json`: all signals and aggregate uncertainty.
- `uncertainty_events.jsonl`: deduplicated event stream.
- `grounding_plan.json`: selected event-triggered grounding requests.
- Optional `grounding_execution_manifest.json` plus per-frame grounding JSONs when executing a plan.

## Signals

The monitor computes independent signals:

- `TARGET_PRESENCE`: consecutive absence of the current raw track.
- `TRACK_CONFIDENCE`: low MOT confidence only when confidence exists.
- `MOTION_JUMP`: large center displacement from observed boxes.
- `SEMANTIC_MARGIN`: weak semantic/fused winner margin.
- `APPEARANCE_DRIFT`: low M4 appearance score without updating prototypes.
- `NEIGHBOR_AMBIGUITY`: nearby raw tracks that may make the target ambiguous.
- `TRACK_GAP`: internal missing intervals for the same raw track.
- `GROUNDING_STALENESS`: stale semantic grounding context.

Unavailable data is represented as `data_available=false` with an explicit reason, never as a fake zero score.

## CLI

Analyze and plan:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.locate_tracking.cli analyze-target-uncertainty `
  --source-video F:\videos\1.mp4 `
  --tracks path\to\tracks.txt `
  --semantic-memory path\to\semantic_memory.json `
  --appearance-result path\to\appearance_verification.json `
  --fusion-result path\to\fusion_result.json `
  --output-dir outputs\locate_tracking\uncertainty\video_1 `
  --overwrite
```

Execute a plan with the mock backend:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.locate_tracking.cli execute-grounding-plan `
  --plan outputs\locate_tracking\uncertainty\video_1\grounding_plan.json `
  --output-dir outputs\locate_tracking\uncertainty\video_1\grounding_execution `
  --backend mock `
  --overwrite
```

The execution step saves new grounding artifacts only. Reacquisition and ID reassignment are intentionally outside this milestone.
