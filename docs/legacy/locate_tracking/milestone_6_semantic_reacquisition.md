# Milestone 6 - Semantic Target Reacquisition And Identity Continuity

Milestone 6 adds a semantic identity layer after the existing detector, tracker,
LocateAnything grounding, semantic memory, appearance verification, and uncertainty
monitoring steps. It does not rewrite MOT TXT files, BoT-SORT IDs, detector outputs,
grounding outputs, semantic memory, or appearance prototypes.

The core idea is to keep two separate ID spaces:

- Raw tracker IDs: emitted by BoT-SORT/ReID and treated as read-only.
- Semantic target IDs: stable language-level identities such as `target_player_blue`.

A semantic target can point to one raw track segment at first, then later point to a
different raw track segment if the evidence is strong enough.

## Inputs

- `semantic_target.json`: stable target state and raw-ID segment history.
- MOT TXT tracking artifact from the tracker/video command.
- `uncertainty_events.jsonl` from Milestone 5.
- Optional `grounding_execution_manifest.json` from Milestone 5.
- Optional `appearance_scores.json` from Milestone 4.
- `configs/locate_tracking/reacquisition.yaml`.

## Outputs

- `reacquisition_result.json`: candidate search window, hard gates, scores, decision.
- `summary.md`: compact readable summary.
- `semantic_target.json`: committed semantic identity state, only when `--commit` is used.
- `semantic_target_timeline.json`: readable target-to-raw-track segment timeline.
- `identity_transitions.jsonl`: append-only transition log.

## Decision Flow

1. Select a `target_absent` or `track_gap` uncertainty event.
2. Build a bounded search window around the loss event.
3. Check whether the same raw ID resumes after the event.
4. If not, generate candidate raw tracks in the window.
5. Apply hard gates before ranking:
   - minimum observations
   - no temporal conflict with the previous target context
   - plausible motion distance
   - grounding support when configured
   - no semantic identity history conflict
6. Rank passed candidates using available evidence:
   - grounding compatibility
   - frozen appearance verification score
   - motion continuity
   - temporal proximity
   - identity history consistency
7. Return one of:
   - `same_raw_id_resumed`
   - `provisional`
   - `ambiguous`
   - `rejected`
   - `not_found`

`provisional` means a new raw ID is plausible but should remain under probation until
enough frames support it. Confirmation changes the semantic target back to `ACTIVE`.

## CLI Flow

Set the MOT TXT path from your actual tracking run. For example, if your one-command
video run saved a tracking txt next to `F:\videos\1.mp4`, set it here:

```powershell
$tracks = "F:\videos\1_Tracking_qwen.txt"
```

Create a semantic target. The example below says raw track `7` was the target and was
last trusted at frame `361`, just before the first mock loss event at frames 362-366.

```powershell
.\.venv\Scripts\python.exe -m football_tracking.locate_tracking.cli init-semantic-target `
  --query "the player in blue" `
  --raw-track-id 7 `
  --start-frame 1 `
  --last-confirmed-frame 361 `
  --semantic-target-id target_player_blue `
  --output outputs\locate_tracking\runs\video_1_reacquisition\target_player_blue\semantic_target.json `
  --overwrite
```

Run a no-commit candidate search first. This is the safest inspection step because it
does not mutate `semantic_target.json`.

```powershell
.\.venv\Scripts\python.exe -m football_tracking.locate_tracking.cli search-reacquisition-candidates `
  --config configs\locate_tracking\reacquisition.yaml `
  --semantic-target outputs\locate_tracking\runs\video_1_reacquisition\target_player_blue\semantic_target.json `
  --tracks $tracks `
  --events outputs\locate_tracking\runs\video_1_uncertainty_mock\uncertainty_events.jsonl `
  --grounding-manifest outputs\locate_tracking\runs\video_1_grounding_execution_mock\grounding_execution_manifest.json `
  --appearance-result outputs\locate_tracking\runs\video_1_appearance_mock\appearance_scores.json `
  --event-id event_1ecbb13bb025d140 `
  --output-dir outputs\locate_tracking\runs\video_1_reacquisition\target_player_blue\search `
  --overwrite
```

Commit only if the decision is good enough. A `provisional` decision opens a probation
segment for the semantic target while preserving the raw MOT TXT unchanged.

```powershell
.\.venv\Scripts\python.exe -m football_tracking.locate_tracking.cli reacquire-language-target `
  --config configs\locate_tracking\reacquisition.yaml `
  --semantic-target outputs\locate_tracking\runs\video_1_reacquisition\target_player_blue\semantic_target.json `
  --tracks $tracks `
  --events outputs\locate_tracking\runs\video_1_uncertainty_mock\uncertainty_events.jsonl `
  --grounding-manifest outputs\locate_tracking\runs\video_1_grounding_execution_mock\grounding_execution_manifest.json `
  --appearance-result outputs\locate_tracking\runs\video_1_appearance_mock\appearance_scores.json `
  --event-id event_1ecbb13bb025d140 `
  --output-dir outputs\locate_tracking\runs\video_1_reacquisition\target_player_blue\commit `
  --commit `
  --overwrite
```

Confirm probation after enough frames exist for the selected raw ID.

```powershell
.\.venv\Scripts\python.exe -m football_tracking.locate_tracking.cli confirm-reacquisition `
  --config configs\locate_tracking\reacquisition.yaml `
  --semantic-target outputs\locate_tracking\runs\video_1_reacquisition\target_player_blue\semantic_target.json `
  --tracks $tracks `
  --decision outputs\locate_tracking\runs\video_1_reacquisition\target_player_blue\commit\reacquisition_result.json `
  --output-dir outputs\locate_tracking\runs\video_1_reacquisition\target_player_blue\confirm `
  --overwrite
```

Render a semantic-target overlay if you want a video view of the target timeline.

```powershell
.\.venv\Scripts\python.exe -m football_tracking.locate_tracking.cli render-semantic-target `
  --source-video F:\videos\1.mp4 `
  --tracks $tracks `
  --semantic-target outputs\locate_tracking\runs\video_1_reacquisition\target_player_blue\semantic_target.json `
  --output outputs\locate_tracking\runs\video_1_reacquisition\target_player_blue\semantic_target_overlay.mp4
```

## Safety Rules

- Raw MOT rows are read-only and never rewritten.
- M4 appearance prototypes are frozen during ranking and probation.
- M5 grounding artifacts are read-only evidence.
- `search-reacquisition-candidates` is no-commit by default.
- `reacquire-language-target --commit` changes only semantic identity artifacts.
- Ambiguous candidates are not committed.

## Interpreting Results

Open `reacquisition_result.json` first. The most important fields are:

- `decision.status`
- `decision.selected_raw_track_id`
- `decision.final_score`
- `decision.score_margin`
- `candidates[].gate_results`
- `candidates[].component_scores`

Then open `semantic_target_timeline.json` to see the semantic target history across
raw IDs. This is the artifact to inspect when you want to know whether one language
target stayed continuous across ID switches.

## A-L Implementation Notes

### A. Motivation

Raw tracker identity is an algorithm-owned local ID. Stable semantic identity is a
user/query-owned target concept. Milestone 6 links them without rewriting the tracker.

### B. Architecture

The layer consumes M5 loss events, M5 grounding artifacts, MOT candidate tracks, frozen
appearance evidence, and motion history. It then performs gating, evidence scoring,
ranking, probation, and semantic identity transition logging.

### C. Raw Track ID vs Semantic Target ID

`raw_track_id` stays owned by BoT-SORT. `semantic_target_id` stays owned by
`locate_tracking.identity`. The mapping is stored as ordered semantic identity segments.

### D. Candidate Generation

Candidates are raw tracks observed inside a bounded `CandidateSearchWindow`. The same
raw ID resume path is checked before new-ID reacquisition, and its resume frame is the
first observation after the loss event.

### E. Hard Gating

The current gates are:

- temporal gate: rejects candidates present during the previous target context;
- minimum observation gate: rejects short-lived candidates;
- motion gate: rejects implausible center displacement;
- spatial grounding gate: requires grounding compatibility when configured;
- identity conflict gate: rejects candidates overlapping existing confirmed segments.

### F. Evidence

- `S_ground`: weighted geometry from LocateAnything box overlap, candidate coverage,
  and center similarity.
- `S_appearance`: M4 appearance score for the candidate raw ID.
- `S_motion`: normalized continuity from previous target center to candidate first center.
- `S_temporal`: proximity to the last confirmed target frame inside the search window.
- `S_history`: 1.0 when no semantic identity conflict exists, 0.0 when conflict exists.

### G. Score Fusion

The final score is a weighted average:

```text
S_final = sum(weight_i * S_i) / sum(available_weight_i)
```

If `missing_evidence_policy` is `ignore`, unavailable evidence is removed from the
denominator. If it is `zero`, unavailable evidence contributes weight with zero score.
Candidates are sorted deterministically by final score, first observed frame, then raw ID.

### H. Probation

A provisional winner creates a probation segment. Confirmation requires at least
`probation_min_observations` inside `probation_window_frames`. Confirmation changes the
semantic target back to `ACTIVE`; failure keeps the target from being silently accepted.

### I. Prototype Safety

Appearance references are read-only during ranking and probation. M6 does not update
prototype vectors or candidate embeddings.

### J. Identity Timeline

`semantic_target_timeline.json` is the readable audit trail:

```text
target_player_blue:
raw 7 -> raw 42
```

The MOT TXT still contains raw `7` and raw `42` exactly as originally produced.

### K. Failure Modes

Expected non-commit outcomes are `not_found`, `rejected`, and `ambiguous`. Typical causes
are no candidate, grounding mismatch, appearance confusion, motion inconsistency, identity
history conflict, or probation failure.

### L. RTX 4060 Workflow

Use artifact-based sequential processing:

- run tracking once and reuse MOT TXT;
- reuse M1 grounding cache and M5 execution manifests;
- reuse M4 appearance scores;
- process only candidate tracks in the bounded window;
- keep ranking and identity state CPU-side.

No performance claim is made here. Milestone 6 is runtime identity continuity, not the
final benchmark.

## Milestone Boundary

Not implemented in Milestone 6:

- final benchmark suite;
- SportsMOT language-query benchmark;
- final ablation experiments;
- automatic threshold optimization;
- learning-based candidate fusion;
- cross-camera identity recovery;
- global full-video gallery search by default;
- multi-target interaction optimization;
- research report generation;
- publication claims.
