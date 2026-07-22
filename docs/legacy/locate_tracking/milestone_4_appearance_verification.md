# Milestone 4: ReID-Aware Appearance Verification

Milestone 4 adds an offline appearance verification layer above the M3 semantic memory.
It consumes existing artifacts and creates new M4 artifacts. It does not modify YOLO,
BoT-SORT, TrackEval, Qwen VLM, MOT TXT output, or the M3 semantic memory input.

## Motivation

Language grounding plus geometry can stay ambiguous in crowded football scenes. Two players
may stand close together, overlap with the same grounded box, or repeatedly appear in the
same tactical region. Appearance evidence helps verify whether candidate raw Track IDs look
internally consistent and support the semantic target hypothesis.

## Architecture

```text
M3 Candidate Semantic Memory
        +
Track Crops from Original Source Video
        ->
Appearance Embedding Provider
        ->
Track Appearance Prototype Bank
        ->
Appearance Verification
        ->
Semantic + Appearance Fusion
        ->
Final Resolution
```

## Separation From BoT-SORT

BoT-SORT ReID is used by the original tracker internally to assign raw IDs. M4 appearance
verification is a separate, auditable, offline layer that verifies semantic candidate IDs.
M4 never rewrites raw IDs, never updates Kalman state, and never enters the BoT-SORT loop.

## Crop Selection

Crops are extracted only from `--source-video`, not annotated videos. Each crop records raw
and clipped coordinates, quality metrics, frame index, and raw Track ID. Invalid or poor
quality crops are rejected. The representative selector ranks by deterministic quality and
enforces a configurable temporal gap before filling remaining slots.

## Prototype Creation

The prototype bank supports:

- `mean`: average normalized sample vectors, then L2 normalize.
- `quality_weighted_mean`: weighted average by crop quality score, then L2 normalize.

All sample dimensions must match. Invalid, NaN, infinite, empty, or zero-norm embeddings are
rejected rather than replaced with zeros.

## Circularity Prevention

Appearance consistency uses leave-one-out scoring. When evaluating sample `e_i`, the
prototype is built from all other samples, so `e_i` is never compared against a prototype
containing itself.

## Similarity

Cosine similarity is implemented in pure utilities. Vectors are validated and L2-normalized
before scoring. Scores are mapped to `[0, 1]` for fusion.

## Fusion

Fusion combines:

- `SemanticScore`: M3 candidate aggregate score.
- `AppearanceScore`: M4 leave-one-out internal consistency score.
- `FusedScore`: weighted score with a configurable missing-evidence policy.

If appearance is unavailable, the default policy is `semantic_only`. A stricter policy can
penalize candidates without appearance evidence.

## Cache

The appearance cache is separate from the grounding cache. Its key includes crop content,
shape, dtype, backend name, model ID, and inference configuration. Changing the model,
preprocessing-relevant config, or crop content causes a cache miss.

## RTX 4060 Recommendation

Use the offline workflow:

1. Run tracking once and keep MOT artifacts.
2. Run M1/M2/M3 on a small deterministic frame sample.
3. Release heavy grounding models if used.
4. Run M4 appearance embedding in small crop batches.
5. Reuse the appearance cache.

Do not keep LocateAnything, YOLO detector, BoT-SORT, Qwen, and appearance embedding models
resident at the same time.

## Commands

Build appearance memory and fusion from an existing M3 artifact:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.locate_tracking.cli build-appearance-memory `
  --source-video F:\videos\1.mp4 `
  --tracks F:\videos\1_Tracking.txt `
  --semantic-memory outputs\locate_tracking\semantic_memory\video_1\semantic_memory.json `
  --output-dir outputs\locate_tracking\appearance\video_1 `
  --backend mock `
  --overwrite
```

The same workflow exposed as verification:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.locate_tracking.cli verify-language-track `
  --source-video F:\videos\1.mp4 `
  --tracks F:\videos\1_Tracking.txt `
  --semantic-memory outputs\locate_tracking\semantic_memory\video_1\semantic_memory.json `
  --output-dir outputs\locate_tracking\appearance\video_1 `
  --backend ultralytics `
  --model-id yolo26n-cls.pt `
  --overwrite
```

## Outputs

- `appearance_manifest.json`
- `appearance_scores.json`
- `fusion_result.json`
- `appearance_summary.md`
- optional clean source-video crops under `crops/`

## Limitations

Milestone 4 does not implement raw Track ID merging, raw ID aliasing, global ReID gallery
search, lost-target recovery, uncertainty states, event-triggered grounding, target
reacquisition, semantic identity timelines, online track correction, or BoT-SORT internal
state modification.
