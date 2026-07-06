# Milestone 3: Multi-Frame Semantic Track Memory

Milestone 3 adds a semantic memory layer above the existing LocateAnything grounding
and single-frame track association pipeline.

## Architecture

The pipeline is intentionally separated into three layers:

1. `grounding`: LocateAnything or mock backend grounds a language query on one image.
2. `association`: one grounded box is matched to active MOT tracks in one frame.
3. `semantic_memory`: several frame resolutions are aggregated into one final language-to-track decision.

This milestone only extends `src/football_tracking/locate_tracking/`. It does not modify
YOLO, BoT-SORT, DeepSORT, TrackEval, Qwen VLM, or existing experiment code.

## Inputs And Outputs

The end-to-end input is:

- `source_video`: original video.
- `tracks_path`: read-only MOT TXT tracking artifact.
- `query`: language phrase, for example `the player in red`.
- `sampling_config`: uniform or explicit frame selection.
- `grounding_config`: LocateAnything or mock grounding config.
- `association_config`: single-frame matching config.
- `semantic_memory_config`: evidence aggregation and decision thresholds.

The main outputs are:

- `semantic_memory.json`: candidate-level evidence histories and aggregate scores.
- `final_resolution.json`: final status and selected track id(s).
- `session.json`: deterministic session record with sampling plan, frame outputs, memory, and final decision.
- `semantic_summary.md`: compact human-readable summary.

## Sampling

Frame indices are 1-based. The default selector is deterministic uniform temporal sampling.
It avoids running LocateAnything on every frame.

Uniform example:

```powershell
python -m football_tracking.locate_tracking.cli resolve-language-track `
  --source-video F:\videos\1.mp4 `
  --tracks F:\videos\1_Tracking.txt `
  --query "the player in red" `
  --max-frames 5 `
  --output-dir outputs\locate_tracking\semantic_memory\video_1 `
  --backend mock `
  --overwrite
```

Explicit frame list:

```powershell
python -m football_tracking.locate_tracking.cli resolve-language-track `
  --source-video F:\videos\1.mp4 `
  --tracks F:\videos\1_Tracking.txt `
  --query "the player in red" `
  --frames 100,140,180 `
  --output-dir outputs\locate_tracking\semantic_memory\video_1_explicit `
  --backend mock `
  --overwrite
```

For real LocateAnything, remove `--backend mock` and use the grounding config/model setup.
On an RTX 4060 Laptop GPU, start with 3 to 5 frames and keep batch size at 1.

## Evidence Semantics

Each single-frame association is converted into candidate-level semantic evidence.

- `resolved`: the selected candidate receives full positive support.
- `ambiguous`: candidates are preserved as weak evidence, but they do not count as positive votes.
- `not_found`: no positive support is added.

Track absence in a sampled frame is not treated as contradiction. It simply contributes no
positive evidence for that track.

## Aggregation

Weighted aggregation uses:

```text
S_track = w_support * SupportScore + w_quality * QualityScore + w_consistency * ConsistencyScore
```

The default effective weights are support `0.50`, quality `0.30`, and consistency `0.20`.
The `majority_support` baseline is also implemented. Its deterministic tie-breaker is:

1. support count descending
2. mean association score descending
3. best association score descending
4. raw track id ascending

## Decision Statuses

The final output status is one of:

- `resolved`: one or more tracks pass the configured thresholds.
- `ambiguous`: top single-target candidates are too close.
- `not_found`: grounding was usable, but no candidate passed quality/support thresholds.
- `insufficient_evidence`: too few usable frames or too little positive support.

`query_mode` is explicit. Use `single_target` for one object/person and `multi_target` when
the query is expected to return multiple tracks. The pipeline does not infer this from text.

## Aggregate Existing M2 Artifacts

If you already have single-frame `association.json` files:

```powershell
python -m football_tracking.locate_tracking.cli aggregate-language-track `
  --query "the player in red" `
  --frame-resolution outputs\locate_tracking\queries\f100\association.json `
  --frame-resolution outputs\locate_tracking\queries\f140\association.json `
  --frame-resolution outputs\locate_tracking\queries\f180\association.json `
  --sampled-frames 100,140,180 `
  --output-dir outputs\locate_tracking\semantic_memory\aggregate_red `
  --overwrite
```

This mode is CPU-only and does not call LocateAnything.

## Limitations

This layer improves stability of language-to-track decisions across sampled frames. It does
not repair poor detections or tracking IDs by itself. If the underlying tracking artifact
has many ID switches, the semantic memory can identify the most consistent candidate among
the sampled frames, but full ID repair still needs a tracker-level or post-processing method.
