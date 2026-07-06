# Language Benchmark Annotation Guide

This guide defines how to write annotations for the optional `locate_tracking`
language-guided semantic tracking benchmark.

## Query Rules

- Every query must be understandable from the clip alone.
- Use visible appearance, role, team/group, spatial, or compound descriptions.
- Do not annotate jersey-number queries unless OCR-style evidence is implemented.
- Do not annotate action-history queries unless temporal action reasoning is implemented.

## Query Modes

- `single_target`: one semantic target, such as `the goalkeeper wearing green`.
- `multi_target`: a set of targets, such as `players wearing white`.

The mode is explicit in the manifest. The evaluator does not infer it from text.

## Difficulty

Use:

- `easy`: visually distinct target, low crowding, few occlusions.
- `medium`: some ambiguity, intermittent occlusion, or moderate raw-ID fragmentation.
- `hard`: heavy crowding, same-kit confusion, frequent occlusion, or major camera motion.

Difficulty is metadata only. Runtime prediction must not use it.

## Ground Truth Segments

Each query needs identity segments:

```json
{
  "gt_track_id": 3,
  "start_frame": 100,
  "end_frame": 900
}
```

GT IDs are dataset labels. Predicted raw IDs are arbitrary and must be matched spatially,
not by numeric equality.

## Evaluation Interval

Set `evaluation_start_frame` and `evaluation_end_frame` to the frames where the query is
valid and the target is expected to be evaluated. Exclude intervals where the target is
not annotated.

## Loss And Reacquisition Opportunities

Annotate opportunities only when the target should be recoverable:

- `target_lost_frame`
- candidate search range
- GT reappearance frame
- evaluation window after recovery

Do not mix actual visual disappearance with predicted raw-ID fragmentation unless noted.

## Ambiguous Queries

If two or more targets satisfy the wording equally, rewrite the query or mark the case as
not suitable for the current benchmark version.
