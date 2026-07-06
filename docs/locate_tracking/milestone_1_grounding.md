# Milestone 1: Standalone LocateAnything Grounding

## Purpose

This milestone adds a parallel image-grounding subsystem for future
language-guided football tracking.  It does not replace or modify the existing
detector, tracker, evaluator, or Qwen VLM workflow.

## Existing Pipeline

```text
EXISTING PIPELINE - UNCHANGED

Video
  ->
YOLO26m
  ->
BoT-SORT ReID
  ->
MOT Outputs
  ->
Evaluation / Qwen Analysis
```

Existing MOT outputs, tracking videos, TrackEval results, VLM context files, and
Qwen reports keep their current contracts.

## New Parallel Pipeline

```text
NEW PARALLEL PIPELINE

Image
+
Language Query
  ->
LocateAnything
  ->
Grounding Parser
  ->
Grounded Boxes
  ->
Grounding JSON
```

Integration between the two pipelines is intentionally deferred to later
milestones.

## Directory Structure

```text
src/football_tracking/locate_tracking/
  grounding/
    schemas.py
    backend.py
    locate_anything_backend.py
    parser.py
    coordinates.py
    cache.py
    service.py
  cli/
    locate_image.py

configs/locate_tracking/
  locateanything_grounding.yaml

outputs/locate_tracking/
  cache/
  grounding/
```

## Coordinate Format

LocateAnything responses are parsed from tokens such as:

```text
<ref>goalkeeper</ref><box><100><200><400><900></box>
```

The parser treats box coordinates as 0-1000 normalized XYXY values and converts
them into pixel XYXY values using the image width and height.  Invalid boxes are
reported as parser errors and are not converted into detections.

## Cache Behavior

The cache lives only under:

```text
outputs/locate_tracking/cache
```

Cache keys include image content hash, query text, backend name, model id, and
inference configuration.  Two files with the same filename but different content
therefore do not collide.

## CLI

```powershell
cd F:\Tracking

.\.venv\Scripts\python.exe -m football_tracking.cli locate-image `
  --config configs\locate_tracking\locateanything_grounding.yaml `
  --image F:\Tracking\data\sample.jpg `
  --query "the goalkeeper wearing green" `
  --output outputs\locate_tracking\grounding\sample.json `
  --overwrite
```

The command loads one image, runs phrase grounding, and writes one JSON artifact.
It does not initialize YOLO, BoT-SORT, TrackEval, or Qwen.

## Mock Testing

Unit tests use `MockGroundingBackend`, which returns deterministic text outputs
without internet, GPU, CUDA, Hugging Face downloads, or LocateAnything weights.

## Optional Dependencies

The real LocateAnything backend is lazy loaded.  Users who only run the existing
YOLO26m + BoT-SORT ReID pipeline do not need LocateAnything dependencies.

The real backend needs the usual Hugging Face/PyTorch image-text-to-text stack,
for example `transformers`, `accelerate`, `torch`, and `pillow`.

## RTX 4060 Laptop 8 GB Notes

Start with one image at a time, keep `batch_size: 1`, and use `bfloat16` or
`float16` on CUDA.  If memory is tight, reduce prompt scope and keep outputs
short with `max_new_tokens`.

## Known Limitations

- Image-only grounding.
- No BoT-SORT integration.
- No track ID association.
- No ReID verification.
- No semantic track memory.
- No lost-target detection or reacquisition.
- No full-video language tracking.

