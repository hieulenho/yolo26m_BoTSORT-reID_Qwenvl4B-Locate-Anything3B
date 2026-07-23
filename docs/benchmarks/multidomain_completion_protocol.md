# Multi-domain completion protocol

This document defines when the adaptive tracking system may be described as
quantitatively validated. A model proposal, rendered label, or heuristic diagnosis is
never treated as ground truth without an independent reference.

## Official benchmark sources

| Domain | Source | Local format | Primary metrics | Access gate |
|---|---|---|---|---|
| Sports | [SportsMOT](https://github.com/MCG-NJU/SportsMOT) | MOTChallenge | HOTA, DetA, AssA, MOTA, IDF1, IDSW, FPS | Present locally |
| Open world | [TAO](https://taodataset.org/) | COCO-video/TAO | TETA/HOTA, class accuracy, FPS | Official download |
| Traffic | [BDD100K](https://github.com/bdd100k/bdd100k) | Scalabel JSON | TETA/HOTA, MOTA, IDF1, class accuracy | Account and terms |
| Wildlife | [AnimalTrack](https://hengfan2010.github.io/projects/AnimalTrack/) | class-aware MOT | HOTA, AssA, IDF1, IDSW, species accuracy | Non-commercial research terms |
| Medical microscopy | [Cell Tracking Challenge](https://celltrackingchallenge.net/datasets/) | masks and lineage | official TRA/DET/SEG; optional bbox HOTA | Check conditions of use |
| Education | Local classroom review package | semantic review package | macro-F1, fine-label accuracy, rejection F1 | Independent human review |

TAO contains 2,907 videos, 833 classes and 17,287 tracks. BDD100K contains 100,000
40-second driving videos and includes multi-object tracking. AnimalTrack contains 58
sequences, about 25,000 frames and ten animal categories. These properties make the
matrix meaningfully broader than a small collection of unreviewed web clips.

## Readiness audit

```powershell
.\.venv\Scripts\python.exe scripts\benchmarks\audit_multidomain_sources.py `
  --registry configs\benchmarks\multidomain_sources.yaml `
  --output outputs\benchmarks\multidomain\dataset_readiness.json
```

The command does not bypass data portals or licenses. `ready: true` means every local
requirement exists. A semantic review package is ready only after all rows and sample
metadata have independent human review.

Official class labels matched from benchmark annotations are audited separately:

```powershell
.\.venv\Scripts\python.exe scripts\benchmarks\audit_official_semantic_gt.py `
  --manifest outputs\benchmarks\semantic\ua_detrac_mvi_40774\pipeline_a_qwen\semantic_gt_manifest.yaml `
  --manifest outputs\benchmarks\semantic\animaltrack_zebra\pipeline_a_qwen\semantic_gt_manifest.yaml `
  --minimum-domains 2 `
  --minimum-tracks 40 `
  --output outputs\benchmarks\multidomain\official_semantic_gt_status.json
```

## Ground-truth normalization

```powershell
# BDD100K Scalabel JSON
.\.venv\Scripts\python.exe scripts\benchmarks\convert_multidomain_gt.py `
  --format bdd100k_scalabel `
  --annotations data\external\bdd100k\labels\box_track_20 `
  --output-dir data\normalized\bdd100k `
  --overwrite

# TAO COCO-video JSON
.\.venv\Scripts\python.exe scripts\benchmarks\convert_multidomain_gt.py `
  --format tao_coco_video `
  --annotations data\external\tao\annotations\validation.json `
  --output-dir data\normalized\tao `
  --overwrite

# AnimalTrack class-aware MOT files
.\.venv\Scripts\python.exe scripts\benchmarks\convert_multidomain_gt.py `
  --format animaltrack_mot `
  --annotations data\external\animaltrack\annotations `
  --category-map data\external\animaltrack\classes.json `
  --output-dir data\normalized\animaltrack `
  --overwrite

# CTC masks to an optional bbox-MOT view
.\.venv\Scripts\python.exe scripts\benchmarks\convert_multidomain_gt.py `
  --format ctc_masks_lineage `
  --annotations data\external\cell_tracking_challenge `
  --output-dir data\normalized\cell_tracking_challenge `
  --overwrite
```

The CTC conversion supports cross-domain bbox tracking analysis. It does not replace the
official CTC evaluator for TRA, DET and SEG.

## Semantic A/B/C metrics

All three pipelines must use the same reviewed manifest and track IDs. The evaluator
reports:

- domain and detector-router accuracy;
- dynamic-vocabulary precision, recall and F1;
- track-label accuracy and macro-F1;
- fine-label accuracy for subtype, species, vehicle type, role, make or model;
- coverage and selective accuracy;
- unknown-rejection precision, recall and F1;
- false-accept rate on unknown GT and false-reject rate on known GT;
- accepted hallucination rate;
- tracking FPS, Qwen/Locate wall time and peak VRAM when artifacts provide timings.

An accepted prediction is a hallucination for this benchmark when its semantic label
disagrees with reviewed GT. This is an operational error rate, not a free-form language
quality score.

## ID-switch validation

The official IDSW total comes from TrackEval. The five diagnostic causes are a separate,
human-reviewed analysis.

```powershell
.\.venv\Scripts\python.exe scripts\benchmarks\review_idsw_taxonomy.py evidence `
  --events outputs\benchmarks\tracking\sportsmot_yolo26m\smoke\idsw_taxonomy\idsw_taxonomy_events.csv `
  --dataset-root data\mot\sportsmot_football `
  --tracks-root outputs\benchmarks\tracking\sportsmot_yolo26m\smoke\tracks `
  --output-dir outputs\benchmarks\tracking\sportsmot_yolo26m\smoke\idsw_taxonomy\review_evidence `
  --overwrite
```

Two reviewers independently complete separate review CSVs. Agreement is then measured:

```powershell
.\.venv\Scripts\python.exe scripts\benchmarks\review_idsw_taxonomy.py agreement `
  --review-a data\idsw_review\reviewer_a.csv `
  --review-b data\idsw_review\reviewer_b.csv `
  --output outputs\benchmarks\tracking\idsw_reviewer_agreement.json
```

Disagreements require adjudication. Heuristic category percentages must not be presented
as human-verified before this step is complete.

## Physical realtime protocol

Point the camera at a stable test scene or provide an RTSP URL, then run:

```powershell
.\scripts\benchmarks\run_physical_realtime_protocol.ps1 `
  -Source "0" `
  -ProtocolName webcam_900_frames `
  -MaxFrames 900 `
  -Repeats 3 `
  -Overwrite
```

The protocol measures bounded-latency tracking, bounded tracking with deferred semantics,
and no-drop processing. It calibrates once, reuses the detector/tracker plan, and reports
mean plus standard deviation across repeated runs. Physical streams use a latest-frame
reader, so dropped frames come from successful camera reads rather than backend-dependent
`grab()` calls. It records p50/p95/p99 latency, FPS, drop rate, startup and shutdown time,
RAM, VRAM, GPU and CPU. `-IncludeLiveSemantic` is optional because running the tracker and
both VLMs concurrently can exceed an 8 GB GPU budget.

## Release gate

The system is quantitatively complete only when all of the following are true:

1. Every published domain has downloaded, licensed GT and a recorded dataset version.
2. Semantic GT comes from official benchmark labels with audited provenance, or human
   review coverage is 100% with reviewer and timestamp metadata.
3. A/B/C use the same GT and publish accuracy, macro-F1, fine accuracy, rejection F1,
   hallucination rate, runtime and peak memory.
4. IDSW totals come from TrackEval; diagnostic causes have two-reviewer agreement and
   adjudication.
5. Webcam/RTSP measurements include at least three repeated runs per profile and report
   mean, standard deviation and percentiles on named hardware.
6. Dataset adapters, evaluators, PowerShell scripts and the full test suite pass.

Until these gates are satisfied, reports must state the exact missing evidence instead of
claiming 100% completion.

Rebuild the machine-readable gate with command 16 in `commands.txt`. Its canonical output is
`outputs/benchmarks/multidomain/completion/completion_gate.json` plus a Markdown summary.
