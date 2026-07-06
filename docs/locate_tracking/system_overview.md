# Locate Tracking System Overview

## 1. Problem

The project tracks objects with YOLO + BoT-SORT, then optionally resolves language
queries such as `the goalkeeper wearing green` onto saved raw tracking artifacts.

## 2. Design Principles

- Original tracking remains independently usable.
- Language tracking is an optional parallel subsystem.
- Ground truth is used only by evaluators, never by runtime prediction.
- Raw MOT IDs are not rewritten.

## 3. Original Tracking Pipeline

```text
video -> YOLO26m -> BoT-SORT ReID -> MOT TXT -> TrackEval / rendering / VLM
```

## 4. Parallel Language Tracking Pipeline

```text
query + saved frames/MOT -> grounding -> association -> semantic memory
-> appearance verification -> monitoring -> grounding events -> reacquisition
-> stable semantic identity
```

## 5. Grounding

M1 stores `GroundingResult` artifacts from LocateAnything or mock backends.

## 6. Association

M2 matches grounded boxes to active MOT rows using geometry.

## 7. Semantic Memory

M3 aggregates multi-frame evidence into semantic track memory.

## 8. Appearance Verification

M4 builds frozen appearance evidence from source-video crops.

## 9. Monitoring

M5 observes uncertainty signals and produces event-triggered grounding plans.

## 10. Event-Triggered Grounding

M5 executes selected grounding requests and saves results for later use.

## 11. Reacquisition

M6 searches raw-track candidates after loss events and ranks them using grounding,
appearance, motion, temporal, and history evidence.

## 12. Semantic Identity

M6 stores stable semantic targets as raw-ID segments. This preserves target identity
without changing raw tracker IDs.

## 13. Benchmark

M7 adds a separate language benchmark manifest, evaluator, ablation runner, failure
analysis, report generation, and demo manifest builder.

## 14. Commands

Use `python -m football_tracking.locate_tracking.cli --help` for the language subsystem.

## 15. Artifact Contracts

Runtime writes prediction artifacts. Evaluation reads prediction artifacts and benchmark
ground truth. Reports read metric artifacts.

## 16. Limitations

The benchmark starts with a synthetic smoke fixture. Real claims require more annotated
queries, splits, and frozen thresholds.
