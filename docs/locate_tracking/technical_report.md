# Language-Guided Semantic Tracking Technical Report

## Abstract

This template is populated by generated benchmark artifacts. Do not manually type metric
values here; use `generate-language-report`.

## 1. Introduction

Describe the problem of resolving natural-language targets over saved MOT tracking.

## 2. Related Problem Setting

Separate raw MOT tracking quality from language-guided semantic identity quality.

## 3. System Architecture

Summarize YOLO + BoT-SORT as the original pipeline and `locate_tracking` as the optional
parallel semantic layer.

## 4. Language-Guided Grounding

Describe M1 grounding artifacts.

## 5. Semantic Track Association

Describe M2 frame-level geometric matching.

## 6. Multi-Frame Semantic Memory

Describe M3 aggregation.

## 7. Appearance Verification

Describe M4 frozen appearance evidence.

## 8. Event-Triggered Grounding

Describe M5 uncertainty monitoring and grounding scheduling.

## 9. Semantic Target Reacquisition

Describe M6 candidate search, evidence, ranking, probation, and identity transitions.

## 10. Experimental Setup

Reference benchmark manifest, prediction manifests, and hardware.

## 11. Benchmark Protocol

Explain dev/val/test split policy and annotation rules.

## 12. Metrics

Report initial selection accuracy, resolution rate, target precision/recall/F1,
continuity ratio, semantic switches, reacquisition success, false reacquisition, frames
to reacquire, and grounding calls per 1000 frames.

## 13. Main Results

Generated reports should inject numbers from `aggregate_metrics.json`.

## 14. Ablation Study

Use A0-A5 variant outputs from `ablation_results.json`.

## 15. Efficiency Analysis

Use saved grounding call counts and runtime metadata.

## 16. Failure Analysis

Use deterministic failure categories from `failure_cases.json`.

## 17. Limitations

Discuss small manually annotated data, football focus, tracker dependence, visual
similarity, lack of OCR/action reasoning, offline artifact workflow, and threshold-based
fusion.

## 18. Conclusion

Summarize what the measured artifacts support, without unsupported claims.
