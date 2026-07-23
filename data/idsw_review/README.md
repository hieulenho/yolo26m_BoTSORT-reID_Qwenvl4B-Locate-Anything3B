# IDSW double-review package

This package is the only remaining manual release gate. Do not copy the
`heuristic_type` column into the reviewed result without inspecting the evidence.

## Evidence

Open:

`outputs/benchmarks/tracking/sportsmot_yolo26m/smoke/idsw_taxonomy/review_evidence/index.html`

Each row links one image containing the frame before, at, and after the reported switch.
The GT identity, old prediction ID, new prediction ID, detector boxes, and nearby tracks
are shown together.

## Allowed labels

| Label | Use when |
|---|---|
| `fragmentation` | The same target disappears briefly and resumes with a new ID. |
| `identity_swap` | Two visible targets exchange their assigned identities. |
| `re_identification_failure` | A target returns after a longer absence but receives a new ID. |
| `association_error` | The continuous target is linked to the wrong detection without a clear appearance-confusion cue. |
| `appearance_confusion` | Similar clothing or appearance is the dominant reason for the wrong match. |

## Independent review

Reviewer A edits `reviewer_a.csv`; reviewer B edits `reviewer_b.csv`. For every row:

1. Set `reviewed_type` to exactly one allowed label.
2. Set `review_status` to `reviewed`, or `ignored` only when the evidence is unusable.
3. Fill `reviewer` with a stable reviewer name.
4. Add a short visual reason in `notes`.

Do not let reviewer B see reviewer A's labels before both files are complete.

## Validate and compare

```powershell
.\.venv\Scripts\python.exe scripts\benchmarks\review_idsw_taxonomy.py status `
  --review data\idsw_review\reviewer_a.csv

.\.venv\Scripts\python.exe scripts\benchmarks\review_idsw_taxonomy.py status `
  --review data\idsw_review\reviewer_b.csv

.\.venv\Scripts\python.exe scripts\benchmarks\review_idsw_taxonomy.py agreement `
  --review-a data\idsw_review\reviewer_a.csv `
  --review-b data\idsw_review\reviewer_b.csv `
  --output outputs\benchmarks\tracking\sportsmot_yolo26m\smoke\idsw_taxonomy\idsw_reviewer_agreement.json
```

Resolve disagreements in a separate adjudication pass. Rebuild the completion gate only
after both status commands report full coverage and the agreement artifact reports
`status: agreed`.
