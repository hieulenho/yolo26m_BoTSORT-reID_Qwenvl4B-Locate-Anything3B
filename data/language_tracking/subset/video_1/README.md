# Language Benchmark Template

Review these files before reporting numbers:

- `data\language_tracking\subset\video_1\benchmark_manifest.json`: GT annotation manifest.
- `data\language_tracking\subset\video_1\predictions_a5_full_system.json`: prediction artifact manifest.

Checklist:

1. Confirm `target_gt_track_ids` uses dataset GT IDs, not predicted IDs.
2. Confirm `identity_segments` frame ranges match visible GT target boxes.
3. Confirm prediction `semantic_target_path` points to runtime output only.
4. Run validation before benchmark evaluation.
