# Video 1 Expanded Team Benchmark

This folder contains an expanded real-video benchmark for `F:/videos/1.mp4`.

- `benchmark_manifest_expanded.json`: 21 reviewed tracks and 6 language queries.
- `track_annotation_expanded.csv`: CSV view of labels and contact-sheet paths.
- `pipeline_a_qwen_expanded_bootstrap.json`: Qwen-style bootstrap predictions.
- `pipeline_b_locate_qwen_expanded_bootstrap.json`: Locate+Qwen bootstrap predictions.

Important: the expanded prediction manifests are bootstrap artifacts based on
contact-sheet labels. They are useful for checking benchmark mechanics over more
tracks, but final research claims should replace them with true model outputs.
