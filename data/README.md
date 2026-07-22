# Data Layout

```text
raw/                 immutable downloaded inputs
interim/             conversion workspace
yolo/                detector-format datasets
mot/                 MOT sequences and ground truth
samples/             licensed multi-domain sample manifests
semantic_benchmark/  human-reviewed semantic annotations
```

`language_tracking/` and `team_benchmark/` contain historical annotation formats retained for
reproducing earlier experiments. Generated files and large media remain ignored by Git.
