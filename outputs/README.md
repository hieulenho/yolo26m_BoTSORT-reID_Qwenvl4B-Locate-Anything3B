# Output Layout

Generated outputs are ignored by Git. The current adaptive system writes to:

```text
adaptive_runs/       offline per-video runs
adaptive_realtime/   live/file stream sessions
benchmarks/          detector, tracker, semantic, and runtime reports
cache/               reusable semantic discovery cache
detections/cache/    shared detector results used for fair tracker comparison
```

Other folders contain historical football-only experiments and are preserved locally only when
they are still referenced by a benchmark contract. Disposable test files belong in
`pytest_tmp/` and can be removed at any time.
