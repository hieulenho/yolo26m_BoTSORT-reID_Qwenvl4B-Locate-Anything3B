# Local outputs

`scripts/run_traffic.ps1` writes to `outputs/traffic/<RunName>/` by default.
Each run contains its MP4, MOT TXT, runtime metrics and (when enabled) semantic
worker artifacts. Choose a new run name to preserve an earlier run.

Generated videos, measurements, credentials and temporary files are not committed.
Only reviewed traffic reports under `docs/benchmarks/traffic_quality/` are public.
