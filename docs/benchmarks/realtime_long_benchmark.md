# Realtime Benchmark

Hardware: NVIDIA GeForce RTX 4060 Laptop GPU (8.0 GB VRAM), 20 logical CPU threads.

| Run | Process FPS | Source progress FPS | p95 | Drop | Startup | RAM | VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_fp32 | 21.04 | 20.37 | 77.1 ms | 0.0% | 5.6s | 1.67 GB | 0.16 GB |
| optimized_no_drop | 27.35 | 26.10 | 45.1 ms | 0.0% | 10.6s | 2.01 GB | 0.16 GB |
| bounded_live | 27.88 | 29.98 | 44.0 ms | 11.3% | 11.5s | 2.03 GB | 0.15 GB |

The bounded-latency run drops late input frames instead of accumulating camera lag.
Use the no-drop profile for offline accuracy evaluation.

![FPS](../assets/benchmarks/realtime_long_fps.png)

![Latency](../assets/benchmarks/realtime_long_latency_drop.png)
