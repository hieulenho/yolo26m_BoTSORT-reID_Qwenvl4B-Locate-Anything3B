#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
DEVICE="${DEVICE:-0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" && -x "$ROOT/.venv/Scripts/python.exe" ]]; then
  PYTHON="$ROOT/.venv/Scripts/python.exe"
fi
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="${PYTHON_BIN:-python}"
fi

run_step() {
  local name="$1"
  shift
  printf '\n==> %s\n' "$name"
  "$PYTHON" -m football_tracking.cli "$@"
}

case "$MODE" in
  full)
    TRAIN_CONFIG="configs/yolov8m_sportsmot_train.yaml"
    EVAL_CONFIG="configs/yolov8m_sportsmot_eval.yaml"
    CACHE_CONFIG="configs/detection_cache.yaml"
    COMPARE_CONFIG="configs/compare_trackers.yaml"
    RENDER_CONFIG="configs/render_video.yaml"
    BENCHMARK_CONFIG="configs/benchmark.yaml"
    REPORT_CONFIG="configs/report.yaml"
    ;;
  smoke)
    TRAIN_CONFIG="configs/yolov8m_sportsmot_smoke.yaml"
    EVAL_CONFIG="configs/yolov8m_sportsmot_smoke_eval.yaml"
    CACHE_CONFIG="configs/detection_cache_smoke.yaml"
    COMPARE_CONFIG="configs/compare_trackers_smoke.yaml"
    RENDER_CONFIG="configs/render_video_smoke.yaml"
    BENCHMARK_CONFIG="configs/benchmark_smoke.yaml"
    REPORT_CONFIG="configs/report_smoke.yaml"
    ;;
  *)
    echo "Usage: demo/demo.sh [smoke|full]" >&2
    exit 2
    ;;
esac

cd "$ROOT"
run_step "doctor" doctor
run_step "prepare dataset" prepare-dataset --config configs/sportsmot_data.yaml --overwrite
run_step "train detector" train-detector --config "$TRAIN_CONFIG" --device "$DEVICE" --overwrite
run_step "evaluate detector" evaluate-detector --config "$EVAL_CONFIG"
run_step "cache detections" cache-detections --config "$CACHE_CONFIG" --device "$DEVICE" --overwrite
run_step "compare trackers" compare-trackers --config "$COMPARE_CONFIG" --overwrite
run_step "evaluate tracking" evaluate-tracking --config "$COMPARE_CONFIG"
run_step "render video" render-video --config "$RENDER_CONFIG" --overwrite
run_step "benchmark" benchmark --config "$BENCHMARK_CONFIG"
run_step "generate report" generate-report --config "$REPORT_CONFIG"

printf '\nDemo pipeline completed in %s mode.\n' "$MODE"
