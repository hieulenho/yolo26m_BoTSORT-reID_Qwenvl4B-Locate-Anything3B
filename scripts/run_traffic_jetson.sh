#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

SOURCE="0"
MAX_FRAMES=0
NO_WINDOW=0
CAPTURE_WIDTH=1280
CAPTURE_HEIGHT=720
CAPTURE_FPS=30
CAPTURE_FOURCC=MJPG
OUTPUT_ROOT="outputs/traffic_jetson"
RUN_NAME="traffic_$(date +%Y%m%d_%H%M%S_%3N)"

usage() {
    echo "Usage: $0 [--source 0] [--no-window] [--max-frames N] [--run-name NAME]"
    echo "          [--capture-width N] [--capture-height N] [--capture-fps N]"
    echo "          [--capture-fourcc MJPG] [--output-root DIR]"
}

while (($#)); do
    case "$1" in
        --source) SOURCE="$2"; shift 2 ;;
        --no-window) NO_WINDOW=1; shift ;;
        --max-frames) MAX_FRAMES="$2"; shift 2 ;;
        --capture-width) CAPTURE_WIDTH="$2"; shift 2 ;;
        --capture-height) CAPTURE_HEIGHT="$2"; shift 2 ;;
        --capture-fps) CAPTURE_FPS="$2"; shift 2 ;;
        --capture-fourcc) CAPTURE_FOURCC="$2"; shift 2 ;;
        --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
        --run-name) RUN_NAME="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [[ ! -x .venv/bin/python ]]; then
    echo "Missing .venv/bin/python in $PROJECT_ROOT" >&2
    exit 2
fi
if (( ! NO_WINDOW )) && [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
    echo "No desktop display in this session. Run on the Jetson desktop or use --no-window." >&2
    exit 2
fi

source .venv/bin/activate
export BNB_CUDA_VERSION=132
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda-13.2/lib64:${LD_LIBRARY_PATH:-}"

TASK_CONFIG="configs/tasks/traffic_realtime_jetson.yaml"
RUN_ROOT="$OUTPUT_ROOT/$RUN_NAME"
RUNTIME_ROOT="$RUN_ROOT/runtime"
OUTPUT_VIDEO="$RUN_ROOT/tracked_semantic.mp4"
OUTPUT_MOT="$RUN_ROOT/tracked_semantic.txt"
METADATA="$RUN_ROOT/realtime_metrics.json"
SEMANTIC_QUEUE="$RUN_ROOT/semantic_queue"
SEMANTIC_CACHE="$RUN_ROOT/semantic_cache.json"
SEMANTIC_MEMORY="$RUN_ROOT/semantic_memory.json"
WORKER_STOP="$RUN_ROOT/semantic_worker.stop"
WORKER_REPORT="$RUN_ROOT/semantic_worker_report.json"
WORKER_STDOUT="$RUN_ROOT/semantic_worker.stdout.log"
WORKER_STDERR="$RUN_ROOT/semantic_worker.stderr.log"
WORKER_PID_FILE="$RUN_ROOT/semantic_worker.pid"
WORKER_STATUS="$RUN_ROOT/semantic_worker.status.json"

mkdir -p "$RUN_ROOT" "$SEMANTIC_QUEUE"
ln -sfn "$RUN_NAME" "$OUTPUT_ROOT/latest"
echo "Source: $SOURCE | Run: $RUN_ROOT"

python - <<'PY'
import accelerate
import bitsandbytes
import qwen_vl_utils
import torch
import transformers

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable on Jetson")
print(f"torch={torch.__version__} transformers={transformers.__version__} "
      f"bitsandbytes={bitsandbytes.__version__} gpu={torch.cuda.get_device_name(0)}")
PY

python -m football_tracking.task_pipeline.cli build \
    --task "$TASK_CONFIG" \
    --output-dir "$RUNTIME_ROOT" \
    --output-video "$OUTPUT_VIDEO" \
    --device cuda \
    --overwrite

GENERATED_CONFIG="$RUNTIME_ROOT/tracking.generated.yaml"
RESOLVED_TASK="$RUNTIME_ROOT/task.resolved.json"
read -r EVENT_INTERVAL EVENTS_PER_FRAME TASK_MAX_PENDING < <(
    python - "$RESOLVED_TASK" <<'PY'
import json
import sys

semantic = json.load(open(sys.argv[1], encoding="utf-8"))["semantic"]
print(semantic["event_interval_frames"], semantic["events_per_frame"],
      semantic["max_pending_events"])
PY
)
MAX_PENDING=$((TASK_MAX_PENDING < 4 ? TASK_MAX_PENDING : 4))

python scripts/runtime/run_realtime_semantic_worker.py \
    --queue-dir "$SEMANTIC_QUEUE" \
    --vlm-config configs/semantics/dynamic_track.yaml \
    --semantic-output "$SEMANTIC_CACHE" \
    --memory "$SEMANTIC_MEMORY" \
    --max-events 2 \
    --group-events \
    --max-group-images 2 \
    --report "$WORKER_REPORT" \
    --pid-file "$WORKER_PID_FILE" \
    --status-file "$WORKER_STATUS" \
    --parent-pid "$$" \
    --watch \
    --stop-file "$WORKER_STOP" \
    >"$WORKER_STDOUT" 2>"$WORKER_STDERR" &
WORKER_PID=$!

cleanup_worker() {
    shopt -s nullglob
    for _ in $(seq 1 120); do
        if ! kill -0 "$WORKER_PID" 2>/dev/null; then
            break
        fi
        outstanding=("$SEMANTIC_QUEUE"/pending/*.json "$SEMANTIC_QUEUE"/processing/*.json)
        if ((${#outstanding[@]} == 0)); then
            break
        fi
        sleep 1
    done
    touch "$WORKER_STOP" 2>/dev/null || true
    for _ in $(seq 1 30); do
        if ! kill -0 "$WORKER_PID" 2>/dev/null; then
            break
        fi
        sleep 1
    done
    if kill -0 "$WORKER_PID" 2>/dev/null; then
        echo "Qwen worker did not stop in 30 s; terminating it" >&2
        kill "$WORKER_PID" 2>/dev/null || true
    fi
    wait "$WORKER_PID" 2>/dev/null || true
}
trap cleanup_worker EXIT

REALTIME_ARGS=(
    python scripts/runtime/run_realtime_adaptive.py
    --config "$GENERATED_CONFIG"
    --task-config "$TASK_CONFIG"
    --source "$SOURCE"
    --output-video "$OUTPUT_VIDEO"
    --output-mot "$OUTPUT_MOT"
    --metadata "$METADATA"
    --semantic-queue-dir "$SEMANTIC_QUEUE"
    --semantic-cache "$SEMANTIC_CACHE"
    --semantic-event-interval-frames "$EVENT_INTERVAL"
    --semantic-events-per-frame "$EVENTS_PER_FRAME"
    --semantic-max-pending-events "$MAX_PENDING"
    --semantic-cache-reload-frames 5
    --overwrite
)
if [[ "$SOURCE" =~ ^[0-9]+$ ]]; then
    REALTIME_ARGS+=(
        --capture-width "$CAPTURE_WIDTH"
        --capture-height "$CAPTURE_HEIGHT"
        --capture-fps "$CAPTURE_FPS"
        --capture-fourcc "$CAPTURE_FOURCC"
    )
fi
if (( NO_WINDOW )); then
    REALTIME_ARGS+=(--no-window)
fi
if (( MAX_FRAMES > 0 )); then
    REALTIME_ARGS+=(--max-frames "$MAX_FRAMES")
fi

"${REALTIME_ARGS[@]}"
cleanup_worker
trap - EXIT

echo "Video: $OUTPUT_VIDEO"
echo "MOT: $OUTPUT_MOT"
echo "Metrics: $METADATA"
echo "Semantic cache: $SEMANTIC_CACHE"
echo "Worker report: $WORKER_REPORT"
if [[ ! -f "$WORKER_REPORT" ]]; then
    echo "Warning: Qwen worker did not produce a report; inspect $WORKER_STDERR" >&2
    exit 1
fi
read -r PROCESSED FAILED REMAINING CALLS < <(
    python - "$WORKER_REPORT" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
print(report.get("processed_event_count", 0), report.get("failed_event_count", 0),
      report.get("remaining_event_count", 0), report.get("qwen_call_count", 0))
PY
)
echo "Qwen: processed=$PROCESSED failed=$FAILED remaining=$REMAINING calls=$CALLS"
if (( FAILED > 0 || REMAINING > 0 )); then
    echo "Semantic output is incomplete; inspect $WORKER_REPORT and $WORKER_STDERR" >&2
    exit 1
fi
