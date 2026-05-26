#!/usr/bin/env bash
set -u

GPU_ID="${1:?usage: run_rgbt_scenes_rgbt_worker.sh GPU_ID}"
PYTHON_BIN="${PYTHON_BIN:-/home/imglab/anaconda3/bin/python}"

REPO="/home/imglab/GJY/physir-splat-main"
ROOT="$REPO/RGBT-Scenes"
OUT="$REPO/output/rgbt_scenes_rgbt"
LOG="$REPO/logs/rgbt_scenes_rgbt"
RUN="$OUT/.run"
STATUS="$OUT/status.tsv"

SCENES=(
  Building
  DailyStuff
  Dimsum
  Ebike
  IronIngot
  LandScape
  Parterre
  RoadBlock
  RotaryKiln
  Truck
)

mkdir -p "$OUT" "$LOG" "$RUN"
cd "$REPO" || exit 1

append_status() {
  local scene="$1"
  local stage="$2"
  local status="$3"
  printf "%s\t%s\t%s\tgpu%s\t%s\n" "$scene" "$stage" "$status" "$GPU_ID" "$(date --iso-8601=seconds)" >> "$STATUS"
}

train_done() {
  local model="$1"
  test -f "$model/point_cloud/iteration_30000/point_cloud.ply"
}

metrics_done() {
  local model="$1"
  test -f "$model/results.json"
}

run_render_metrics() {
  local scene="$1"
  local model="$OUT/$scene"
  local scene_log="$LOG/${scene}.log"
  local code=0

  append_status "$scene" "render" "started"
  {
    printf "[%s] render start %s gpu=%s\n" "$(date --iso-8601=seconds)" "$scene" "$GPU_ID"
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" render.py -m "$model" --quiet
    code=$?
    printf "[%s] render exit %s code=%s\n" "$(date --iso-8601=seconds)" "$scene" "$code"
  } >> "$scene_log" 2>&1
  local render_code=$code
  if [ "$render_code" -ne 0 ]; then
    append_status "$scene" "render" "failed:$render_code"
    touch "$RUN/$scene.failed"
    return "$render_code"
  fi
  append_status "$scene" "render" "done"

  append_status "$scene" "metrics" "started"
  code=0
  {
    printf "[%s] metrics start %s gpu=%s\n" "$(date --iso-8601=seconds)" "$scene" "$GPU_ID"
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" metrics.py -m "$model"
    code=$?
    printf "[%s] metrics exit %s code=%s\n" "$(date --iso-8601=seconds)" "$scene" "$code"
  } >> "$scene_log" 2>&1
  local metrics_code=$code
  if [ "$metrics_code" -ne 0 ]; then
    append_status "$scene" "metrics" "failed:$metrics_code"
    touch "$RUN/$scene.failed"
    return "$metrics_code"
  fi
  append_status "$scene" "metrics" "done"
  touch "$RUN/$scene.done"
  append_status "$scene" "all" "done"
}

run_scene() {
  local scene="$1"
  local model="$OUT/$scene"
  local scene_log="$LOG/${scene}.log"

  if metrics_done "$model" || [ -f "$RUN/$scene.done" ]; then
    return 0
  fi

  if train_done "$model"; then
    append_status "$scene" "train" "skipped_existing_30k"
  else
    append_status "$scene" "train" "started"
    local code=0
    {
      printf "[%s] train start %s gpu=%s\n" "$(date --iso-8601=seconds)" "$scene" "$GPU_ID"
      CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" train.py \
        -s "$ROOT/$scene" \
        --eval \
        --data_branch rgbt \
        -m "$model" \
        --quiet
      code=$?
      printf "[%s] train exit %s code=%s\n" "$(date --iso-8601=seconds)" "$scene" "$code"
    } >> "$scene_log" 2>&1
    local train_code=$code
    if [ "$train_code" -ne 0 ]; then
      append_status "$scene" "train" "failed:$train_code"
      touch "$RUN/$scene.failed"
      return 0
    fi
    append_status "$scene" "train" "done"
  fi

  run_render_metrics "$scene" || return 0
}

while true; do
  picked=""
  for scene in "${SCENES[@]}"; do
    model="$OUT/$scene"
    if metrics_done "$model" || [ -f "$RUN/$scene.done" ] || [ -f "$RUN/$scene.failed" ]; then
      continue
    fi
    lock="$RUN/$scene.lock"
    if mkdir "$lock" 2>/dev/null; then
      picked="$scene"
      break
    fi
  done

  if [ -z "$picked" ]; then
    break
  fi

  run_scene "$picked"
  rmdir "$RUN/$picked.lock" 2>/dev/null || true
done

append_status "worker" "gpu$GPU_ID" "idle"
