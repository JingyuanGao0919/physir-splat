#!/usr/bin/env bash
set -u

GPU_ID="${1:?usage: run_tinsd_ir_worker.sh GPU_ID [ADOPT_SCENE] [ADOPT_TRAIN_PID]}"
ADOPT_SCENE="${2:-}"
ADOPT_TRAIN_PID="${3:-}"

ROOT="/home/imglab/GJY/physir-splat-main/TI-NSD"
REPO="/home/imglab/GJY/physir-splat-main"
OUT="$REPO/output/ir_only_tinsd"
LOG="$REPO/logs/ir_only_tinsd"
RUN="$OUT/.run"
STATUS="$OUT/status.tsv"

SCENES=(
  UAV1 UAV2 UAV3 UAV4 UAV5 UAV6
  apples basketball_court bicycle bridge car chair corridor heated merge sitting
  soccer_goal standing tall_building wall
)

mkdir -p "$OUT" "$LOG" "$RUN"

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

  append_status "$scene" "render" "started"
  local code=0
  {
    printf "[%s] render start %s gpu=%s\n" "$(date --iso-8601=seconds)" "$scene" "$GPU_ID"
    CUDA_VISIBLE_DEVICES="$GPU_ID" python render.py -m "$model" --quiet
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
    CUDA_VISIBLE_DEVICES="$GPU_ID" python metrics.py -m "$model"
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
      CUDA_VISIBLE_DEVICES="$GPU_ID" python train.py -s "$ROOT/$scene" --eval --data_branch ir -m "$model" --quiet
      code=$?
      printf "[%s] train exit %s code=%s\n" "$(date --iso-8601=seconds)" "$scene" "$code"
    } >> "$scene_log" 2>&1
    local train_code=$code
    if [ "$train_code" -ne 0 ]; then
      append_status "$scene" "train" "failed:$train_code"
      return 0
    fi
    append_status "$scene" "train" "done"
  fi

  run_render_metrics "$scene" || return 0
}

adopt_scene() {
  local scene="$1"
  local pid="$2"
  local model="$OUT/$scene"
  local scene_log="$LOG/${scene}.log"
  local lock="$RUN/$scene.lock"

  mkdir "$lock" 2>/dev/null || return 0
  {
    printf "[%s] adopted in-flight train %s pid=%s gpu=%s\n" "$(date --iso-8601=seconds)" "$scene" "$pid" "$GPU_ID"
  } >> "$scene_log" 2>&1
  append_status "$scene" "train" "adopted_pid:$pid"

  while kill -0 "$pid" 2>/dev/null; do
    sleep 30
  done

  if train_done "$model"; then
    append_status "$scene" "train" "done_adopted"
    run_render_metrics "$scene" || true
  else
    append_status "$scene" "train" "adopt_missing_checkpoint"
    run_scene "$scene"
  fi
  rmdir "$lock" 2>/dev/null || true
}

if [ -n "$ADOPT_SCENE" ] && [ -n "$ADOPT_TRAIN_PID" ]; then
  adopt_scene "$ADOPT_SCENE" "$ADOPT_TRAIN_PID"
fi

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
