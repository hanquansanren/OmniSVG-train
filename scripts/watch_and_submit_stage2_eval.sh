#!/bin/bash
# 等待指定 eval 队列清空后，自动提交 stage2 eval（可反复重试直至 QOS 允许）
#
# 默认：等 grpo j166835 的 gpu4090 eval 全部结束，再提交 230852 / 231348（无 12500、无 best）
#
# 用法:
#   nohup bash scripts/watch_and_submit_stage2_eval.sh >> logs/watch_stage2_eval.log 2>&1 &
#   WAIT_JOB_PATTERN=ev_20261021 bash scripts/watch_and_submit_stage2_eval.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

WAIT_JOB_PATTERN="${WAIT_JOB_PATTERN:-ev_20261021_180817}"
POLL_SEC="${POLL_SEC:-120}"
PARTITION="${PARTITION:-gpu4090}"
STEPS="${STEPS:-27500 25000 22500 20000 17500 15000}"
SKIP_STEPS="${SKIP_STEPS:-12500}"
RUNS="${RUNS:-omnisvg_stage2_4b_20261020_230852 omnisvg_stage2_4b_20261020_231348}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

_wait_queue_clear() {
  while true; do
    if squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -q "$WAIT_JOB_PATTERN"; then
      n=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -c "$WAIT_JOB_PATTERN" || true)
      log "等待中: 仍有 ${n} 个 job 匹配 '${WAIT_JOB_PATTERN}'"
      sleep "$POLL_SEC"
    else
      log "触发条件满足: 队列中已无 '${WAIT_JOB_PATTERN}'"
      return 0
    fi
  done
}

_should_skip_step() {
  local run="$1" step="$2"
  for s in $SKIP_STEPS; do
    [ "$step" = "$s" ] && return 0
  done
  local ev="$REPO_ROOT/eval_results/${run}_step_${step}/logs/summary_cand1.txt"
  [ -f "$ev" ] && return 0
  return 1
}

_submit_one() {
  local run="$1" step="$2"
  local tag="${run#omnisvg_stage2_4b_}"
  local w="output_stage2/${run}/step_${step}/model.safetensors"
  local jname="ev_${tag}_${step}"

  if [ ! -f "$w" ]; then
    log "[skip] 权重不存在: $w"
    return 0
  fi
  if _in_queue "$jname"; then
    log "[skip] 已在队列: $jname"
    return 0
  fi

  while true; do
    if out=$(sbatch --partition="$PARTITION" --job-name="$jname" \
        --export=ALL,WEIGHT_MODEL="$w" scripts/eval_run.sh 2>&1); then
      log "[submit] $run step_${step} -> $out"
      return 0
    fi
    if echo "$out" | grep -q QOSMaxSubmitJobPerUserLimit; then
      log "[wait] QOS 提交上限，${POLL_SEC}s 后重试: $jname"
      sleep "$POLL_SEC"
    else
      log "[error] $run step_${step}: $out"
      return 1
    fi
  done
}

_in_queue() {
  local jname="$1"
  squeue -u "$USER" -n "$jname" -h 2>/dev/null | grep -q .
}

log "监控启动 WAIT_JOB_PATTERN=${WAIT_JOB_PATTERN} PARTITION=${PARTITION}"
log "RUNS=${RUNS}"
log "STEPS=${STEPS} SKIP_STEPS=${SKIP_STEPS}"

_wait_queue_clear

for run in $RUNS; do
  for step in $STEPS; do
    if _should_skip_step "$run" "$step"; then
      log "[skip] ${run} step_${step} (已完成或在 SKIP_STEPS)"
      continue
    fi
    _submit_one "$run" "$step"
  done
done

log "全部待提交 checkpoint 已处理完毕。"
