#!/bin/bash
# 1) 轮询 logs / logs_*，将两组 stage2 checkpoint 的 eval 汇总写入 CSV
# 2) 发现 output_stage2 下新 checkpoint 且尚无指标时，自动 sbatch eval_run.sh
#    分区在 gpu4090 / gpu40901t 中选队列更空的一个
#
# 用法:
#   bash scripts/watch_stage2_eval_logs_to_csv.sh
#   nohup bash scripts/watch_stage2_eval_logs_to_csv.sh >> logs/watch_stage2_csv.log 2>&1 &
#
# 环境变量:
#   RUNS              两组 run 目录名
#   POLL_SEC          轮询间隔（默认 120）
#   AUTO_SUBMIT_EVAL  true/false，是否自动提交 eval（默认 true）
#   PARTITION_CANDIDATES  默认 "gpu4090 gpu40901t"
#   MAX_SUBMIT_PER_POLL   每轮最多提交几个 eval（默认 4，防 QOS 打满）
#   SKIP_STEPS        默认 12500
#   MIN_AUTO_EVAL_STEP  仅自动提交 step>=该值（默认 130000；0=不限制）
#   STAGE2_EVAL_CSV   已测 checkpoint 汇总表（默认 metrics/stage2_checkpoint_eval_summary.csv）
#   CKPT_MIN_BYTES    model.safetensors 最小字节（默认 1e9，约 1GB）
#   CKPT_STABLE_POLLS 连续几轮 poll 内文件大小不变才提交（默认 2）
#   CKPT_MIN_AGE_SEC  距 mtime 至少多少秒（默认 60，防刚落盘）
#   TASKS STAGE CANDIDATE  传给 eval_run.sh（默认 complete / all / 1）
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs metrics

# shellcheck source=scripts/lib_stage2_eval_watch.sh
source "$REPO_ROOT/scripts/lib_stage2_eval_watch.sh"

export POLL_SEC="${POLL_SEC:-120}"
export RUNS="${RUNS:-omnisvg_stage2_4b_20261020_230852 omnisvg_stage2_4b_20261020_231348}"
export AUTO_SUBMIT_EVAL="${AUTO_SUBMIT_EVAL:-true}"
export PARTITION_CANDIDATES="${PARTITION_CANDIDATES:-gpu4090 gpu40901t}"
export MAX_SUBMIT_PER_POLL="${MAX_SUBMIT_PER_POLL:-4}"
export SKIP_STEPS="${SKIP_STEPS:-12500}"
export MIN_AUTO_EVAL_STEP="${MIN_AUTO_EVAL_STEP:-130000}"

CSV="${CSV:-$REPO_ROOT/metrics/stage2_checkpoint_eval_summary.csv}"
export STAGE2_EVAL_CSV="$CSV"
SYNC_PY="$REPO_ROOT/scripts/sync_stage2_eval_logs_csv.py"

stage2_log "watch 启动 CSV=${CSV} AUTO_SUBMIT_EVAL=${AUTO_SUBMIT_EVAL} MIN_AUTO_EVAL_STEP=${MIN_AUTO_EVAL_STEP} RUNS=${RUNS} POLL_SEC=${POLL_SEC}"

last_csv_rows=-1

while true; do
  if out=$(python3 "$SYNC_PY" --repo-root "$REPO_ROOT" --csv "$CSV" --runs $RUNS 2>&1); then
    rows=$(echo "$out" | sed -n 's/^Wrote \([0-9]*\) rows.*/\1/p')
    rows="${rows:-?}"
    if [ "$rows" != "$last_csv_rows" ]; then
      stage2_log "[csv] $out"
      last_csv_rows=$rows
    fi
  else
    stage2_log "[csv][error] $out"
  fi

  if [ "$AUTO_SUBMIT_EVAL" = "true" ]; then
    stage2_submit_pending_checkpoints || true
  fi

  sleep "$POLL_SEC"
done
