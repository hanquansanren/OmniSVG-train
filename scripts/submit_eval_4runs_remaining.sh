#!/bin/bash
# 补提交四组 GRPO eval 任务（跳过已完成 / 已在队列）
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

export PARTITIONS="${PARTITIONS:-gpu4090 gpu40901t gpua8001t}"
export SKIP_STEPS=""
export SKIP_DONE=true

submit_run() {
  local run="$1"
  local steps="$2"
  echo ""
  echo "########## ${run} ##########"
  RUN="$run" STEPS="$steps" bash scripts/submit_eval_batch.sh
}

# j151686 / j151675: 由 SKIP_DONE 自动跳过已完成
submit_run grpo_4b_20261002_010149_j151686 "100"
submit_run grpo_4b_20261002_005027_j151675 "500 400 300 200 100 final"
submit_run grpo_4b_20261001_192246_j151511 "200 100 final"
submit_run grpo_4b_20261001_153531     "1000 900 800 700 600 500 400 300 200 100 final"
