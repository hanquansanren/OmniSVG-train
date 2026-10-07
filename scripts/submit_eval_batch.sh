#!/bin/bash
#
# 批量并行提交 eval_run.sh（多卡、gpu4090 / gpu40901t 交替）
#
# 用法（在仓库根目录）:
#   bash scripts/submit_eval_batch.sh
#
# 自定义 run / 步数 / 分区:
#   RUN=grpo_4b_20260930_232159 STEPS="1000 900 800" SKIP_STEPS="300" \
#     bash scripts/submit_eval_batch.sh
#
#   PARTITIONS="gpu4090 gpu40901t gpua8001t" bash scripts/submit_eval_batch.sh
#
# 也可包一层 sbatch 在登录/计算节点触发提交:
#   sbatch --partition=gpu4090 -G 0 --wrap "bash scripts/submit_eval_batch.sh"

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

# ==============================================================================
# 配置（均可通过环境变量覆盖）
# ==============================================================================
# output_grpo/grpo_4b_20260930_230539
RUN="${RUN:-grpo_4b_20260929_021405}"
STEPS="${STEPS:-1000 900 800 700 600 500 400 300 200 100}"
SKIP_STEPS="${SKIP_STEPS:-}"
SKIP_DONE="${SKIP_DONE:-true}"
# 交替使用的分区列表
PARTITIONS="${PARTITIONS:-gpu4090 gpu40901t gpua8001t}"
# sbatch job 名前缀，默认取 run 时间戳部分
JOB_PREFIX="${JOB_PREFIX:-${RUN#grpo_4b_}}"

# ==============================================================================
# 辅助
# ==============================================================================

_should_skip() {
  local step="$1"
  local out_dir="./eval_results/${RUN}_step_${step}"
  [ "$step" = "final" ] && out_dir="./eval_results/${RUN}_final"

  for s in $SKIP_STEPS; do
    [ "$step" = "$s" ] && return 0
  done
  if [ "$SKIP_DONE" = "true" ] && [ -f "${out_dir}/logs/summary_cand1.txt" ]; then
    return 0
  fi
  return 1
}

_weight_path() {
  local step="$1"
  if [ "$step" = "final" ]; then
    echo "output_grpo/${RUN}/final/pytorch_model.bin"
  else
    echo "output_grpo/${RUN}/step_${step}/pytorch_model.bin"
  fi
}

_in_queue() {
  local step="$1"
  local jname="ev_${JOB_PREFIX}_${step}"
  squeue -u "$USER" -n "$jname" -h 2>/dev/null | grep -q .
}

# ==============================================================================
# 提交
# ==============================================================================

echo "============================================================"
echo "Eval Batch Submit  RUN=${RUN}"
echo "Steps:      ${STEPS}"
echo "Skip:       ${SKIP_STEPS}  SKIP_DONE=${SKIP_DONE}"
echo "Partitions: ${PARTITIONS}"
echo "Job prefix: ev_${JOB_PREFIX}_<step>"
echo "============================================================"

submitted=0
skipped=0
failed=0
part_idx=0
part_list=($PARTITIONS)
nparts=${#part_list[@]}

for step in $STEPS; do
  if _should_skip "$step"; then
    echo "[skip] step_${step} (已完成或在 SKIP_STEPS 中)"
    skipped=$((skipped + 1))
    continue
  fi
  if _in_queue "$step"; then
    echo "[skip] step_${step} (已在队列中)"
    skipped=$((skipped + 1))
    continue
  fi

  weight="$(_weight_path "$step")"
  if [ ! -f "$weight" ]; then
    echo "[warn] 权重不存在，跳过: $weight" >&2
    skipped=$((skipped + 1))
    continue
  fi

  part="${part_list[$((part_idx % nparts))]}"
  part_idx=$((part_idx + 1))
  jname="ev_${JOB_PREFIX}_${step}"

  echo "[submit] step_${step}  partition=${part}  job=${jname}"
  if out=$(sbatch --partition="$part" --job-name="$jname" \
      --export=ALL,WEIGHT_MODEL="$weight" \
      scripts/eval_run.sh 2>&1); then
    echo "  -> $out"
    submitted=$((submitted + 1))
  else
    echo "  -> FAILED: $out" >&2
    failed=$((failed + 1))
  fi
done

echo ""
echo "============================================================"
echo "提交完成: submitted=${submitted}  skipped=${skipped}  failed=${failed}"
echo "查看队列: squeue -u \$USER | grep ev_${JOB_PREFIX}"
echo "============================================================"
