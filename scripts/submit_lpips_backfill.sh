#!/bin/bash
# 为已有 eval_results（LPIPS=N/A）批量补跑 STAGE=eval
# 用法: bash scripts/submit_lpips_backfill.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

PARTITIONS=(${PARTITIONS:-gpu4090 gpu40901t gpua8001t})
part_idx=0
submitted=0 skipped=0 failed=0

tag_to_weight() {
  local tag="$1" run step
  if [[ "$tag" == grpo_4b_* ]]; then
    if [[ "$tag" == *_final ]]; then
      run="${tag%_final}"
      echo "output_grpo/${run}/final/pytorch_model.bin"
    elif [[ "$tag" =~ ^(.+)_step_([0-9]+)$ ]]; then
      echo "output_grpo/${BASH_REMATCH[1]}/step_${BASH_REMATCH[2]}/pytorch_model.bin"
    fi
  elif [[ "$tag" == omnisvg_* ]]; then
    if [[ "$tag" == *_best_model ]]; then
      run="${tag%_best_model}"
      echo "output_stage2/${run}/best_model/model.safetensors"
    elif [[ "$tag" =~ ^(.+)_step_([0-9]+)$ ]]; then
      echo "output_stage2/${BASH_REMATCH[1]}/step_${BASH_REMATCH[2]}/model.safetensors"
    fi
  elif [[ "$tag" == output_grpo_* ]]; then
    echo "output_grpo/${tag#output_grpo_}/model.safetensors"
  fi
}

for summary in ./eval_results/*/logs/summary_cand1.txt; do
  [ -f "$summary" ] || continue
  grep -q "N/A" "$summary" || { skipped=$((skipped+1)); continue; }

  tag=$(basename "$(dirname "$(dirname "$summary")")")
  pred="./eval_results/${tag}/task2_complete"
  [ -d "$pred" ] || { echo "[skip] $tag (无 task2_complete)"; skipped=$((skipped+1)); continue; }

  weight=$(tag_to_weight "$tag")
  [ -n "$weight" ] || { echo "[skip] $tag (无法解析权重路径)"; skipped=$((skipped+1)); continue; }
  [ -f "$weight" ] || { echo "[skip] $tag (权重不存在: $weight)"; skipped=$((skipped+1)); continue; }

  jname="lp$(echo -n "$tag" | md5sum | cut -c1-10)"
  if squeue -u "$USER" -n "$jname" -h 2>/dev/null | grep -q .; then
    echo "[skip] $tag (已在队列)"
    skipped=$((skipped+1))
    continue
  fi

  part="${PARTITIONS[$((part_idx % ${#PARTITIONS[@]}))]}"
  part_idx=$((part_idx + 1))

  echo "[submit] $tag  $part"
  if out=$(sbatch --partition="$part" --job-name="$jname" \
      --export=ALL,STAGE=eval,WEIGHT_MODEL="$weight" \
      scripts/eval_run.sh 2>&1); then
    echo "  -> $out"
    submitted=$((submitted + 1))
  else
    echo "  -> FAILED: $out" >&2
    failed=$((failed + 1))
  fi
done

echo ""
echo "完成: submitted=$submitted  skipped=$skipped  failed=$failed"
echo "查看: squeue -u \$USER | grep lpips"
