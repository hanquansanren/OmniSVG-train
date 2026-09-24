#!/bin/bash
# Wait until A6000 (nvidia-smi index 2) has enough free memory, then run GRPO smoke test.
set -euo pipefail

cd "$(dirname "$0")/.."
source /data/phd23_weiguang_zhang/conda/CONDA35/etc/profile.d/conda.sh
conda activate omni_test

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2
export USE_WANDB=false
export MAX_STEPS=100
export MAX_COMPLETION_LENGTH=2048
export TEMPERATURE=0.30
export SFT_CHECKPOINT="output_stage2/omnisvg_4b_20260708_091120/step_14000"
export SAVE_EVERY=50
export LOG_EVERY=5
export EVAL_EVERY=100

LOG="/tmp/grpo_smoke_100.log"
MIN_FREE_MIB=40000
POLL_SEC=30

echo "[$(date)] Waiting for GPU 2 (A6000) >= ${MIN_FREE_MIB} MiB free..." | tee "$LOG"

while true; do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 2 | tr -d ' ')
  echo "[$(date)] GPU2 free=${FREE} MiB" | tee -a "$LOG"
  if [ "$FREE" -ge "$MIN_FREE_MIB" ]; then
    echo "[$(date)] GPU ready, starting GRPO smoke test..." | tee -a "$LOG"
    break
  fi
  sleep "$POLL_SEC"
done

bash scripts/grpo_run.sh 2>&1 | tee -a "$LOG"
echo "[$(date)] Done, exit=${PIPESTATUS[0]}" | tee -a "$LOG"
