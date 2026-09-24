#!/bin/bash
# grpo_run.sh - GRPO reinforcement learning on top of the stage-2 SFT checkpoint
#
# Recommended order:
#   1. REWARD_SMOKE=100 bash scripts/grpo_run.sh                       # reward pipeline only, no GPU
#   2. CUDA_VISIBLE_DEVICES=4 PROBE_SAMPLING=8 bash scripts/grpo_run.sh  # calibrate temperature
#   3. CUDA_VISIBLE_DEVICES=4 MAX_STEPS=100 bash scripts/grpo_run.sh     # loop smoke test
#   4. CUDA_VISIBLE_DEVICES=4 bash scripts/grpo_run.sh                   # full run
#   5. NUM_GPUS=8 bash scripts/grpo_run.sh                               # multi-GPU (A100/A800)
#
# Use CUDA_VISIBLE_DEVICES=4, same as scripts/stage2_run.sh: CUDA orders devices
# by capability rather than by PCI bus, so index 4 is the 48 GB A6000 while
# nvidia-smi lists it as index 2.  Partial fine-tuning of the 4B policy needs
# roughly 20 GB with the defaults below, so a 24 GB 4090 is tight but workable.

set -e

# ==============================================================================
# Configuration - MODIFY THESE SETTINGS
# ==============================================================================

# Model size: "4B" (Qwen2.5-VL-3B based) or "8B"
MODEL_SIZE="4B"

# Best SFT checkpoint to initialise the policy from.  This is the checkpoint
# that inference_run.sh currently points at.
SFT_CHECKPOINT="${SFT_CHECKPOINT:-output_stage2/omnisvg_4b_20260708_091120/step_14000}"

# Stage-2 data directory (needs svg/, png/, json/, train_meta.csv, val_meta.csv)
DATA_DIR="${DATA_DIR:-/data/phd23_weiguang_zhang/works/svg/my_lis2_2}"

# Output directory for checkpoints, metrics and completion dumps
OUTPUT_DIR="${OUTPUT_DIR:-./output_grpo}"

# ------------------------------------------------------------------------------
# GRPO hyper-parameters
# ------------------------------------------------------------------------------

# Completions sampled per prompt (the group size advantages are computed over)
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"

# Prompts (groups) accumulated into a single optimizer step
PROMPTS_PER_STEP="${PROMPTS_PER_STEP:-1}"

# 5e-7 to 1e-6; GRPO on a converged SFT policy needs a much smaller step than SFT
LEARNING_RATE="${LEARNING_RATE:-5e-6}"

# KL coefficient against the SFT reference; 0.01-0.05
BETA="${BETA:-0.02}"

# Must cover the ground-truth target length, otherwise completions get cut off
# before they can emit EOS.  Measured on my_lis2_2 (120 val samples) with
# `python train_grpo.py --measure-target-length 120`:
#   median 913, p90 1310, p95 1520, max 1872 tokens
#   1024 covers 62.5% of samples, 1536 covers 95.8%, 2048 covers 100%
# Note this is also why MAX_SEQ_LENGTH=1024 in scripts/stage2_run.sh truncates
# 37.5% of the SFT targets, so the policy never learned to stop near 1024: a
# temperature sweep showed the EOS rate stuck at 25-42% regardless of temperature.
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-2048}"

# Number of optimizer steps.  100-500 is the smoke-test range.
MAX_STEPS="${MAX_STEPS:-1000}"

# Only fine-tune the top N decoder layers (0 = every layer).  Keeps the vision
# tower and lower layers frozen, which is what makes this fit on one card.
TRAIN_LAST_LAYERS="${TRAIN_LAST_LAYERS:-8}"

# Reference policy for the KL term: "swap" (cheap and exact), "full", "none"
REF_MODE="${REF_MODE:-swap}"

# ------------------------------------------------------------------------------
# Sampling - GRPO needs enough spread for within-group advantages.
# Calibrated at MAX_COMPLETION_LENGTH=2048 (PROBE_SAMPLING, 16 prompts, A6000):
#   T=0.10 reward +0.43, grp_std 0.140, EOS 72%
#   T=0.30 reward +0.49, grp_std 0.085, EOS 81%  <- default
#   T=0.50 reward +0.51, grp_std 0.046, EOS 88%
# ------------------------------------------------------------------------------
TEMPERATURE="${TEMPERATURE:-0.30}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-0}"

# ------------------------------------------------------------------------------
# Logging / checkpointing
# ------------------------------------------------------------------------------
LOG_EVERY="${LOG_EVERY:-5}"
SAVE_EVERY="${SAVE_EVERY:-100}"
EVAL_EVERY="${EVAL_EVERY:-100}"
# One malformed completion scores -1 instead of ~+0.4, so with 16 samples a
# single flip moves eval reward by ~0.09.  64 keeps that under ~0.02.
EVAL_SAMPLES="${EVAL_SAMPLES:-64}"

USE_WANDB="${USE_WANDB:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-omnisvg-grpo}"

# Reward pipeline settings
CONSISTENCY_MODE="${CONSISTENCY_MODE:-iou}"
CANVAS_SIZE="${CANVAS_SIZE:-256}"

# Optional YAML overriding the reward weights (see configs/grpo_reward.yaml)
REWARD_CONFIG="${REWARD_CONFIG:-}"

# When set, only validate the reward pipeline on N samples and exit
REWARD_SMOKE="${REWARD_SMOKE:-0}"

# When set, only sweep sampling temperatures over N prompts and exit.
# Use this to choose TEMPERATURE and MAX_COMPLETION_LENGTH: a temperature that
# never produces EOS means every completion is truncated.
PROBE_SAMPLING="${PROBE_SAMPLING:-0}"
PROBE_TEMPERATURES="${PROBE_TEMPERATURES:-0.1,0.3,0.5,0.7,0.9}"

# Flash attention (set to "false" on GPUs without support)
USE_FLASH_ATTN="${USE_FLASH_ATTN:-true}"

# Data-parallel GPUs.  Above 1 the run goes through torchrun and every GPU
# samples its own prompts, so one optimizer step covers
# NUM_GPUS * PROMPTS_PER_STEP groups.  Diagnostics always run on one GPU.
NUM_GPUS="${NUM_GPUS:-1}"

# "auto" disables NCCL P2P/IB only on RTX 40-series cards, which need it; on
# A100/A800 it would force every gradient all-reduce around NVLink.
DISABLE_NCCL_P2P_IB="${DISABLE_NCCL_P2P_IB:-auto}"

# ==============================================================================
# Derived settings (do not modify)
# ==============================================================================

PYTHON="${PYTHON:-python}"

CMD_ARGS=""
CMD_ARGS+=" --model-size ${MODEL_SIZE}"
CMD_ARGS+=" --sft-checkpoint ${SFT_CHECKPOINT}"
CMD_ARGS+=" --data-dir ${DATA_DIR}"
CMD_ARGS+=" --output-dir ${OUTPUT_DIR}"
CMD_ARGS+=" --num-generations ${NUM_GENERATIONS}"
CMD_ARGS+=" --prompts-per-step ${PROMPTS_PER_STEP}"
CMD_ARGS+=" --learning-rate ${LEARNING_RATE}"
CMD_ARGS+=" --beta ${BETA}"
CMD_ARGS+=" --max-completion-length ${MAX_COMPLETION_LENGTH}"
CMD_ARGS+=" --max-steps ${MAX_STEPS}"
CMD_ARGS+=" --train-last-layers ${TRAIN_LAST_LAYERS}"
CMD_ARGS+=" --ref-mode ${REF_MODE}"
CMD_ARGS+=" --temperature ${TEMPERATURE}"
CMD_ARGS+=" --top-p ${TOP_P}"
CMD_ARGS+=" --top-k ${TOP_K}"
CMD_ARGS+=" --log-every ${LOG_EVERY}"
CMD_ARGS+=" --save-every ${SAVE_EVERY}"
CMD_ARGS+=" --eval-every ${EVAL_EVERY}"
CMD_ARGS+=" --eval-samples ${EVAL_SAMPLES}"
CMD_ARGS+=" --consistency-mode ${CONSISTENCY_MODE}"
CMD_ARGS+=" --canvas-size ${CANVAS_SIZE}"

if [ "$USE_FLASH_ATTN" != "true" ]; then
    CMD_ARGS+=" --no-flash-attn"
fi

if [ "$USE_WANDB" = "true" ]; then
    CMD_ARGS+=" --use-wandb --wandb-project ${WANDB_PROJECT}"
fi

if [ -n "$REWARD_CONFIG" ]; then
    CMD_ARGS+=" --reward-config ${REWARD_CONFIG}"
fi

if [ "$REWARD_SMOKE" -gt 0 ]; then
    CMD_ARGS+=" --reward-smoke ${REWARD_SMOKE}"
fi

if [ "$PROBE_SAMPLING" -gt 0 ]; then
    CMD_ARGS+=" --probe-sampling ${PROBE_SAMPLING} --probe-temperatures ${PROBE_TEMPERATURES}"
fi

if [ "$DISABLE_NCCL_P2P_IB" = "auto" ]; then
    if nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -q "RTX 40"; then
        DISABLE_NCCL_P2P_IB="true"
    else
        DISABLE_NCCL_P2P_IB="false"
    fi
fi
if [ "$DISABLE_NCCL_P2P_IB" = "true" ]; then
    export NCCL_P2P_DISABLE=1
    export NCCL_IB_DISABLE=1
fi
export DISABLE_TORCH_COMPILE=1

if [ "$REWARD_SMOKE" -gt 0 ] || [ "$PROBE_SAMPLING" -gt 0 ]; then
    NUM_GPUS=1
fi
if [ "$NUM_GPUS" -gt 1 ]; then
    LAUNCHER="${PYTHON} -m torch.distributed.run --standalone --nproc_per_node ${NUM_GPUS}"
else
    LAUNCHER="${PYTHON}"
fi

# ==============================================================================
# Print configuration
# ==============================================================================

echo "============================================================"
echo "OmniSVG GRPO Training"
echo "============================================================"
echo "Model Size:          ${MODEL_SIZE}"
echo "SFT Checkpoint:      ${SFT_CHECKPOINT}"
echo "Data Directory:      ${DATA_DIR}"
echo "Output Directory:    ${OUTPUT_DIR}"
echo "Num Generations:     ${NUM_GENERATIONS}"
echo "Learning Rate:       ${LEARNING_RATE}"
echo "Beta (KL):           ${BETA}"
echo "Max Completion Len:  ${MAX_COMPLETION_LENGTH}"
echo "Max Steps:           ${MAX_STEPS}"
echo "Trainable Layers:    ${TRAIN_LAST_LAYERS} (top)"
echo "Reference Mode:      ${REF_MODE}"
echo "Sampling:            T=${TEMPERATURE} top_p=${TOP_P} top_k=${TOP_K}"
echo "Consistency Mode:    ${CONSISTENCY_MODE}"
if [ "$REWARD_SMOKE" -gt 0 ]; then
echo "Mode:                REWARD SMOKE TEST (${REWARD_SMOKE} samples, no training)"
fi
if [ "$PROBE_SAMPLING" -gt 0 ]; then
echo "Mode:                SAMPLING PROBE (${PROBE_SAMPLING} prompts, T=${PROBE_TEMPERATURES})"
fi
echo "Num GPUs:            ${NUM_GPUS} (${NUM_GPUS} x ${PROMPTS_PER_STEP} prompt groups per step)"
echo "NCCL P2P/IB off:     ${DISABLE_NCCL_P2P_IB}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-(all)}"
echo "============================================================"
echo ""

echo "Command: ${LAUNCHER} train_grpo.py ${CMD_ARGS}"
echo ""

${LAUNCHER} train_grpo.py ${CMD_ARGS}

echo ""
echo "GRPO run completed!"
echo "Metrics:     ${OUTPUT_DIR}/<project>/grpo_metrics.jsonl"
echo "Completions: ${OUTPUT_DIR}/<project>/completions.jsonl"
