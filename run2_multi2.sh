#!/bin/bash
# run2_multi.sh - OmniSVG Stage 2 multi-GPU training script
# Based on scripts/stage2_run.sh, extended with the multi-card launch logic from run_multi1.sh.

set -e

# ==============================================================================
# Configuration - MODIFY THESE SETTINGS
# ==============================================================================

# Model Configuration
# Options: "4B" (Qwen2.5-VL-3B based) or "8B" (Qwen2.5-VL-7B based)
MODEL_SIZE="4B"

# Enable Flash Attention 2 for faster training (recommended)
# Set to "true" or "false"
USE_FLASH_ATTN="true"

# torch.compile() 编译优化（PyTorch 2.x）
# Set to "false" to enable torch.compile (需要Triton 3.1.0)
# Set to "true" to disable torch.compile
DISABLE_TORCH_COMPILE="true"

# Disable P2P and IB for RTX 4000 series compatibility
# Set to "true" for RTX 4000 series, "false" for A100/H100/A6000
# 多卡 A100/A6000 建议保持 false 以启用 P2P/IB。
DISABLE_NCCL_P2P_IB="false"

# Number of GPUs to use
NUM_GPUS=4

# Batch size per GPU
BATCH_SIZE=1

# Maximum SVG sequence length # 输出SVG长度
MAX_SEQ_LENGTH=2048

# Stage 2 task switch
CODE_COMPLEMENT_TASK="true"

# Data directory (should contain: train_meta.csv, val_meta.csv, svg/, png/)
# 注意：如果不指定或留空，会使用 train_config 文件中的 data_dir
# DATA_DIR="/data/phd23_weiguang_zhang/works/svg/MMSVG-icon-sample"
DATA_DIR="/gpfs/work/int/weiguangzhang21/data/my_lis2_1"
# "/data/phd23_weiguang_zhang/works/svg/my_lis2_1_overfit20"
# "/data/phd23_weiguang_zhang/works/svg/MMSVG-icon-sample"

# Output directory for checkpoints and logs
OUTPUT_DIR="./output_stage2"

# Project name (leave empty for auto-generated name)
PROJECT_NAME="omnisvg_stage2_4b_$(date +%Y%m%d_%H%M%S)"

# Resume from checkpoint
# Options:
#   - "": Start from scratch
#   - "auto": Download and use official OmniSVG checkpoint
#   - "/path/to/checkpoint": Resume from specific checkpoint
RESUME_CHECKPOINT="/gpfs/work/int/weiguangzhang21/weights/pytorch_model.bin"
# "/data/phd23_weiguang_zhang/works/svg/models--OmniSVG--OmniSVG1.1_4B/snapshots/e4d03a89aaa28468520b45dc2541098102264d4e/pytorch_model.bin"
# "output/omnisvg_4b_20260214_205636/step_3000"
# "output/omnisvg_4b_20260209_021556/step_12000"
# "output/omnisvg_4b_20260408_020736/step_8000"

# Use HuggingFace datasets (set to "true" to auto-download)
USE_HF_DATA="false"

# HuggingFace datasets to use (only if USE_HF_DATA="true")
# Options: "illustration", "icon", or "illustration icon" (both)
HF_DATASETS="illustration icon"

# ==============================================================================
# Logging Configuration
# ==============================================================================

# Enable Weights & Biases for cloud visualization
# Set to "true" to enable remote access to training metrics
USE_WANDB="true"

# Weights & Biases project name (optional, defaults to "omnisvg-training")
WANDB_PROJECT="omnisvg-training"

# ==============================================================================
# Advanced Configuration
# ==============================================================================

# Config directory
CONFIG_DIR="./configs"

# Training config file name
# Options:
#   - "train_config_code_complement.yaml" (Stage 2 SVG code-complement)
#   - "train_config_low_memory.yaml" (optimized for low VRAM)
#   - Or create your own custom config file
TRAIN_CONFIG_FILE="train_config_cc_a100fat_fsdp.yaml"

# Accelerate config file (for DeepSpeed, FSDP, DDP, etc.)
# Faster FSDP path: shards gradients/optimizer state but avoids FULL_SHARD all-gather overhead.
ACCELERATE_CONFIG="configs/fsdp_config_speed.yaml"
# ACCELERATE_CONFIG="configs/ddp_config.yaml"
# ACCELERATE_CONFIG="configs/fsdp_config_stable.yaml"
# ACCELERATE_CONFIG="configs/fsdp_config_performance.yaml"
# ACCELERATE_CONFIG="configs/fsdp_config_sharded.yaml"
# ACCELERATE_CONFIG="configs/fsdp_config_minimal.yaml"
# ACCELERATE_CONFIG="configs/fsdp_config_transformer.yaml"

# Mixed precision training
MIXED_PRECISION="bf16"

# ==============================================================================
# Derived Settings (do not modify)
# ==============================================================================

# Auto-generate project name if not specified
if [ -z "$PROJECT_NAME" ]; then
    PROJECT_NAME="omnisvg_stage2_${MODEL_SIZE,,}_$(date +%Y%m%d_%H%M%S)"
fi

# Build command arguments
CMD_ARGS=""
CMD_ARGS+=" --model_size ${MODEL_SIZE}"
if [ -n "$DATA_DIR" ]; then
    CMD_ARGS+=" --data_dir ${DATA_DIR}"
fi
CMD_ARGS+=" --output_dir ${OUTPUT_DIR}"
CMD_ARGS+=" --project_name ${PROJECT_NAME}"
CMD_ARGS+=" --batch_size ${BATCH_SIZE}"
if [ -n "$MAX_SEQ_LENGTH" ]; then
    CMD_ARGS+=" --max_seq_length ${MAX_SEQ_LENGTH}"
fi
CMD_ARGS+=" --config_dir ${CONFIG_DIR}"
CMD_ARGS+=" --train_config_file ${TRAIN_CONFIG_FILE}"

# Flash attention flag
if [ "$USE_FLASH_ATTN" = "true" ]; then
    CMD_ARGS+=" --use_flash_attn"
else
    CMD_ARGS+=" --no_flash_attn"
fi

# Resume checkpoint
if [ -n "$RESUME_CHECKPOINT" ]; then
    CMD_ARGS+=" --resume_from_checkpoint ${RESUME_CHECKPOINT}"
fi

# HuggingFace data
if [ "$USE_HF_DATA" = "true" ]; then
    CMD_ARGS+=" --use_hf_data --datasets ${HF_DATASETS}"
fi

# Weights & Biases
if [ "$USE_WANDB" = "true" ]; then
    CMD_ARGS+=" --use_wandb"
    if [ -n "$WANDB_PROJECT" ]; then
        CMD_ARGS+=" --wandb_project ${WANDB_PROJECT}"
    fi
fi

# Build accelerate command
ACCELERATE_CMD="accelerate launch"
ACCELERATE_CMD+=" --num_processes ${NUM_GPUS}"
ACCELERATE_CMD+=" --mixed_precision ${MIXED_PRECISION}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-$((10000 + ${SLURM_JOB_ID:-0} % 50000))}"
ACCELERATE_CMD+=" --main_process_port ${MAIN_PROCESS_PORT}"

if [ -n "$ACCELERATE_CONFIG" ]; then
    ACCELERATE_CMD+=" --config_file ${ACCELERATE_CONFIG}"
fi

# ==============================================================================
# Print Configuration
# ==============================================================================

echo "============================================================"
echo "OmniSVG Stage 2 Multi-GPU Training"
echo "============================================================"
echo "Model Size:        ${MODEL_SIZE}"
echo "Flash Attention:   ${USE_FLASH_ATTN}"
echo "Disable NCCL P2P:  ${DISABLE_NCCL_P2P_IB}"
echo "Train Config File: ${TRAIN_CONFIG_FILE}"
echo "Code Complement:   ${CODE_COMPLEMENT_TASK}"
echo "Number of GPUs:    ${NUM_GPUS}"
echo "Batch Size:        ${BATCH_SIZE}"
if [ -n "$MAX_SEQ_LENGTH" ]; then
    echo "Max Seq Length:    ${MAX_SEQ_LENGTH}"
fi
if [ -n "$DATA_DIR" ]; then
    echo "Data Directory:    ${DATA_DIR}"
else
    echo "Data Directory:    (from config file)"
fi
echo "Output Directory:  ${OUTPUT_DIR}/${PROJECT_NAME}"
echo "Use HF Data:       ${USE_HF_DATA}"
echo "Use Wandb:         ${USE_WANDB}"
if [ "$USE_WANDB" = "true" ] && [ -n "$WANDB_PROJECT" ]; then
    echo "Wandb Project:     ${WANDB_PROJECT}"
fi
if [ -n "$RESUME_CHECKPOINT" ]; then
    echo "Resume From:       ${RESUME_CHECKPOINT}"
fi
echo "Accelerate Config: ${ACCELERATE_CONFIG:-default}"
echo "Main Port:         ${MAIN_PROCESS_PORT}"
echo "============================================================"
echo ""

# ==============================================================================
# Verify Data Directory
# ==============================================================================

# if [ "$USE_HF_DATA" = "false" ]; then
#     echo "Checking data directory: ${DATA_DIR}"
#
#     if [ ! -f "${DATA_DIR}/train_meta.csv" ]; then
#         echo "ERROR: ${DATA_DIR}/train_meta.csv not found!"
#         echo "Please prepare your data or use --use_hf_data to download from HuggingFace."
#         exit 1
#     fi
#
#     if [ ! -f "${DATA_DIR}/val_meta.csv" ]; then
#         echo "ERROR: ${DATA_DIR}/val_meta.csv not found!"
#         exit 1
#     fi
#
#     if [ ! -d "${DATA_DIR}/svg" ]; then
#         echo "ERROR: ${DATA_DIR}/svg directory not found!"
#         exit 1
#     fi
#
#     echo "Data directory verified."
#     echo ""
# fi

# ==============================================================================
# Run Training
# ==============================================================================

# Set NCCL environment variables
# For A100/A6000: explicitly enable P2P and IB to override accelerate's auto-detection
# For RTX 4000: disable P2P and IB for compatibility
if [ "$DISABLE_NCCL_P2P_IB" = "true" ]; then
    echo "Disabling NCCL P2P and IB (RTX 4000 series mode)"
    export NCCL_P2P_DISABLE=1
    export NCCL_IB_DISABLE=1
else
    echo "Enabling NCCL P2P and IB (A100/A6000 mode)"
    # Keep these unset unless you need to override cluster defaults explicitly.
    # export NCCL_P2P_DISABLE=0
    # export NCCL_IB_DISABLE=0
fi

# 不要在 Slurm 任务启动时 pkill train.py/accelerate；这会杀掉同一节点上的其它训练任务。

# torch.compile() 控制
if [ "$DISABLE_TORCH_COMPILE" = "true" ]; then
    export DISABLE_TORCH_COMPILE=1
    echo "torch.compile() disabled"
else
    unset DISABLE_TORCH_COMPILE
    echo "torch.compile() enabling"
    echo "需要Triton 3.1.0：pip install triton==3.1.0"
fi
echo ""

# CUDA调试环境变量 - 训练稳定后关闭以提升性能
# 如果遇到CUDA错误，取消注释下面几行以获取详细错误信息
# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_USE_CUDA_DSA=1
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

# 关键：设置NCCL超时时间 - PyTorch 2.4+ 使用新的环境变量
# 默认10分钟(600秒)对于FSDP checkpoint保存可能不够
#
# PyTorch 2.4+ 需要使用这些环境变量：
# export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
# export TORCH_NCCL_BLOCKING_WAIT=1
# export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# 旧版本兼容（PyTorch < 2.4）
# export NCCL_TIMEOUT=3600

echo "NCCL Configuration:"
echo "  - Heartbeat timeout: 3600 seconds (60 minutes, enable above if checkpoint save times out)"
echo "  - Blocking wait / async error handling: available as commented options"
echo ""

# 可选：NCCL性能调优（根据网络情况调整）
# export NCCL_DEBUG=INFO
# export NCCL_IB_TIMEOUT=50
# export NCCL_SOCKET_NTHREADS=8

echo "--- GPU Diagnostics ---"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "SLURM_JOB_GPUS:      ${SLURM_JOB_GPUS:-not set}"
echo "SLURM_NODELIST:      ${SLURM_NODELIST:-not set}"
if command -v nvidia-smi >/dev/null 2>&1; then
    echo ""
    echo "nvidia-smi (memory):"
    nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total --format=csv
    echo ""
    echo "nvidia-smi (processes):"
    nvidia-smi --query-compute-apps=pid,used_gpu_memory,name --format=csv 2>/dev/null || echo "(no compute processes)"
fi
echo "--- End GPU Diagnostics ---"
echo ""

echo "Starting stage 2 training..."
echo "Command: ${ACCELERATE_CMD} train.py ${CMD_ARGS}"
echo ""

${ACCELERATE_CMD} train.py ${CMD_ARGS}

echo ""
echo "Training completed!"
echo "Checkpoints saved to: ${OUTPUT_DIR}/${PROJECT_NAME}"
