#!/bin/bash
# run.sh - OmniSVG Training Script
# Supports both 4B and 8B models with configurable options

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

# Disable P2P and IB for RTX 4000 series compatibility
# Set to "true" for RTX 4000 series, "false" for A100/H100
DISABLE_NCCL_P2P_IB="true" 

# Number of GPUs to use
NUM_GPUS=4

# Batch size per GPU
BATCH_SIZE=1

# Maximum SVG sequence length
# MAX_SEQ_LENGTH=2048

# Data directory (should contain: train_meta.csv, val_meta.csv, svg/, png/)
# 注意：如果不指定或留空，会使用 train_config 文件中的 data_dir
# DATA_DIR="/data/phd23_weiguang_zhang/works/svg/MMSVG-icon-sample"
DATA_DIR="/home/bingxing2/home/scx7l3f/weiguang_zhang/project/weights/my_zhuan"

# Output directory for checkpoints and logs
OUTPUT_DIR="./output"

# Project name (leave empty for auto-generated name)
PROJECT_NAME="omnisvg_4b_$(date +%Y%m%d_%H%M%S)"

# Resume from checkpoint
# Options:
#   - "": Start from scratch
#   - "auto": Download and use official OmniSVG checkpoint
#   - "/path/to/checkpoint": Resume from specific checkpoint
RESUME_CHECKPOINT=""
# "output/omnisvg_4b_20260214_205636/step_3000"
# "output/omnisvg_4b_20260209_021556/step_12000"
# "/data/phd23_weiguang_zhang/works/svg/models--OmniSVG--OmniSVG1.1_4B/snapshots/e4d03a89aaa28468520b45dc2541098102264d4e/pytorch_model.bin"

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
#   - "train_config.yaml" (standard configuration)
#   - "train_config_low_memory.yaml" (optimized for low VRAM)
#   - Or create your own custom config file
TRAIN_CONFIG_FILE="train_config_zhuan.yaml"

# Accelerate config file (for DeepSpeed, FSDP, etc.)
# Leave empty for default settings 多卡训练时需要配置
# ACCELERATE_CONFIG="configs/zero_stage2.yaml"         # DeepSpeed ZeRO Stage 2 (与PyTorch 2.5.0不兼容)
# ACCELERATE_CONFIG="configs/fsdp_config.yaml"         # FSDP SIZE_BASED (与PyTorch 2.5.0的DTensor有冲突)
# ACCELERATE_CONFIG="configs/ddp_config.yaml"          # DDP (最稳定，但显存占用高)
ACCELERATE_CONFIG="configs/fsdp_config_sharded.yaml"  # FSDP TRANSFORMER_BASED + Activation Checkpointing (显存优化) 
# fsdp_config_sharded.yaml
# fsdp_config_transformer.yaml

# Mixed precision training
MIXED_PRECISION="bf16"

# ==============================================================================
# Derived Settings (do not modify)
# ==============================================================================

# Auto-generate project name if not specified
if [ -z "$PROJECT_NAME" ]; then
    PROJECT_NAME="omnisvg_${MODEL_SIZE,,}_$(date +%Y%m%d_%H%M%S)"
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

if [ -n "$ACCELERATE_CONFIG" ]; then
    ACCELERATE_CMD+=" --config_file ${ACCELERATE_CONFIG}"
fi

# ==============================================================================
# Print Configuration
# ==============================================================================

echo "============================================================"
echo "OmniSVG Training"
echo "============================================================"
echo "Model Size:        ${MODEL_SIZE}"
echo "Flash Attention:   ${USE_FLASH_ATTN}"
echo "Disable NCCL P2P:  ${DISABLE_NCCL_P2P_IB}"
echo "Train Config File: ${TRAIN_CONFIG_FILE}"
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
echo "============================================================"
echo ""

# ==============================================================================
# Verify Data Directory
# ==============================================================================

# if [ "$USE_HF_DATA" = "false" ]; then
#     echo "Checking data directory: ${DATA_DIR}"
    
#     if [ ! -f "${DATA_DIR}/train_meta.csv" ]; then
#         echo "ERROR: ${DATA_DIR}/train_meta.csv not found!"
#         echo "Please prepare your data or use --use_hf_data to download from HuggingFace."
#         exit 1
#     fi
    
#     if [ ! -f "${DATA_DIR}/val_meta.csv" ]; then
#         echo "ERROR: ${DATA_DIR}/val_meta.csv not found!"
#         exit 1
#     fi
    
#     if [ ! -d "${DATA_DIR}/svg" ]; then
#         echo "ERROR: ${DATA_DIR}/svg directory not found!"
#         exit 1
#     fi
    
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
    export NCCL_P2P_DISABLE=0
    export NCCL_IB_DISABLE=0
fi

# ⭐⭐⭐ 重要：清理GPU状态，防止CUDA错误 ⭐⭐⭐
echo "🧹 Cleaning GPU state..."
# 杀死可能残留的Python训练进程
pkill -9 -f "train.py" || true
pkill -9 -f "accelerate" || true
# 等待进程完全退出
sleep 2
echo "✓ GPU cleanup completed"
echo ""

# ⭐ CUDA调试环境变量 - 帮助定位CUDA错误
export CUDA_LAUNCH_BLOCKING=1                         # 同步CUDA操作，获取准确错误堆栈
export TORCH_USE_CUDA_DSA=1                           # 启用设备端断言
export TORCH_DISTRIBUTED_DEBUG=DETAIL                 # 详细调试信息

# ⭐ 关键：设置NCCL超时时间 - PyTorch 2.4+ 使用新的环境变量
# 默认10分钟(600秒)对于FSDP checkpoint保存可能不够
# 
# PyTorch 2.4+ 需要使用这些环境变量（以毫秒为单位）：
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600         # 心跳超时：60分钟
export TORCH_NCCL_BLOCKING_WAIT=1                     # 使用阻塞等待（更稳定）
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1              # 异步错误处理

# 旧版本兼容（PyTorch < 2.4）
export NCCL_TIMEOUT=3600

echo "⚙️  CUDA & NCCL Configuration:"
echo "  - CUDA_LAUNCH_BLOCKING: 1 (synchronous mode for debugging)"
echo "  - TORCH_USE_CUDA_DSA: 1 (device-side assertions)"
echo "  - NCCL Heartbeat timeout: 3600 seconds (60 minutes)"
echo "  - NCCL Blocking wait: enabled"
echo "  - NCCL Async error handling: enabled"
echo ""

# 可选：NCCL性能调优（根据网络情况调整）
# export NCCL_DEBUG=INFO              # 详细日志（调试时启用）
# export NCCL_IB_TIMEOUT=50           # InfiniBand超时（如果使用IB）
# export NCCL_SOCKET_NTHREADS=8       # Socket线程数

echo "Starting training..."
echo "Command: ${ACCELERATE_CMD} train.py ${CMD_ARGS}"
echo ""

${ACCELERATE_CMD} train.py ${CMD_ARGS}

echo ""
echo "Training completed!"
echo "Checkpoints saved to: ${OUTPUT_DIR}/${PROJECT_NAME}"
