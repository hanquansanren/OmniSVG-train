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
NUM_GPUS=1

# Batch size per GPU
BATCH_SIZE=1

# Maximum SVG sequence length
MAX_SEQ_LENGTH=2048

# Data directory (should contain: train_meta.csv, val_meta.csv, svg/, png/)
DATA_DIR="/data/phd23_weiguang_zhang/works/svg/MMSVG-icon-sample"

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

# Use HuggingFace datasets (set to "true" to auto-download)
USE_HF_DATA="false"

# HuggingFace datasets to use (only if USE_HF_DATA="true")
# Options: "illustration", "icon", or "illustration icon" (both)
HF_DATASETS="illustration icon"

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
TRAIN_CONFIG_FILE="train_config.yaml"

# Accelerate config file (for DeepSpeed, FSDP, etc.)
# Leave empty for default settings
ACCELERATE_CONFIG=""

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
CMD_ARGS+=" --data_dir ${DATA_DIR}"
CMD_ARGS+=" --output_dir ${OUTPUT_DIR}"
CMD_ARGS+=" --project_name ${PROJECT_NAME}"
CMD_ARGS+=" --batch_size ${BATCH_SIZE}"
CMD_ARGS+=" --max_seq_length ${MAX_SEQ_LENGTH}"
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
echo "Max Seq Length:    ${MAX_SEQ_LENGTH}"
echo "Data Directory:    ${DATA_DIR}"
echo "Output Directory:  ${OUTPUT_DIR}/${PROJECT_NAME}"
echo "Use HF Data:       ${USE_HF_DATA}"
if [ -n "$RESUME_CHECKPOINT" ]; then
echo "Resume From:       ${RESUME_CHECKPOINT}"
fi
echo "============================================================"
echo ""

# ==============================================================================
# Verify Data Directory
# ==============================================================================

if [ "$USE_HF_DATA" = "false" ]; then
    echo "Checking data directory: ${DATA_DIR}"
    
    if [ ! -f "${DATA_DIR}/train_meta.csv" ]; then
        echo "ERROR: ${DATA_DIR}/train_meta.csv not found!"
        echo "Please prepare your data or use --use_hf_data to download from HuggingFace."
        exit 1
    fi
    
    if [ ! -f "${DATA_DIR}/val_meta.csv" ]; then
        echo "ERROR: ${DATA_DIR}/val_meta.csv not found!"
        exit 1
    fi
    
    if [ ! -d "${DATA_DIR}/svg" ]; then
        echo "ERROR: ${DATA_DIR}/svg directory not found!"
        exit 1
    fi
    
    echo "Data directory verified."
    echo ""
fi

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

echo "Starting training..."
echo "Command: ${ACCELERATE_CMD} train.py ${CMD_ARGS}"
echo ""

${ACCELERATE_CMD} train.py ${CMD_ARGS}

echo ""
echo "Training completed!"
echo "Checkpoints saved to: ${OUTPUT_DIR}/${PROJECT_NAME}"
