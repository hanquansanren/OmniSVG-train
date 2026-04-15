#!/bin/bash
# OmniSVG 推理脚本（image-to-svg）
# 等效于:
#   python inference.py --task image-to-svg --input ./backup/examples_zhuan3 \
#     --output ./output_image_zhuan --save-png --save-all-candidates

set -euo pipefail

# cd "$(dirname "$0")"

# ==============================================================================
# 配置（按需修改）
# ==============================================================================

PYTHON="${PYTHON:-python}"

TASK="image-to-svg"
INPUT="./backup/examples_zhuan3"
OUTPUT="./output_image_zhuan1"

SAVE_PNG="true"
SAVE_ALL_CANDIDATES="true"
BASE_MODEL="/home/bingxing2/home/scx7l3f/weiguang_zhang/project/weights/qwen25vl3b"
WEIGHT_MODEL="/home/bingxing2/home/scx7l3f/weiguang_zhang/project/OmniSVG-train/output/omnisvg_4b_20260410_215008/step_7500"
# 追加传给 inference.py 的参数，例如: EXTRA_ARGS=(--verbose --model-size 4B)
EXTRA_ARGS=()

# ==============================================================================
# 构建命令
# ==============================================================================

CMD=( "$PYTHON" ./inference.py
  --task "$TASK"
  --input "$INPUT"
  --output "$OUTPUT"
  --model-path "$BASE_MODEL"
  --weight-path "$WEIGHT_MODEL"
)

if [ "$SAVE_PNG" = "true" ]; then
  CMD+=( --save-png )
fi
if [ "$SAVE_ALL_CANDIDATES" = "true" ]; then
  CMD+=( --save-all-candidates )
fi

if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
  CMD+=( "${EXTRA_ARGS[@]}" )
fi

echo "============================================================"
echo "OmniSVG Inference"
echo "============================================================"
echo "Task:   ${TASK}"
echo "Input:  ${INPUT}"
echo "Output: ${OUTPUT}"
echo "------------------------------------------------------------"
echo "Command: ${CMD[*]}"
echo "============================================================"
echo ""

exec "${CMD[@]}"
