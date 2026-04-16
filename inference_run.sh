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
OUTPUT="./output_image_zhuan224"
WEIGHT_MODEL="output/omnisvg_4b_20260409_174406/step_30000"
# "/home/bingxing2/home/scx7l3f/weiguang_zhang/project/OmniSVG-train/output/omnisvg_4b_20260410_215008/step_7500"




SAVE_PNG="true"
SAVE_ALL_CANDIDATES="true"
# 基底模型目录：按顺序使用第一个在磁盘上存在的路径（与 tokenization.yaml 多机写法一致）
# 覆盖自动选择：启动前执行 export BASE_MODEL=/你的/路径
BASE_MODEL_CANDIDATES=(
  "/data/phd23_weiguang_zhang/works/svg/qwen25vl3b"
  "/home/bingxing2/home/scx7l3f/weiguang_zhang/project/weights/qwen25vl3b"
)
if [ -z "${BASE_MODEL:-}" ]; then
  BASE_MODEL=""
  for _p in "${BASE_MODEL_CANDIDATES[@]}"; do
    if [ -e "$_p" ]; then
      BASE_MODEL="$_p"
      break
    fi
  done
  if [ -z "$BASE_MODEL" ]; then
    echo "Error: 未找到可用的 BASE_MODEL，已尝试：" >&2
    for _p in "${BASE_MODEL_CANDIDATES[@]}"; do echo "  - $_p" >&2; done
    exit 1
  fi
fi


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
echo "Model:  ${BASE_MODEL}"
echo "------------------------------------------------------------"
echo "Command: ${CMD[*]}"
echo "============================================================"
echo ""

exec "${CMD[@]}"
