#!/bin/bash
#SBATCH --job-name=svg_eval
#SBATCH --partition=gpu40901t
#SBATCH -N 1
#SBATCH --qos=16gpus
#SBATCH -G 1
#SBATCH --output=logs/eval_%j.out
# OmniSVG 测试集推理 + 指标评测（SSIM / LPIPS / MSE）
#   任务1 img2svg：        ./eval/eval_task1   (my_zhuan4 验证集抽样)
#   任务2 code-complement：./eval/eval_task2   (my_lis2_2 验证集抽样)
#
# 用法（在仓库根目录下）:
#   sbatch scripts/eval_run.sh                                  # 集群提交
#   bash   scripts/eval_run.sh                                  # 本地直接运行
#   WEIGHT_MODEL=output_grpo/xxx/step_800/pytorch_model.bin sbatch scripts/eval_run.sh
#   TASKS="complete" STAGE=eval CANDIDATE=best bash scripts/eval_run.sh   # 只重新评测任务2
#
# 注意: sbatch 的 --output 目录需事先存在（mkdir -p logs）。

set -euo pipefail

# ==============================================================================
# 配置（均可通过同名环境变量覆盖）
# ==============================================================================

# 仓库根目录：sbatch 会把脚本拷到 spool 目录，因此优先用提交目录
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
cd "$REPO_ROOT"
if [ ! -f ./inference.py ] || [ ! -f ./metrics/eval_svg_tasks.py ]; then
  echo "Error: $REPO_ROOT 不是 OmniSVG-train 仓库根目录，请在根目录提交或设置 REPO_ROOT" >&2
  exit 1
fi

# 可选：激活 conda 环境（为空则使用当前环境）
module load anaconda3
source activate svg2
# CONDA_ENV="${CONDA_ENV:-}"
# if [ -n "$CONDA_ENV" ]; then
#   source "$(conda info --base)/etc/profile.d/conda.sh"
#   conda activate "$CONDA_ENV"
# fi

PYTHON="${PYTHON:-python}"
# 评测需要 cairosvg / scikit-image / lpips；若与推理环境不同可单独指定
EVAL_PYTHON="${EVAL_PYTHON:-$PYTHON}"

# 要运行的任务：img2svg / complete，空格分隔
TASKS="${TASKS:-complete}"
# all = 推理 + 评测；infer = 只推理；eval = 只评测（复用已有推理结果）
STAGE="${STAGE:-all}"

TEST_DIR_IMG2SVG="${TEST_DIR_IMG2SVG:-./eval/eval_task1}"
TEST_DIR_COMPLETE="${TEST_DIR_COMPLETE:-./eval/eval_task2}"

WEIGHT_MODEL="${WEIGHT_MODEL:-output_grpo/20260928_022117_5000/model.safetensors}"
# output_grpo/20260928_022117_5000/model.safetensors
# output_stage2/omnisvg_stage2_4b_20260926_031414/step_15000/model.safetensors
# output_grpo/grpo_4b_20260930_224442/step_300/pytorch_model.bin
# output_grpo/grpo_4b_20260930_230539/step_300/pytorch_model.bin
# output_grpo/grpo_4b_20260930_232159/step_300/pytorch_model.bin
# output_grpo/grpo_4b_20261001_002150/step_300/pytorch_model.bin
# img2svg 与 complete 使用不同权重时分别指定，默认都用 WEIGHT_MODEL
WEIGHT_IMG2SVG="${WEIGHT_IMG2SVG:-$WEIGHT_MODEL}"
WEIGHT_COMPLETE="${WEIGHT_COMPLETE:-$WEIGHT_MODEL}"

# 结果根目录，默认按权重路径命名，例如 ./eval_results/grpo_4b_20260928_223915_step_800
_weight_tag() {
  local p="${1%/}"
  case "$p" in
    *.bin|*.safetensors|*.pt|*.pth) p="$(dirname "$p")" ;;
  esac
  echo "$(basename "$(dirname "$p")")_$(basename "$p")"
}
OUTPUT_ROOT="${OUTPUT_ROOT:-./eval_results/$(_weight_tag "$WEIGHT_MODEL")}"

# 推理参数
SAVE_PNG="${SAVE_PNG:-true}"
SAVE_ALL_CANDIDATES="${SAVE_ALL_CANDIDATES:-true}"
NUM_CANDIDATES="${NUM_CANDIDATES:-}"          # 为空则用 inference.py 默认值
SKELETON_COT="${SKELETON_COT:-true}"          # 仅对 code-complement 生效
# 追加传给 inference.py 的参数，例如: EXTRA_INFER_ARGS="--temperature 0.3 --verbose"
EXTRA_INFER_ARGS="${EXTRA_INFER_ARGS:-}"

# 评测参数
CANDIDATE="${CANDIDATE:-1}"                   # 1 / 2 / ... / best
BEST_BY="${BEST_BY:-lpips}"                   # CANDIDATE=best 时的挑选指标
MISSING="${MISSING:-fallback}"                # fallback / skip
PRED_SOURCE="${PRED_SOURCE:-overlay}"         # 任务2: overlay / combined
SAVE_VIS="${SAVE_VIS:-true}"
NO_LPIPS="${NO_LPIPS:-false}"
EXTRA_EVAL_ARGS="${EXTRA_EVAL_ARGS:-}"

# 基底模型目录：按顺序使用第一个在磁盘上存在的路径
BASE_MODEL_CANDIDATES=(
  "/data/phd23_weiguang_zhang/works/svg/qwen25vl3b"
  "/home/bingxing2/home/scx7l3f/weiguang_zhang/project/weights/qwen25vl3b"
  "/gpfs/work/int/weiguangzhang21/weights/qwen25vl3b"
)
if [ -z "${BASE_MODEL:-}" ]; then
  BASE_MODEL=""
  for _p in "${BASE_MODEL_CANDIDATES[@]}"; do
    if [ -e "$_p" ]; then
      BASE_MODEL="$_p"
      break
    fi
  done
  if [ -z "$BASE_MODEL" ] && [ "$STAGE" != "eval" ]; then
    echo "Error: 未找到可用的 BASE_MODEL，已尝试：" >&2
    for _p in "${BASE_MODEL_CANDIDATES[@]}"; do echo "  - $_p" >&2; done
    exit 1
  fi
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "OmniSVG Eval"
echo "============================================================"
echo "Repo:         $REPO_ROOT"
echo "Host:         $(hostname)  Job: ${SLURM_JOB_ID:-local}"
echo "Tasks:        $TASKS   Stage: $STAGE"
echo "Base model:   ${BASE_MODEL:-N/A}"
echo "Weight (1):   $WEIGHT_IMG2SVG"
echo "Weight (2):   $WEIGHT_COMPLETE"
echo "Output root:  $OUTPUT_ROOT"
echo "Candidate:    $CANDIDATE   Missing: $MISSING"
echo "============================================================"

# ==============================================================================
# 推理
# ==============================================================================

run_infer() {
  local task="$1" test_dir="$2" pred_dir="$3" weight="$4"
  local infer_task input
  if [ "$task" = "img2svg" ]; then
    infer_task="image-to-svg"
    input="$test_dir/png"
  else
    infer_task="code-complement"
    input="$test_dir"
  fi

  local cmd=( "$PYTHON" ./inference.py
    --task "$infer_task"
    --input "$input"
    --output "$pred_dir"
    --model-path "$BASE_MODEL"
    --weight-path "$weight"
    --save-svg
  )
  [ "$SAVE_PNG" = "true" ] && cmd+=( --save-png )
  [ "$SAVE_ALL_CANDIDATES" = "true" ] && cmd+=( --save-all-candidates )
  [ -n "$NUM_CANDIDATES" ] && cmd+=( --num-candidates "$NUM_CANDIDATES" )
  if [ "$task" = "complete" ]; then
    cmd+=( --use-train-tokenizer )
    [ "$SKELETON_COT" = "true" ] && cmd+=( --skeleton-cot )
  fi
  # shellcheck disable=SC2206
  [ -n "$EXTRA_INFER_ARGS" ] && cmd+=( $EXTRA_INFER_ARGS )

  echo ""
  echo "[infer:$task] ${cmd[*]}"
  "${cmd[@]}" 2>&1 | tee "$LOG_DIR/infer_${task}.log"
}

# ==============================================================================
# 评测
# ==============================================================================

run_eval() {
  local task="$1" test_dir="$2" pred_dir="$3"
  local metrics_dir="$pred_dir/metrics_cand${CANDIDATE}"
  [ "$task" = "complete" ] && metrics_dir="${metrics_dir}_${PRED_SOURCE}"

  local cmd=( "$EVAL_PYTHON" ./metrics/eval_svg_tasks.py "$task"
    --test_dir "$test_dir"
    --pred_dir "$pred_dir"
    --out_dir "$metrics_dir"
    --candidate "$CANDIDATE"
    --best_by "$BEST_BY"
    --missing "$MISSING"
  )
  [ "$task" = "complete" ] && cmd+=( --pred_source "$PRED_SOURCE" )
  [ "$SAVE_VIS" = "true" ] && cmd+=( --save_vis )
  [ "$NO_LPIPS" = "true" ] && cmd+=( --no_lpips )
  # shellcheck disable=SC2206
  [ -n "$EXTRA_EVAL_ARGS" ] && cmd+=( $EXTRA_EVAL_ARGS )

  echo ""
  echo "[eval:$task] ${cmd[*]}"
  "${cmd[@]}" 2>&1 | tee "$LOG_DIR/eval_${task}_cand${CANDIDATE}.log"
  SUMMARIES+=( "$metrics_dir/${task}_summary.json" )
}

SUMMARIES=()
for task in $TASKS; do
  case "$task" in
    img2svg)  test_dir="$TEST_DIR_IMG2SVG";  weight="$WEIGHT_IMG2SVG";  pred_dir="$OUTPUT_ROOT/task1_img2svg" ;;
    complete) test_dir="$TEST_DIR_COMPLETE"; weight="$WEIGHT_COMPLETE"; pred_dir="$OUTPUT_ROOT/task2_complete" ;;
    *) echo "Error: 未知任务 '$task'（可选 img2svg / complete）" >&2; exit 1 ;;
  esac
  if [ ! -f "$test_dir/test_meta.csv" ]; then
    echo "Error: $test_dir/test_meta.csv 不存在，请先运行 metrics/eval_svg_tasks.py split" >&2
    exit 1
  fi

  case "$STAGE" in
    all)   run_infer "$task" "$test_dir" "$pred_dir" "$weight"; run_eval "$task" "$test_dir" "$pred_dir" ;;
    infer) run_infer "$task" "$test_dir" "$pred_dir" "$weight" ;;
    eval)  run_eval "$task" "$test_dir" "$pred_dir" ;;
    *) echo "Error: 未知 STAGE '$STAGE'（可选 all / infer / eval）" >&2; exit 1 ;;
  esac
done

# ==============================================================================
# 汇总
# ==============================================================================

if [ "${#SUMMARIES[@]}" -gt 0 ]; then
  echo ""
  "$EVAL_PYTHON" - "${SUMMARIES[@]}" <<'EOF' | tee "$LOG_DIR/summary_cand${CANDIDATE:-1}.txt"
import json, sys
fmt = lambda v: "N/A" if v is None else f"{v:.5f}"
print("=" * 72)
print(f"{'task':<10}{'cand':<6}{'ok/total':<10}{'MSE↓':<12}{'SSIM↑':<12}{'LPIPS↓':<12}")
print("-" * 72)
for path in sys.argv[1:]:
    s = json.load(open(path))
    print(f"{s['task']:<10}{str(s['candidate']):<6}{str(s['num_ok']) + '/' + str(s['num_samples']):<10}"
          f"{fmt(s['mse']):<12}{fmt(s['ssim']):<12}{fmt(s['lpips']):<12}")
print("=" * 72)
EOF
fi
