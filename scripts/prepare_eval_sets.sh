#!/bin/bash
# 生成 ./eval 下三套测试集（eval_task1_1 / eval_task1_2 / eval_task2）
#
# 默认 60 条、seed=42、每个汉字最多 1 个样本（与当前 eval 一致）。
# 扩到 600 条示例:
#   NUM=600 ALLOW_DUP_CHAR=1 bash scripts/prepare_eval_sets.sh --overwrite
#
# 仅预览可选数量（不写文件）:
#   NUM=600 ALLOW_DUP_CHAR=1 bash scripts/prepare_eval_sets.sh --dry_run
#
# 环境变量（均可覆盖）:
#   PYTHON          Python 解释器（需 cairosvg、Pillow；渲染 gt200 时用）
#   NUM             样本数，默认 60
#   SEED            随机种子，默认 42
#   ALLOW_DUP_CHAR  设为 1 允许同一字的多个 _vXX 变体（扩大量时建议开启）
#   EVAL_ROOT       输出根目录，默认 ./eval
#   SRC_IMG2SVG_OLD stage1_old2old 根目录
#   SRC_IMG2SVG_NEW stage1_old2new 根目录
#   SRC_COMPLETE    my_lis2_2 根目录

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
NUM="${NUM:-60}"
SEED="${SEED:-42}"
EVAL_ROOT="${EVAL_ROOT:-./eval}"

CMD=( "$PYTHON" ./scripts/prepare_eval_sets.py
  --eval_root "$EVAL_ROOT"
  --num "$NUM"
  --seed "$SEED"
)

[ -n "${SRC_IMG2SVG_OLD:-}" ] && CMD+=( --src_img2svg_old "$SRC_IMG2SVG_OLD" )
[ -n "${SRC_IMG2SVG_NEW:-}" ] && CMD+=( --src_img2svg_new "$SRC_IMG2SVG_NEW" )
[ -n "${SRC_COMPLETE:-}" ] && CMD+=( --src_complete "$SRC_COMPLETE" )
[ "${ALLOW_DUP_CHAR:-0}" = "1" ] && CMD+=( --allow_dup_char )

# 透传 --overwrite / --dry_run / --tasks 等
CMD+=( "$@" )

echo "============================================================"
echo "Prepare eval sets"
echo "  Repo:  $REPO_ROOT"
echo "  Out:   $EVAL_ROOT"
echo "  Num:   $NUM   Seed: $SEED   allow_dup_char: ${ALLOW_DUP_CHAR:-0}"
echo "  Cmd:   ${CMD[*]}"
echo "============================================================"

exec "${CMD[@]}"
