#!/usr/bin/env bash
# 将指定 step_* checkpoint 目录 rsync 到远程服务器。
#
# 用法（在脚本所在项目根目录或任意目录执行均可）：
#   export RSYNC_SSH_PASSWORD='你的SSH密码'
#   bash scripts/upload_stage2_checkpoints_to_remote.sh
#
# 可选环境变量：
#   SRC_BASE          本地 checkpoint 父目录（默认 omnisvg_stage2_4b_20261020_231348）
#   REMOTE_DIR        远端目录（默认 .../weights/2313）
#   UPLOAD_STEPS      空格分隔的 step 编号，例如 "205000 305000 235000 297500"
#   RSYNC_EXTRA_OPTS  追加传给 rsync 的参数，例如 "--dry-run"
#
# 安全提示：请勿把密码写进本文件或提交到 git；用后建议 unset RSYNC_SSH_PASSWORD。
# 更推荐：ssh-copy-id 配置免密后再运行（此时无需设置 RSYNC_SSH_PASSWORD）。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# --- 本地 checkpoint 父目录（五个 step 均在同一 run 下）---
SRC_BASE="${SRC_BASE:-${PROJECT_ROOT}/output_stage2/omnisvg_stage2_4b_20261020_231348}"

if [[ -n "${UPLOAD_STEPS:-}" ]]; then
  # shellcheck disable=SC2206
  STEPS=(${UPLOAD_STEPS})
else
  STEPS=(255000 265000 275000 285000 295000)
fi

# --- 远程 ---
REMOTE_USER="${REMOTE_USER:-phd23_weiguang_zhang}"
REMOTE_HOST="${REMOTE_HOST:-10.7.88.77}"
REMOTE_DIR="${REMOTE_DIR:-/Data_New/phd23_weiguang_zhang/weights/2313}"

missing=()
for s in "${STEPS[@]}"; do
  d="${SRC_BASE}/step_${s}"
  if [[ ! -d "$d" ]]; then
    missing+=("$d")
  fi
done
if (( ${#missing[@]} > 0 )); then
  echo "错误：以下目录不存在：" >&2
  printf '  %s\n' "${missing[@]}" >&2
  exit 1
fi

SSH_OPTS=(
  -o BatchMode=no
  -o StrictHostKeyChecking=accept-new
)

RSYNC_E=(ssh "${SSH_OPTS[@]}")
RSYNC_BASE=(rsync -avz --info=progress2 --human-readable)
if [[ -n "${RSYNC_EXTRA_OPTS:-}" ]]; then
  # shellcheck disable=SC2206
  RSYNC_BASE+=(${RSYNC_EXTRA_OPTS})
fi

run_rsync() {
  if [[ -n "${RSYNC_SSH_PASSWORD:-}" ]]; then
    askpass="$(mktemp)"
    trap 'rm -f "$askpass"' EXIT
    printf '#!/bin/sh\necho "$RSYNC_SSH_PASSWORD"\n' > "$askpass"
    chmod 700 "$askpass"
    export SSH_ASKPASS="$askpass"
    export SSH_ASKPASS_REQUIRE=force
    # 无图形界面时强制走 SSH_ASKPASS
    export DISPLAY="${DISPLAY:-:0}"
    # shellcheck disable=SC2068
    "${RSYNC_BASE[@]}" -e "${RSYNC_E[*]}" "$@" || return $?
  else
    echo "提示：未设置 RSYNC_SSH_PASSWORD，将使用交互式密码输入（每个目录可能各输一次）。"
    echo "      或先配置 SSH 公钥免密：ssh-copy-id ${REMOTE_USER}@${REMOTE_HOST}"
    # shellcheck disable=SC2068
    "${RSYNC_BASE[@]}" -e "${RSYNC_E[*]}" "$@" || return $?
  fi
}

sources=()
for s in "${STEPS[@]}"; do
  sources+=("${SRC_BASE}/step_${s}")
done

dest="${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"

echo "源（共 ${#sources[@]} 个目录）："
printf '  %s\n' "${sources[@]}"
echo "目标：${dest}"
echo ""

run_rsync "${sources[@]}" "$dest"

echo ""
echo "完成。"
