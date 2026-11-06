#!/usr/bin/env bash
# Git Bash / WSL / Linux 本机拉取三个 step 到 E:/temp/2308
# 用法:
#   export REMOTE_HOST=你在 Cursor/SSH 里用的 Host 名
#   bash transfer_stage2_steps_local_pull.sh
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-xeclogin1.xjtlu.edu.cn}"
REMOTE_USER="${REMOTE_USER:-weiguangzhang21}"
REMOTE_BASE="${REMOTE_BASE:-/gpfs/work/int/weiguangzhang21/project/OmniSVG-train/output_stage2/omnisvg_stage2_4b_20261020_230852}"
STEPS=(step_110000 step_145000 step_155000)

# Git Bash: /e/temp/2308；也可 DEST=/e/temp/2308
DEST="${DEST:-/e/temp/2308}"
mkdir -p "$DEST"

TS="$(date +%Y%m%d_%H%M%S)"
LOG="${DEST}/transfer_pull_${TS}.log"

log() { echo "$(date -Is) $*" | tee -a "$LOG"; }

log "LOG=$LOG"
log "REMOTE=${REMOTE_USER}@${REMOTE_HOST}"
log "DEST=$DEST"

if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "${REMOTE_USER}@${REMOTE_HOST}" "echo ok" >/dev/null 2>&1; then
  log "ERROR: 无法 SSH 到 ${REMOTE_USER}@${REMOTE_HOST}"
  log "请先: 1) 连学校 VPN  2) 设置 REMOTE_HOST 为 Cursor SSH 配置里的 Host（见 transfer_audit/README.txt）"
  exit 1
fi

for step in "${STEPS[@]}"; do
  remote="${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}/${step}/"
  local_dir="${DEST}/${step}"
  mkdir -p "$local_dir"
  log "START $step -> $local_dir"

  if command -v rsync >/dev/null 2>&1; then
    rsync -avh --partial --inplace --info=progress2 -e ssh \
      "$remote" "${local_dir}/" 2>&1 | tee -a "$LOG"
  else
    log "无 rsync，使用 scp -r"
    scp -r "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}/${step}" "$DEST/" 2>&1 | tee -a "$LOG"
  fi

  log "DONE $step"
done

log "全部完成"
