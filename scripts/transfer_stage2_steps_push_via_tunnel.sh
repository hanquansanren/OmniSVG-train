#!/usr/bin/env bash
# 经反向 SSH 隧道，从集群把 step_* 推到本机 E:/temp/2308（需本机 OpenSSH + 隧道）
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AUDIT="${ROOT}/transfer_audit"
CONFIG="${AUDIT}/transfer_config.env"
# shellcheck disable=SC1090
[[ -f "$CONFIG" ]] && source "$CONFIG"

LOCAL_USER="${LOCAL_USER:-}"
LOCAL_SSH_PORT="${LOCAL_SSH_PORT:-2222}"
LOCAL_DEST="${LOCAL_DEST:-E:/temp/2308}"
RUN_BASE="${RUN_BASE:-/gpfs/work/int/weiguangzhang21/project/OmniSVG-train/output_stage2/omnisvg_stage2_4b_20261020_230852}"
STEPS="${STEPS:-step_110000 step_145000 step_155000}"
WAIT_TUNNEL_SEC="${WAIT_TUNNEL_SEC:-86400}"
POLL_SEC="${POLL_SEC:-15}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG:-${AUDIT}/transfer_push_${TS}.log}"
mkdir -p "$AUDIT"
touch "$LOG"

log() { echo "$(date -Is) $*" | tee -a "$LOG"; }

if [[ -z "$LOCAL_USER" ]]; then
  log "ERROR: 未设置 LOCAL_USER。请复制 transfer_audit/transfer_config.env.example 为 transfer_config.env 并填写 Windows 用户名。"
  exit 1
fi

SSH_OPTS=(-o "StrictHostKeyChecking=accept-new" -o "ConnectTimeout=8" -p "$LOCAL_SSH_PORT")
REMOTE="${LOCAL_USER}@127.0.0.1"

tunnel_ok() {
  ssh "${SSH_OPTS[@]}" -o BatchMode=yes "$REMOTE" "echo ok" >/dev/null 2>&1
}

log "等待本机反向隧道 (127.0.0.1:${LOCAL_SSH_PORT})，最长 ${WAIT_TUNNEL_SEC}s …"
log "本机需先执行: ssh -N -R ${LOCAL_SSH_PORT}:127.0.0.1:22 ${USER}@xeclogin1.xjtlu.edu.cn"

deadline=$((SECONDS + WAIT_TUNNEL_SEC))
while ! tunnel_ok; do
  if (( SECONDS >= deadline )); then
    log "ERROR: 超时，仍未检测到隧道。可改用 scripts/transfer_stage2_steps_local_pull.ps1 在本机拉取。"
    exit 2
  fi
  sleep "$POLL_SEC"
done

log "隧道已就绪，创建目标目录 ${LOCAL_DEST}"
ssh "${SSH_OPTS[@]}" "$REMOTE" "powershell -NoProfile -Command \"New-Item -ItemType Directory -Force -Path '${LOCAL_DEST}' | Out-Null\"" \
  2>&1 | tee -a "$LOG" || true

transfer_one() {
  local step="$1"
  local src="${RUN_BASE}/${step}"
  if [[ ! -d "$src" ]]; then
    log "ERROR: 源目录不存在: $src"
    return 1
  fi
  log "START $step ($(du -sh "$src" | awk '{print $1}')) -> ${REMOTE}:${LOCAL_DEST}/${step}/"

  if ssh "${SSH_OPTS[@]}" "$REMOTE" "command -v rsync" >/dev/null 2>&1; then
    rsync -avh --partial --inplace --info=progress2 \
      -e "ssh -p ${LOCAL_SSH_PORT} -o StrictHostKeyChecking=accept-new -o ConnectTimeout=8" \
      "${src}/" "${REMOTE}:${LOCAL_DEST}/${step}/" 2>&1 | tee -a "$LOG"
  else
    log "本机无 rsync，使用 scp -r（进度较粗）"
    scp -r "${SSH_OPTS[@]}" "$src" "${REMOTE}:${LOCAL_DEST}/" 2>&1 | tee -a "$LOG"
  fi

  log "DONE $step"
}

for step in $STEPS; do
  transfer_one "$step"
done

log "全部完成。日志: $LOG"
