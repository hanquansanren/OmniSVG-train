#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AUDIT="${ROOT}/transfer_audit"
CONFIG="${AUDIT}/transfer_config.env"
PUSH="${ROOT}/scripts/transfer_stage2_steps_push_via_tunnel.sh"

mkdir -p "$AUDIT"

if [[ ! -f "$CONFIG" ]]; then
  cp "${AUDIT}/transfer_config.env.example" "$CONFIG"
  echo "已创建 ${CONFIG} — 请编辑 LOCAL_USER 后重新运行本脚本。"
  exit 1
fi

# shellcheck disable=SC1090
source "$CONFIG"
if [[ -z "${LOCAL_USER:-}" || "${LOCAL_USER}" == *"你的Windows"* ]]; then
  echo "请先在 ${CONFIG} 中设置 LOCAL_USER（Windows OpenSSH 用户名）"
  exit 1
fi

TS="$(date +%Y%m%d_%H%M%S)"
NOHUP_LOG="${AUDIT}/nohup_push_${TS}.out"
PID_FILE="${AUDIT}/push.pid"

nohup bash "$PUSH" >>"$NOHUP_LOG" 2>&1 &
echo $! >"$PID_FILE"

echo "后台传输已启动 PID=$(cat "$PID_FILE")"
echo "本机请先执行反向隧道:"
echo "  ssh -N -R ${LOCAL_SSH_PORT:-2222}:127.0.0.1:22 ${USER}@xeclogin1.xjtlu.edu.cn"
echo "传输日志: transfer_audit/transfer_push_*.log"
echo "nohup 输出: $NOHUP_LOG"
echo "查看进度: bash scripts/watch_transfer_log.sh"
