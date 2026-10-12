#!/bin/bash
# Submit a Slurm script with a frozen train YAML (survives queue wait).
#
# Usage:
#   bash scripts/submit_slurm_train.sh run1_multi1.slurm configs/train_config_zhuan_a100_fat.yaml
#   bash scripts/submit_slurm_train.sh run1_multi2.slurm --job-name run2 configs/foo.yaml
#   bash scripts/submit_slurm_train.sh run2_multi2.slurm configs/train_config_cc_a100fat_fsdp.yaml --partition=gpua8001t
#
# Wrappers: submit_run1_multi1.sh, submit_run1_multi2.sh, submit_run2_multi2.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SLURM_SCRIPT=""
SBATCH_ARGS=()
CONFIG_SRC=""

while [ $# -gt 0 ]; do
    case "$1" in
        --job-name)
            SBATCH_ARGS+=("--job-name=$2")
            shift 2
            ;;
        --job-name=*)
            SBATCH_ARGS+=("$1")
            shift
            ;;
        --)
            shift
            break
            ;;
        -*)
            SBATCH_ARGS+=("$1")
            shift
            ;;
        *)
            if [ -z "$SLURM_SCRIPT" ] && [[ "$1" == *.slurm ]]; then
                SLURM_SCRIPT="$1"
            elif [ -z "$CONFIG_SRC" ]; then
                CONFIG_SRC="$1"
            else
                SBATCH_ARGS+=("$1")
            fi
            shift
            ;;
    esac
done

while [ $# -gt 0 ]; do
    SBATCH_ARGS+=("$1")
    shift
done

if [ -z "$SLURM_SCRIPT" ] || [ -z "$CONFIG_SRC" ]; then
    echo "Usage: bash scripts/submit_slurm_train.sh <job.slurm> [--job-name NAME] <train_config.yaml> [sbatch flags...]" >&2
    exit 1
fi

if [ ! -f "$SLURM_SCRIPT" ]; then
    echo "Error: Slurm script not found: $SLURM_SCRIPT" >&2
    exit 1
fi

if [ ! -f "$CONFIG_SRC" ]; then
    echo "Error: config not found: $CONFIG_SRC" >&2
    exit 1
fi

BIND_DIR="$ROOT/configs/job_bindings"
mkdir -p "$BIND_DIR"

SUBMIT_TIME="$(date +%Y%m%d_%H%M%S)"
BASENAME="$(basename "$CONFIG_SRC" .yaml)"
BASENAME="${BASENAME// /_}"
SNAP_NAME="${SUBMIT_TIME}_${BASENAME}.yaml"
SNAP_PATH="$BIND_DIR/$SNAP_NAME"
REL_CONFIG="job_bindings/$SNAP_NAME"

cp "$CONFIG_SRC" "$SNAP_PATH"

MANIFEST="$BIND_DIR/${SUBMIT_TIME}_${BASENAME}.submit.txt"
{
    echo "submit_time=$SUBMIT_TIME"
    echo "slurm_script=$SLURM_SCRIPT"
    echo "source_config=$CONFIG_SRC"
    echo "snapshot=$SNAP_PATH"
    echo "train_config_file=$REL_CONFIG"
    echo "user=${USER:-unknown}"
    echo "host=$(hostname -s 2>/dev/null || hostname)"
    if command -v git >/dev/null 2>&1 && git -C "$ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        echo "git_commit=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
        echo "git_dirty=$(git -C "$ROOT" status --porcelain 2>/dev/null | wc -l)"
    fi
} > "$MANIFEST"

EXPORT_VARS="ALL,TRAIN_CONFIG_FILE=${REL_CONFIG}"
EXPORT_VARS+=",BOUND_CONFIG_SOURCE=${CONFIG_SRC}"
EXPORT_VARS+=",BOUND_CONFIG_SNAPSHOT=${SNAP_PATH}"
EXPORT_VARS+=",BOUND_CONFIG_SUBMIT_TIME=${SUBMIT_TIME}"

echo "Binding train config at submit time:"
echo "  slurm:    $SLURM_SCRIPT"
echo "  source:   $CONFIG_SRC"
echo "  snapshot: $SNAP_PATH"
echo "  manifest: $MANIFEST"
echo ""

sbatch --export="$EXPORT_VARS" "${SBATCH_ARGS[@]}" "$SLURM_SCRIPT"
