# Resolve TRAIN_CONFIG_FILE for training launch scripts.
# Source from repo root (or any cwd); uses RUN_SCRIPT_DIR set by caller.
#
# Inputs:
#   _DEFAULT_TRAIN_CONFIG — default yaml basename under CONFIG_DIR
# Optional env (from sbatch --export):
#   TRAIN_CONFIG_FILE, BOUND_CONFIG_*
# Optional on disk (queued jobs, no requeue):
#   configs/job_bindings/by_job/${SLURM_JOB_ID}.yaml

resolve_train_config() {
    local _default="${1:?default train config basename required}"
    TRAIN_CONFIG_FILE="${TRAIN_CONFIG_FILE:-${_default}}"
    if [ "${TRAIN_CONFIG_FILE}" = "${_default}" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
        local _job_bound="configs/job_bindings/by_job/${SLURM_JOB_ID}.yaml"
        if [ -f "${_job_bound}" ]; then
            TRAIN_CONFIG_FILE="job_bindings/by_job/${SLURM_JOB_ID}.yaml"
            BOUND_CONFIG_SOURCE="${_job_bound} (by SLURM_JOB_ID)"
            BOUND_CONFIG_SNAPSHOT="${_job_bound}"
        fi
    fi
}

print_bound_train_config() {
    if [ -n "${BOUND_CONFIG_SOURCE:-}" ] || [ -n "${BOUND_CONFIG_SNAPSHOT:-}" ]; then
        echo "Bound At Submit:   ${BOUND_CONFIG_SUBMIT_TIME:-unknown}"
        if [ -n "${BOUND_CONFIG_SOURCE:-}" ]; then
            echo "Bound Source:      ${BOUND_CONFIG_SOURCE}"
        fi
        if [ -n "${BOUND_CONFIG_SNAPSHOT:-}" ]; then
            echo "Bound Snapshot:    ${BOUND_CONFIG_SNAPSHOT}"
        fi
    fi
}
