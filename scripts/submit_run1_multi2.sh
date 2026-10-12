#!/bin/bash
exec "$(dirname "$0")/submit_slurm_train.sh" run1_multi2.slurm "$@"
