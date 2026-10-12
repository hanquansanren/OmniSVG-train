#!/bin/bash
exec "$(dirname "$0")/submit_slurm_train.sh" run2_multi2.slurm "$@"
