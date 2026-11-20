#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export XFORMERS_DISABLED=1 PYTHONNOUSERSITE=1
exec "${PYTHON:-python}" \
  -m scripts.run_sem_augmentation_vits14 "$@"
