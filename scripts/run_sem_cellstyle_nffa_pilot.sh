#!/usr/bin/env bash
# Explicit user-run NFFA pilot. Checks/calibration precede scientific training.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec bash scripts/run_sem_cellstyle_regularization_grid.sh \
  --stage run \
  --config configs/sem_ssl/sem_cellstyle_nffa_vitl14_pilot.yaml \
  --output outputs/sem_ssl_regularization/nffa_split_a_vitl14_residual_heads_v1_resume_approved_seed42 \
  --verified-data-root outputs/encoder_evaluation/nffa_5000step_workers_v1/dinov2_vits14_b938bf1bc15cd2ec \
  "$@"
