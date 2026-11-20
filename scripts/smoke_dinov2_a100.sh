#!/usr/bin/env bash
# Run manually on the GPU host. At most one short synthetic GPU pilot.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
pilot_python="${PYTHON:-python}"
pilot_checkpoint=${DINOV2_VITS14_CHECKPOINT:-outputs/pretrained/dinov2_vits14_pretrain.pth}
pilot_output=${1:-outputs/sem_ssl_checks/a100_$(date +%Y%m%d_%H%M%S)}
if [[ -e "$pilot_output" ]]; then
    printf 'Output already exists: %s\n' "$pilot_output" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONNOUSERSITE=1

# Keep Meta's complete official trainer as the first probe. Its report must
# identify this exact GPU/PyTorch stack before the adapter can be used.
if "$pilot_python" -m scripts.probe_dinov2_official \
    --checkpoint "$pilot_checkpoint" --steps 2 --output "$pilot_output/official"; then
    printf 'Official trainer passed. Report: %s/official/report.json\n' "$pilot_output"
else
    "$pilot_python" -m scripts.pilot_dinov2 \
        --checkpoint "$pilot_checkpoint" \
        --official-report "$pilot_output/official/report.json" \
        --synthetic --device cuda --precision bf16 --batch-size 2 \
        --steps 3 --num-workers 0 --output "$pilot_output/native"
    "$pilot_python" -m scripts.verify_dinov2_checkpoint --run "$pilot_output/native"
    printf 'Native pilot and checkpoint/export checks passed. Report: %s/native/report.json\n' "$pilot_output"
fi
