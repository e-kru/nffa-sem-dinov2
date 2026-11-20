#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export XFORMERS_DISABLED=1 PYTHONNOUSERSITE=1

stage=all
case "${1:-}" in
  '') ;;
  --preflight) stage=preflight ;;
  *) echo "Usage: $0 [--preflight]" >&2; exit 2 ;;
esac

python="${PYTHON:-python}"
checkpoint=outputs/pretrained/dinov2_vitl14_pretrain.pth
checkpoint_sha=d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428
output=outputs/encoder_evaluation/original_vitl14_vs_vits14_seed42
verified=outputs/encoder_evaluation/nffa_5000step_workers_v1/dinov2_vits14_b938bf1bc15cd2ec
mkdir -p "$output"
exec 9>"$output/.writer.lock"
flock -n 9
printf '%s  %s\n' "$checkpoint_sha" "$checkpoint" | sha256sum --check -

command=("$python" -u -m scripts.evaluate_sem_encoder
  --encoder-name dinov2_vitl14 --encoder-checkpoint "$checkpoint"
  --output-root "$output" --verified-data-root "$verified"
  --splits A --seed 42 --head-profile residual_v1 --device cuda:0 --resume)

"${command[@]}" --stage "$stage" --task classification --heads logreg mlp --workers 8 \
  2>&1 | tee -a "$output/classification.log"
"${command[@]}" --stage "$stage" --task segmentation --heads linear cnn8 --workers 16 \
  2>&1 | tee -a "$output/segmentation.log"

if [[ "$stage" == all ]]; then
  "${command[@]}" --stage report --task both --classification-heads logreg mlp \
    --segmentation-heads linear cnn8 2>&1 | tee -a "$output/report.log"
fi
