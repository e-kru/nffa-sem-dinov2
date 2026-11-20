#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=''

python_bin="${PYTHON:-python}"
audit_dir=outputs/sem_corpus/sem_v0_full_decode_audit
result="$audit_dir/result.json"
revision=outputs/sem_corpus/sem_posttraining_v0_decode_verified
mkdir -p "$audit_dir"
exec 9>"$audit_dir/.prepare.lock"
flock -n 9
if [[ ! -f "$result" ]]; then
  "$python_bin" -u -m scripts.audit_sem_v0_decode \
    --manifest outputs/sem_corpus/sem_posttraining_v0_decode_trim20/training_manifest.csv \
    --output "$result" --workers "${1:-8}"
fi
if [[ ! -d "$revision" ]]; then
  "$python_bin" -u -m scripts.select_sem_v0_decode_verified --audit "$result"
fi
bash scripts/run_sem_ssl_v0_300epochs.sh --stage preflight
