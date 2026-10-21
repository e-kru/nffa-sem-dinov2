# Unified SEM downstream pipeline

## Responsibility boundaries

- `src/config.py`: `DATASET_ROOT`, configurable through `NFFA_SEM_ROOT`.
- `src/preprocessing_v2.py`: canonical V2 crop, resize 256, 224 center-crop,
  ImageNet normalization, and aligned mask transforms (nearest-neighbor mask
  resize). The old V1 crop is removed; do **not** apply the crop twice.
- `src/dinov2_encoder.py`: canonical DINOv2 loader, input transform and
  compatible backbone checkpoint loading.
- `src/downstream/data.py`: train/validation datasets and train-only scaling.
- `src/downstream/model.py`, `heads.py`: encoder, task heads, upsampling.
- `src/downstream/engine.py`, `metrics.py`: optimization and evaluation.
- `scripts/run_downstream.py`: CLI for both downstream tasks.

## Frozen classification (existing SEM data)

```bash
python -m scripts.run_downstream --task classification --head linear \
  --mode cached --split A --output outputs/downstream/linear_A
python -m scripts.run_downstream --task classification --head mlp \
  --mode cached --split A --dropout 0 \
  --output outputs/downstream/mlp_A
```

The cached 384-d vectors come from the **original** DINOv2 ViT-S/14 encoder.
Split A/B are read from `outputs/dataset_v3/splits`. Only training features
fit the scaler. The PyTorch linear head is distinct from the original sklearn
Logistic Regression baseline. Runs save `best_model.pt`, `history.csv`,
`summary.json`, and a cached-mode `scaler.joblib`.

## Semantic segmentation (requires real pixel labels)

The existing NFFA classification dataset contains no pixelwise ground-truth
masks. Create a CSV with `image_path,mask_path,split`, with relative paths
resolved relative to the CSV file. Masks must be indexed PNGs with class IDs
0..C-1 and optionally ignore label 255; aligned to *uncropped* source images.

```csv
image_path,mask_path,split
images/a.png,masks/a.png,train
images/b.png,masks/b.png,val
images/c.png,masks/c.png,test
```

```bash
python -m scripts.run_downstream --task segmentation --head linear \
  --mode image --manifest data/segmentation/split.csv --num-classes 2 \
  --batch-size 8 --output outputs/downstream/seg_linear_A

python -m scripts.run_downstream --task segmentation --head mlp \
  --mode image --manifest data/segmentation/split.csv --num-classes 2 \
  --batch-size 8 --output outputs/downstream/seg_mlp_A
```

Patch map for DINOv2 ViT-S/14 on 224x224 is 16x16x384; bilinear upsampling
returns 224x224 logits. 1x1 convolutional heads are a **coarse baseline**,
not a high-resolution decoder. Pixel mIoU selects checkpoints. Real dataset
validation is pending.

## Encoder adaptation later

- `--mode image --encoder-checkpoint PATH` can evaluate compatible
  backbone-only posttrained DINOv2 weights.
- `--train-encoder` is **supervised** end-to-end fine-tuning, not SSL.
- SSL SEM-domain posttraining is **not** implemented yet.
- After changing encoder weights, old `.npz` features cannot be used. Use
  `--mode image` or recompute features into a new, versioned cache directory.

## Research and reproducibility safeguards

Old V1 preprocessing and throwaway V1 audit scripts were removed. Canonical
V2 cropping is unchanged. Historical SHA/pHash and split-generation outputs
must not be regenerated with an altered pipeline and reported as equivalent
without a controlled comparison. Keep outputs and original experiment provenance.
Early stopping / model selection use validation only; the test sets were
previously inspected and must not be reused for further hyperparameter search
as if they were untouched.

## Tests

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```
