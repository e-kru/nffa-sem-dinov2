# SEM representation learning with DINOv2

Modular KIT research pipeline for scanning electron microscopy (SEM). The
baseline uses frozen DINOv2 ViT-S/14 features and linear / MLP classification
heads. Segmentation with pixelwise linear / MLP heads is implemented but has
**not yet been evaluated on a real pixel-mask dataset**. Self-supervised
encoder posttraining is future work.

## Project layout

```text
src/
  config.py                 # Single NFFA root configuration
  preprocessing_v2.py       # Single SEM image+mask preprocessing
  dinov2_encoder.py        # Single DINOv2 factory / transform / checkpoint loader
  downstream/
    data.py                 # Cached feature and image/mask datasets
    heads.py                # Classification and segmentation linear/MLP heads
    model.py                # Backbone + head composition
    engine.py               # Training, validation, early stopping
    metrics.py              # Accuracy, macro-F1, mIoU
scripts/
  run_downstream.py         # New classification/segmentation training entrypoint
  extract_all_features.py   # DINOv2 baseline embedding extraction
  build_dataset_v2.py       # Green-overlay filtered dataset
  build_dataset_v3.py       # Final YOLO-audited dataset
  create_candidate_pairs_v3.py
  create_split_A_exact.py
  create_split_B_full.py
  train_linear_probe_A.py   # Reproducibility: original sklearn LR baseline
  train_linear_probe_B.py   # Reproducibility: original sklearn LR baseline
  ...                       # Remaining dataset provenance/audit scripts
```

The previous V1 cropping implementation and old throwaway training/demo
scripts have been removed from the active tree. Final preprocessing is
`src.preprocessing_v2`; there is no fallback to V1. Historical published
outputs are retained (do **not** rewrite old caches).

## NFFA classification

- Original dataset: 21,169 images, 10 classes.
- Final cleaned V3 dataset: 18,286 images.
- Split A: exact-duplicate group aware; Split B: conservative visual-similarity grouping.
- Frozen DINOv2 ViT-S/14: 384-dimensional global feature vectors.

```bash
python -m scripts.run_downstream --task classification --head linear \
  --mode cached --split A --output outputs/downstream/linear_A
python -m scripts.run_downstream --task classification --head mlp \
  --mode cached --split A --dropout 0 --output outputs/downstream/mlp_A
```

Use `--split B` with a distinct `--output` to train Split B. These use the
previously extracted V2 frozen-encoder feature caches. Selection and early
stopping use **train and validation only**; test is untouched by the CLI.

The sklearn logistic-regression reference baselines (`train_linear_probe_A/B`)
use a different optimizer and tuned regularization, and are intentionally
retained to reproduce earlier numbers. Original test Macro-F1: **0.9087** on
Split A, **0.8785** on Split B. Original 3-layer MLP test Macro-F1 with dropout
0.3: **0.9361** on Split A, **0.9113** on Split B. These are historical
measurements, not freshly rerun after the refactor.

## Segmentation and posttraining

Refer to [PIPELINE.md](PIPELINE.md). Segmentation uses spatial patch features,
not the cached global CLS vectors. Future SEM-specific DINOv2 checkpoints
require either image-mode evaluation or newly extracted, versioned caches.

## Reproducing dataset construction

Old NFFA inventory construction (`scripts/create_splits.py`), V2/V3 dataset
manifests, duplicate audits, feature extraction, and group-split builders are
kept. The legacy V1 pHash and V1 crop-comparison scripts were intentionally
removed: silently changing their crop would invalidate their historical
interpretation. The generated reports remain available in historical outputs
or Git history.

The shared dataset root defaults to
`/mnt/data/stud-ueqio/datasets/nffa_sem_100` and can be overridden with
`NFFA_SEM_ROOT`. New outputs must go to new directories; do not overwrite
historical feature caches because source-fingerprint logic tracks code files.

## Checks

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```
