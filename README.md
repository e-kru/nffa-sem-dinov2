# DINOv2 baseline for NFFA SEM

Image classification experiments on the NFFA-EUROPE scanning electron microscopy (SEM) dataset. A frozen DINOv2 ViT-S/14 (`dinov2_vits14`) produces 384-dimensional features; a linear logistic-regression classifier is trained on top. The repository also contains dataset-cleaning and duplicate-audit tools.

## Pipeline

- **Preprocessing:** convert to RGB, heuristically remove a white metadata strip near the bottom (`src/preprocessing_v2.py`), resize the shorter side to 256 pixels, center-crop to 224 × 224, and apply ImageNet normalization.
- **Splits:** `create_splits.py` creates class-stratified 70%/15%/15% train/validation/test splits with seed 42 in `outputs/splits.json`. These are image-level splits and do not keep duplicate groups together.
- **Cleaning (Dataset V2):** count green pixels after metadata cropping and exclude images with at least 15 detected green pixels. `build_dataset_v2.py` records inclusion decisions in a manifest without modifying the source images; it currently asserts an original inventory of 21,169 images.
- **Duplicate audits:** file SHA-256, cropped-pixel SHA-256, perceptual hashes, and DINOv2 cosine neighbors help identify exact duplicates and similar images. Attaching SHA groups records group IDs; it does not deduplicate images or create new splits.
- **Classification:** standardize features and fit class-balanced logistic regression. Scripts report accuracy and macro-F1; the full-data classifier also prints per-class reports.

## Setup

Use Python 3.10+ and run commands from the repository root. `requirements.txt` is currently empty; install the packages used by the baseline and audit scripts explicitly:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch torchvision numpy Pillow scikit-learn matplotlib tqdm ImageHash
```

For GPU extraction, use a CUDA-compatible PyTorch installation. DINOv2 is loaded through `torch.hub` from `facebookresearch/dinov2`; the first run needs internet access to download the model code and pretrained weights. Dependency versions are not pinned.

Provide the dataset separately in `ImageFolder` layout (one subdirectory per class):

```text
nffa_sem_100/
  class_a/image_001.jpg
  class_b/image_002.jpg
  ...
```

Set `DATASET_ROOT` in `src/dataset.py` to this directory. Audit and preview scripts also contain their own `DATASET_ROOT` or `ROOT` constants pointing to `/mnt/data/stud-ueqio/datasets/nffa_sem_100`; update those paths for the scripts you use. There is no shared path configuration or dataset downloader.

## Usage

### Small baseline

```bash
python -m scripts.create_splits
python -m scripts.run_baseline
```

This CPU smoke experiment uses the first 5 training, 2 validation, and 2 test images per class from the saved splits. It saves `outputs/features_smoke.npz` and reports validation/test metrics. It uses metadata cropping but does not apply the V2 green-overlay filter.

### Dataset V2 cleaning and feature extraction

After creating `outputs/splits.json`:

```bash
python -m scripts.audit_green_overlays --workers 8
python -m scripts.build_dataset_v2
python -m scripts.audit_duplicates
python -m scripts.attach_sha_groups
python -m scripts.extract_all_features --device cpu --batch-size 64 --chunk-size 512 --num-workers 8
```

The extractor processes only included manifest entries and saves resumable `chunk_*.npz` files containing `X`, `y`, and relative image `paths` under `outputs/dataset_v2/features/`. Use `--device cuda` for a supported GPU. `--chunk-size` must be divisible by `--batch-size`. For a short check, use `--limit 128 --output-dir outputs/v2_smoke_features`; use a separate output directory when changing the selected dataset, preprocessing, or chunk configuration.

Optional similarity audits after full extraction:

```bash
python -m scripts.audit_dataset_v2_phash
python -m scripts.audit_global_phash --workers 8 --max-distance 4
python -m scripts.audit_dinov2_neighbors --device cpu --top-k 10
```

Reports are written to `outputs/audit/` and `outputs/dataset_v2/`. Preview and visualization scripts in `scripts/` support manual inspection of crops, overlays, and candidate duplicate pairs.

### Existing full-data classifier

`python -m scripts.train_classifier` expects legacy feature chunks for all 21,169 images in `outputs/features_dinov2_vits14/` and the original `outputs/splits.json`. It trains on the training split and reports training/validation metrics. With those features and the SHA audit available, `python -m scripts.evaluate_clean_val` also evaluates validation after excluding byte-identical copies of training images.

**Current limitation:** the V2 extractor's filtered output is not directly compatible with this classifier. Group-aware V2 splits and a classifier consuming the V2 manifest/features are not implemented. The audits alone do not establish a duplicate-free evaluation.

## Repository layout

- `src/`: dataset loading, metadata cropping, and the frozen DINOv2 encoder.
- `scripts/`: baseline, splits, feature extraction, classification, audits, and visual checks.
- `outputs/`: generated manifests, features, and reports (ignored by Git).

Dataset images and generated results are not included. The checked-in YOLO weight files are not used by the DINOv2 baseline.
