# Microscopy Image Classification with DINOv2

## Overview

This project investigates DINOv2 as a pretrained visual feature extractor for scanning electron microscopy (SEM) images.

The first experiment establishes a classification baseline using frozen DINOv2 embeddings and a linear classifier, with particular attention to dataset quality and evaluation reliability.

## Dataset

The NFFA-EUROPE SEM dataset contains **21,169 images across 10 classes**: `Biological`, `Fibres`, `Films_Coated_Surface`, `MEMS_devices_and_electrodes`, `Nanowires`, `Particles`, `Patterned_surface`, `Porous_Sponge`, `Powder`, and `Tips`.

Initial inspection revealed instrument metadata, scale information, measurement boxes, green annotations, and duplicate images. Identical images appeared across the original random train/validation/test partitions, motivating systematic cleaning and duplicate-aware splitting.

## Data Engineering

1. **Green-overlay filtering:** a color-based heuristic identifies green graphical annotations. Images with detected overlays are excluded rather than edited or inpainted. The implemented audit counts green pixels after Crop V2 and flags images with at least 15 such pixels.
2. **Metadata cropping:** Crop V2 (`src/preprocessing_v2.py`) detects and removes metadata strips when images are loaded. Source images remain unchanged.
3. **Residual annotation filtering:** a pretrained YOLO11n detector was fine-tuned on manually annotated SEM images with two classes: `measurement_box` (0) and `metadata_strip` (1). It was applied to all 18,614 green-filtered images after Crop V2, using a confidence threshold of **0.50**. Images with detected measurement boxes or residual metadata strips were excluded. Additional cropping was investigated but rejected because it did not reliably remove the remaining strips. Samples of detections and non-detections were reviewed manually.

| Stage | Images remaining | Images excluded at this stage |
| --- | ---: | ---: |
| Original NFFA-SEM dataset | 21,169 | — |
| Green-overlay filtering | 18,614 | 2,555 |
| Measurement-box filtering | 18,404 | 210 |
| Remaining metadata-strip filtering | 18,286 | 118 |
| **Total** | **18,286** | **2,883** |

All retained images use Crop V2 without additional fixed cropping. Inclusion status, class, exclusion reasons, and preprocessing configuration are recorded in the V3 manifest and configuration described below. This model-assisted procedure does not guarantee that every retained image is annotation-free.

## Duplicate Detection and Splitting

The audit combines SHA-256 hashes of original files, pixel-level SHA-256 after Crop V2, perceptual hashes (pHash), and cosine similarity between DINOv2 embeddings. Exact hashes identify duplicates; pHash and embedding similarity identify potential near-duplicates. Visual similarity alone does not establish a shared physical specimen or acquisition.

### Split A: exact-duplicate-aware

Images are connected if their original-file SHA-256 **or** their cropped-pixel SHA-256 matches. Union-Find constructs connected components, each assigned entirely to one partition. No detected exact-duplicate group crosses partition boundaries.

### Split B: conservative similarity grouping

Images are connected if **at least one** condition holds:

- Original-file SHA-256 matches.
- Cropped-pixel SHA-256 matches.
- pHash Hamming distance is **≤ 8**.
- DINOv2 cosine similarity is **≥ 0.995**.

Union-Find constructs components over these edges, including transitive connections: two images can share a component without meeting a similarity threshold directly. No similarity-graph edge crosses partition boundaries. This intentionally conservative grouping can create large components spanning multiple classes; it is not verified physical-specimen grouping.

| Grouping statistic | Split A | Split B |
| --- | ---: | ---: |
| Exact duplicate pairs / candidate similarity edges | 286 | 13,923 |
| Total components, including singletons | 18,011 | 16,074 |
| Components containing multiple images | 266 | 778 |
| Images in multi-image components | 541 | 2,990 |
| Redundant exact copies | 275 | — |
| Largest component | — | 430 images |

Both cleaned-data splits target 70%/15%/15% allocation, keeping components intact while approximately preserving class proportions. Duplicate copies remain in the dataset within their assigned partition.

| Experiment | Training images | Validation images | Test images | Total |
| --- | ---: | ---: | ---: | ---: |
| Original random split | 14,818 | 3,175 | 3,176 | 21,169 |
| Split A | 12,801 | 2,742 | 2,743 | 18,286 |
| Split B | 12,799 | 2,743 | 2,744 | 18,286 |

## Model and Training

All experiments use a **frozen DINOv2 ViT-S/14** encoder producing 384-dimensional embeddings, followed by multinomial logistic regression. The encoder is not fine-tuned.

For the cleaned-data experiments, image preprocessing converts to RGB, applies Crop V2, resizes the shorter side to 256 pixels, center-crops to 224 × 224, and applies ImageNet normalization. Training uses the following procedure:

- Embeddings were extracted once for the 18,614 green-filtered images and cached. Both splits reuse the embeddings of the final 18,286 retained images.
- `StandardScaler` is fitted exclusively on training features.
- Logistic regression uses scikit-learn's `lbfgs` solver with `max_iter=3000`.
- Regularization is selected from `C = [0.001, 0.01, 0.1, 1, 10, 100]` by validation macro-F1. Both splits select **C = 0.1**; test data are not used for this selection.

The earlier committed classifier uses fixed `C=1.0`, `max_iter=1000`, and balanced class weights; it does not implement the cleaned-data tuning procedure.

## Benchmark Results

| Experiment | Dataset | Split strategy | Val Accuracy | Val Macro-F1 | Test Accuracy | Test Macro-F1 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Original baseline | 21,169 | Random | 92.76% | 89.80% | Not evaluated | Not evaluated |
| Split A | 18,286 | Exact-duplicate-aware | 92.74% | 92.29% | 92.45% | 90.87% |
| Split B | 18,286 | Similarity-grouped | 92.45% | 91.45% | 90.96% | 87.85% |

Both cleaned baselines retain over 90% test accuracy. Split B yields lower test performance, indicating sensitivity to the splitting strategy, particularly for some smaller classes. This difference is not a direct estimate of leakage: A and B contain different test images, and B can group visually similar but independent images. The original baseline also uses a different dataset and reports validation metrics only.

The purpose of cleaning is to make evaluation more reproducible, transparent, and interpretable; it does not necessarily increase accuracy.

## Limitations

- Model-assisted filtering may miss annotations.
- Split A controls detected exact duplicates but may leave near-duplicates across partitions.
- Split B may over-group independent images through visual similarity and transitive connections.
- Specimen/acquisition identifiers are unavailable, so neither split guarantees specimen-level independence.
- Different evaluation partitions prevent a controlled measurement of the cleaning effect.

## Repository Structure and Reproduction

| Location | Contents |
| --- | --- |
| `src/` | Dataset loading, Crop V2, and the frozen DINOv2 encoder |
| `scripts/` | Original baseline, V2 cleaning, feature extraction, duplicate audits, and visual inspection |
| `weights/`, `yolo11n.pt` | Checked-in YOLO weight files; these do not provide the V3 training workflow |
| `outputs/` | Generated experiment artifacts; ignored by Git |

The V3 methodology and results above document completed exploratory experiments. The current repository contains the earlier V2 scripts, but not the V3 cleaning, Union-Find splitting, or tuned classifier training scripts. V3 artifacts are also not checked in, so the reported A/B experiments cannot yet be reproduced end to end from this checkout.

The experiment artifact paths are:

| Path | Contents |
| --- | --- |
| `outputs/dataset_v3/manifest.csv` | Image inclusion, classes, exclusion reasons, and preprocessing settings |
| `outputs/dataset_v3/config.json` | Preprocessing configuration |
| `outputs/dataset_v3/duplicates/candidate_pairs_v3.csv` | Duplicate/similarity candidate pairs |
| `outputs/dataset_v3/splits/split_A_exact.csv` and `.json` | Split A assignments |
| `outputs/dataset_v3/splits/split_B_full.csv` and `.json` | Split B assignments |
| `outputs/dataset_v3/baselines/split_A/` | Trained linear classifier and validation/test results for A |
| `outputs/dataset_v3/baselines/split_B/` | Trained linear classifier and validation/test results for B |

### Running the committed scripts

Run commands from the repository root. Provide the dataset separately in `ImageFolder` layout, with one directory per class. Set `DATASET_ROOT` in `src/dataset.py` and the independent dataset-root constants in the audit scripts to your local path; the current default is `/mnt/data/stud-ueqio/datasets/nffa_sem_100`.

`requirements.txt` is currently empty. Install the packages used by the commands below in a Python environment:

```bash
python -m pip install torch torchvision numpy Pillow scikit-learn matplotlib tqdm ImageHash
```

DINOv2 is loaded through `torch.hub`; the initial download of model code and pretrained weights requires internet access. CUDA extraction requires a compatible PyTorch installation.

For a small smoke experiment, using 5/2/2 images per class from the original splits:

```bash
python -m scripts.create_splits
python -m scripts.run_baseline
```

This saves `outputs/features_smoke.npz`; it does not reproduce the full-data benchmark table.

The committed V2 workflow can be run after creating the original splits:

```bash
python -m scripts.audit_green_overlays --workers 8
python -m scripts.build_dataset_v2
python -m scripts.audit_duplicates
python -m scripts.attach_sha_groups
python -m scripts.extract_all_features --device cpu --batch-size 64 --chunk-size 512 --num-workers 8
python -m scripts.audit_dataset_v2_phash
python -m scripts.audit_global_phash --workers 8 --max-distance 8
python -m scripts.audit_dinov2_neighbors --device cpu --top-k 10
```

This produces a V2 manifest, resumable feature chunks under `outputs/dataset_v2/features/`, and audit reports. It does not perform the YOLO exclusions or generate V3 splits. Use `--device cuda` for supported GPU feature extraction; chunk size must be divisible by batch size.

`python -m scripts.train_classifier` runs the original full-data classifier only when all 21,169 legacy embeddings are available under `outputs/features_dinov2_vits14/`, together with `outputs/splits.json`. It reports training/validation metrics and does not consume the V2/V3 filtered feature sets.

## Next Steps

Planned experiments include a k-NN baseline, comparisons of DINOv2 model sizes and input resolutions, DINOv2 fine-tuning on SEM images, and evaluation of microscopy-specific pretrained encoders.
