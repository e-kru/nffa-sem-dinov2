# SEM Image Representation Learning with DINOv2

Research project exploring foundation models for scanning electron microscopy (SEM) and microstructure analysis.

The goal is to develop domain adapted visual representations for microscopy images. As a first step, we evaluate pretrained DINOv2 features on the 10-class NFFA-SEM dataset, with a focus on reliable data preparation and evaluation.

## Approach

- **Data engineering:** Curated 21,169 SEM images, removing graphical annotations, measurement boxes, and residual metadata using image preprocessing and a fine-tuned YOLO11n detector. Final dataset: **18,286 images**.
- **Duplicate-aware evaluation:** Identified duplicates and visually similar images using SHA-256, perceptual hashing, and DINOv2 feature similarity. Constructed two group-aware dataset splits to investigate evaluation robustness.
- **Representation learning baseline:** Extracted 384-dimensional embeddings using a frozen **DINOv2 ViT-S/14** encoder and trained a linear classifier.

## Results

| Evaluation | Validation Accuracy | Test Accuracy | Test Macro-F1 |
|---|---:|---:|---:|
| Original random split | 92.76% | — | — |
| Split A — exact-duplicate-aware | 92.74% | **92.45%** | **90.87%** |
| Split B — conservative similarity grouping | 92.45% | **90.96%** | **87.85%** |

Both cleaned-data evaluations achieve over **90% test accuracy** using frozen DINOv2 features.

The original baseline reports validation results only. Because the experiments use different data partitions, their performance differences should not be interpreted as a direct measure of data leakage.

## Next Steps

- Compare DINOv2 architectures and input resolutions.
- Investigate self supervised adaptation to microscopy images.
- Benchmark microscopy-specific foundation models and evaluate feature transfer to additional downstream tasks.

**Status:** Initial classification baselines completed; further representation learning experiments in progress.