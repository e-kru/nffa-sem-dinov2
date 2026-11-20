# SEM Representation Learning with DINOv2

Frozen encoder comparisons now have a shared runner with configurable existing
heads: classification `--heads logreg mlp`, segmentation `--heads linear cnn8`.
See [Downstream evaluation and manual A100 commands](docs/downstream_evaluation.md).

Der [vollständige Bildaudit](docs/dataset_full_inventory.md) inventarisiert neun
weitere Datensätze anhand ihrer Originalauflösungen und prüft exakte
SHA-256-Dateiduplikate; NFFA und Aachen bleiben davon ausgeschlossen.

This research project investigates frozen DINOv2 representations for scanning electron microscopy (SEM) and materials-science microstructures. It evaluates image classification on NFFA-SEM and semantic segmentation on Aachen-Heerlen, with controlled experiments on input resolution and cropping.

## Key Findings

- **Classification.** DINOv2 ViT-S/14 with an MLP reached **94.9% macro-F1** on NFFA Split A validation after 500 SEM adaptation updates.
- **Segmentation.** Frozen ViT-L/14 features with a CNN8 decoder reached **68.6% mIoU** on Aachen-Heerlen validation at 896 × 672.
- **Domain adaptation.** One epoch on 172,300 SEM images improved Aachen CNN8 MA-IoU by **1.34 percentage points** over the original ViT-S/14.

These are single-seed validation results. Segmentation thresholds were selected on validation.

## Method

Downstream heads train on frozen DINOv2 features. Global embeddings feed classification heads, and spatial patch features feed linear or CNN8 segmentation heads. Resolution experiments process complete images, while cropping uses four non-overlapping quadrants and one shared decoder, reconstructing predictions at original resolution.

In the seed-42 comparison, increasing input resolution improves segmentation performance, particularly with CNN8. Four high-resolution crops improve over the low-resolution full image but perform slightly below the highest-resolution full image.

## Repository / Usage

Python/PyTorch workflows support cached features and versioned experiment outputs. See the [pipeline guide](PIPELINE.md) for downstream usage and the [resolution/cropping documentation](docs/aachen_resolution_cropping_ablation.md) for extraction, training, evaluation, and reproducibility details.

For bounded SEM self-supervised post-training checks using a pinned external
checkout of Meta's original DINOv2 objectives, see the [SSL pilot guide](docs/sem_ssl_pilot.md).
The pilot tests the official trainer first and provides a single-device adapter
for environments where that trainer cannot run on one GPU.

### Local inputs

Use a Python environment with the project requirements. Bash entrypoints use
`python` from that environment. Set `PYTHON` to select another interpreter.

- `NFFA_SEM_ROOT` selects the NFFA dataset directory. `SEM_DATA_ROOT` sets the
  parent directory for the datasets used by the Python tools.
- `DINOV2_ROOT` selects the clean Meta checkout pinned in `dinov2-upstream.json`.
- `DINOV2_VITS14_CHECKPOINT` selects the original ViT-S/14 checkpoint. Its default
  location is `outputs/pretrained/dinov2_vits14_pretrain.pth`.
- `DINOV2_LEGACY_HUB` selects the legacy checkout for encoder equivalence checks.

Provide approved datasets and model weights locally. The real-data experiment
scripts also require the generated manifests, split tables and verification
receipts described in their guides. Paths under `outputs/` in the guides refer
to local outputs.

The detailed [pipeline and file guide (German)](docs/sem_ssl_pipeline.md) explains
data selection, augmentation, student/teacher updates, losses, configuration,
checkpoints, export, and each relevant file's responsibility.

The current 500-step NFFA ViT-S/14 entrypoint defaults to Aug-A, the SEM
baseline recipe with its existing crops and brightness/contrast jitter.
Run `bash scripts/run_sem_augmentation_vits14.sh` for that baseline, or
pass `--stage run` for the full 2×2 study. See the [augmentation guide](docs/sem_augmentation_vits14.md).

The [200-step SEM experiment guide](docs/sem_ssl_200step_experiment.md) documents
the dedicated resumable trainer, immutable encoder milestones at 0/50/100/200,
unchanged Meta augmentations and independent validation benchmarks. It includes
the real-NFFA A100 smoke test and all commands; GPU execution remains manual.

Use the separate [A100 batch-size benchmark](docs/sem_ssl_a100_benchmark.md) to
measure complete SSL training throughput and VRAM for physical batches
8/16/32/64 before choosing a longer-run configuration.

For loader-limited B32/B64 runs, see the [loader profiling and shared decoded
image-cache guide](docs/sem_ssl_loader_throughput.md). All tuning options preserve
the original Meta augmentations and remain optional.

The [Low LR / Freeze 8 ablation guide](docs/sem_ssl_ablations.md) records the
teacher audit and the prepared three-run study, including recovery, unchanged
validation protocols and the single manual A100 command. The new GPU runs have
not been started.

The new [SEM Cellstyle ViT-L/14 recipe](docs/sem_cellstyle_vitl14.md) prepares
518/224 multicrop adaptation, absolute backbone/head learning rates, accumulated
batch128, update-wise centering, recovery and 1024-feature validation. Its corpus
interface and physical batch are intentionally unresolved; it has not started.

The [Large L2-SP / feature-distillation study](docs/sem_cellstyle_regularization.md)
extends that recipe with four controlled arms, shared training-only gradient
calibration, complete GPU checks and one resumable manual Bash entrypoint.
The corpus, calibrated strengths and real A100 checks remain prerequisites.

The separate [NFFA Large pilot](docs/sem_cellstyle_nffa_pilot.md) connects this
four-arm study to the existing 12,801 Split-A NFFA training originals, without
waiting for the mixed SEM corpus. Its A100 entrypoint performs the required GPU
checks and calibration before starting any scientific training.
Split A is now the default for new NFFA SSL runs and their automatic NFFA
validation; historical A/B experiments retain their original configurations.
The current NFFA pilot uses [residual downstream heads with 10% dropout](docs/residual_downstream_heads.md):
a new residual MLP and the existing residual CNN8, consistently across all encoder checkpoints.
