"""Train a linear/MLP segmentation probe on cached Aachen-Heerlen DINOv2 patch features.

Place in scripts/train_aachen_cached_segmentation.py and run from repository root.

Example:
    CUDA_VISIBLE_DEVICES=4 python -m scripts.train_aachen_cached_segmentation \
      --cache-dir outputs/aachen_heerlen/features/vits14_448x336_float32 \
      --manifest outputs/aachen_heerlen/splits/group_split_seed42.csv \
      --output outputs/aachen_heerlen/runs/cached_linear_vits14_448x336

IMPORTANT: This trains on TRAIN and validates on VAL. It does not read test features
or test masks. The trained encoder is not involved in this script.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from src.downstream.engine import train
from src.downstream.heads import make_head
from src.downstream.data import load_rows


DEFAULT_CACHE = Path("outputs/aachen_heerlen/features/vits14_448x336_float32")
DEFAULT_MANIFEST = Path("outputs/aachen_heerlen/splits/group_split_seed42.csv")
HEIGHT, WIDTH = 336, 448


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class DenseCachedSegmentationDataset(Dataset):
    """Map a cached (N,D,24,32) row to its expert mask using the index CSV."""

    def __init__(self, cache_dir: Path, manifest: Path, split: str, manifest_rows, limit=None):
        self.cache_dir = Path(cache_dir)
        self.manifest = Path(manifest)
        self.split = split
        self.features = np.load(self.cache_dir / f"{split}.npy", mmap_mode="r", allow_pickle=False)
        with (self.cache_dir / f"{split}_index.csv").open(newline="", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))

        assert self.features.ndim == 4, self.features.shape
        if tuple(self.features.shape[2:]) != (24, 32):
            raise ValueError(f"Wrong patch shape: {self.features.shape} (need 24x32)")
        if self.features.dtype not in (np.float32, np.float16):
            raise ValueError(f"Unsupported cache dtype: {self.features.dtype}")
        if len(self.rows) != len(self.features):
            raise ValueError(f"Cache/index mismatch for {split}")
        if [int(r["cache_index"]) for r in self.rows] != list(range(len(self.rows))):
            raise ValueError(f"Out-of-order cache_index for {split}")
        if any(r["split"] != split for r in self.rows):
            raise ValueError(f"Wrong split in index for {split}")
        if len({r["image_id"] for r in self.rows}) != len(self.rows):
            raise ValueError(f"Duplicate images in cache index for {split}")

        expected = {r["image_id"]: r for r in manifest_rows if r["split"] == split}
        if set(r["image_id"] for r in self.rows) != set(expected):
            raise ValueError(f"Cached IDs don't match manifest IDs for {split}")
        for row in self.rows:
            ref = expected[row["image_id"]]
            if row["mask_path"] != ref["mask_path"] or row["group_id"] != ref["group_id"]:
                raise ValueError(f"Wrong mask/group pairing for {row['image_id']}")
        if limit is not None:
            self.rows = self.rows[:limit]
        if not self.rows:
            raise ValueError(f"Empty dataset {split}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        feature_idx = int(row["cache_index"])
        # Explicit copy: np.memmap is read-only; torch must own a writable buffer.
        features = torch.from_numpy(np.array(self.features[feature_idx], dtype=np.float32, copy=True))
        mask_path = Path(row["mask_path"])
        if not mask_path.is_absolute():
            mask_path = self.manifest.parent / mask_path
        with Image.open(mask_path) as img:
            if img.mode not in ("L", "P"):
                raise ValueError(f"Expected indexed mask, got {img.mode}: {mask_path}")
            mask = TF.resize(img, [HEIGHT, WIDTH], interpolation=InterpolationMode.NEAREST)
            labels = np.array(mask, dtype=np.int64, copy=True)
        if not np.isin(labels, (0, 1, 255)).all():
            raise ValueError(f"Mask has invalid class values: {mask_path}")
        return features, torch.from_numpy(labels)


class CachedSegmentationProbe(nn.Module):
    def __init__(self, dim: int, kind: str):
        super().__init__()
        self.head = make_head(task="segmentation", kind=kind, in_dim=dim,
                              num_classes=2)

    def forward(self, x):
        logits = self.head(x)
        return F.interpolate(logits, size=(HEIGHT, WIDTH),
                             mode="bilinear", align_corners=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--output", type=Path,
                   default=Path("outputs/aachen_heerlen/runs/cached_linear_vits14_448x336"))
    p.add_argument("--head", choices=("linear", "mlp"), default="linear")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None,
                   help="Smoke test only: use first N train and first N validation rows")
    p.add_argument("--allow-cpu", action="store_true", help="For local smoke tests only")
    args = p.parse_args()
    if not torch.cuda.is_available() and not args.allow_cpu:
        p.error("CUDA unavailable. Run on a CUDA host or use --allow-cpu for smoke testing")
    if args.limit is not None and args.limit < 2:
        p.error("--limit must be >= 2")

    cache_dir = args.cache_dir.resolve()
    manifest = args.manifest.resolve()
    meta = json.loads((cache_dir / "cache_metadata.json").read_text(encoding="utf-8"))
    if sha256_file(manifest) != meta["manifest_sha256"]:
        raise RuntimeError("Manifest changed after feature extraction: cache invalid")
    current_pre = sha256_file(Path("src/preprocessing_v2.py"))
    if current_pre != meta["preprocessing_file_sha256"]:
        raise RuntimeError("Preprocessing file changed after extraction: inspect cache consistency")
    if meta.get("input_shape_per_image") != [3, HEIGHT, WIDTH]:
        raise RuntimeError("Wrong cached input geometry")
    if meta.get("limit_per_split") is not None:
        raise RuntimeError("This is a partial feature cache; use a complete cache for training")
    if meta.get("encoder") != "dinov2_vits14":
        raise RuntimeError(f"Unexpected encoder {meta.get('encoder')}")
    manifest_rows = load_rows(manifest)

    datasets = {}
    for split in ("train", "val"):
        datasets[split] = DenseCachedSegmentationDataset(
            cache_dir, manifest, split, manifest_rows, limit=args.limit)
        print(f"{split}: {len(datasets[split])} samples, "
              f"feature shape={datasets[split].features.shape}, "
              f"dtype={datasets[split].features.dtype}", flush=True)
    x, y = datasets["train"][0]
    print(f"Example: features {tuple(x.shape)}, mask {tuple(y.shape)}, "
          f"labels {torch.unique(y).tolist()}", flush=True)

    model = CachedSegmentationProbe(dim=int(x.shape[0]), kind=args.head)
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad),
          flush=True)

    config = {
        "architecture": "frozen_dinov2_dense_features+segmentation_probe",
        "feature_cache": str(cache_dir),
        "feature_manifest_sha256": meta["manifest_sha256"],
        "head": args.head,
        "output_shape": [2, HEIGHT, WIDTH],
        "limit": args.limit,
    }
    summary = train(
        model, datasets["train"], datasets["val"],
        output=args.output, task="segmentation", num_classes=2,
        epochs=args.epochs, patience=args.patience,
        batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, seed=args.seed,
        workers=args.num_workers, cached=True, config=config,
    )
    # The shared trainer reports mIoU. Also report IoU for the foreground class
    # explicitly, since high background accuracy can otherwise be misleading.
    checkpoint = torch.load(args.output / "best_model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    val_loader = DataLoader(datasets["val"], batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)
    cm = torch.zeros((2, 2), dtype=torch.int64)
    with torch.inference_mode():
        for feats, target in val_loader:
            pred = model(feats.to(device)).argmax(dim=1).cpu()
            valid = target != 255
            codes = target[valid].long() * 2 + pred[valid].long()
            cm += torch.bincount(codes, minlength=4).reshape(2, 2)
    tn, fp, fn, tp = (int(v) for v in cm.flatten())
    iou_background = tn / max(tn + fp + fn, 1)
    iou_ma = tp / max(tp + fp + fn, 1)
    metrics = {
        "iou_background": iou_background,
        "iou_ma": iou_ma,
        "miou": (iou_background + iou_ma) / 2,
        "ma_precision": tp / max(tp + fp, 1),
        "ma_recall": tp / max(tp + fn, 1),
        "confusion_matrix_true_rows_pred_columns": cm.tolist(),
        "split": "val",
        "samples": len(datasets["val"]),
    }
    (args.output / "val_class_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print("TRAINING COMPLETE:", json.dumps(summary, indent=2), flush=True)
    print("VALIDATION CLASS METRICS:", json.dumps(metrics, indent=2), flush=True)
    print("Test set was NOT evaluated.", flush=True)


if __name__ == "__main__":
    main()
