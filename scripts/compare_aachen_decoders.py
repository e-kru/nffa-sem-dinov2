"""Compare four frozen-DINOv2 decoders on Aachen-Heerlen."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from scripts.train_aachen_cached_segmentation import (
    DEFAULT_CACHE, DEFAULT_MANIFEST, HEIGHT, WIDTH,
    DenseCachedSegmentationDataset, sha256_file,
)
from src.downstream.data import load_rows
from src.downstream.engine import train, set_seed


def block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
        nn.GroupNorm(8, out_c),
        nn.GELU(),
        nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
        nn.GroupNorm(8, out_c),
        nn.GELU(),
    )


class PointwiseMLP(nn.Module):
    """Nonlinear classifier applied independently at each patch."""

    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, 256, 1),
            nn.GELU(),
            nn.Conv2d(256, 128, 1),
            nn.GELU(),
            nn.Conv2d(128, 2, 1),
        )

    def forward(self, x):
        return self.net(x)


class SpatialCNN(nn.Module):
    """Spatial decoder with two 3x3 convolutions."""

    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, 128, 3, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 2, 1),
        )

    def forward(self, x):
        return self.net(x)


class FeatureUNet(nn.Module):
    """U-Net-like decoder operating on cached 24x32 features."""

    def __init__(self, in_dim):
        super().__init__()
        self.enc0 = block(in_dim, 64)
        self.enc1 = block(64, 96)
        self.bottom = block(96, 128)
        self.dec1 = block(128 + 96, 96)
        self.dec0 = block(96 + 64, 64)
        self.classifier = nn.Conv2d(64, 2, 1)

    def forward(self, x):
        # Encoder: 24x32 -> 12x16 -> 6x8
        s0 = self.enc0(x)
        s1 = self.enc1(F.avg_pool2d(s0, 2))
        z = self.bottom(F.avg_pool2d(s1, 2))

        # Decoder with skip-connections
        z = F.interpolate(
            z, size=s1.shape[-2:],
            mode="bilinear", align_corners=False
        )
        z = self.dec1(torch.cat((z, s1), dim=1))

        z = F.interpolate(
            z, size=s0.shape[-2:],
            mode="bilinear", align_corners=False
        )
        z = self.dec0(torch.cat((z, s0), dim=1))

        return self.classifier(z)


class TransformerDecoder(nn.Module):
    """Task-specific self-attention over DINOv2 patch tokens."""

    def __init__(self, in_dim, patch_height=24, patch_width=32):
        super().__init__()

        dim = 128
        self.patch_height = patch_height
        self.patch_width = patch_width

        self.projection = nn.Conv2d(in_dim, dim, 1)

        self.positions = nn.Parameter(
            torch.zeros(1, patch_height * patch_width, dim)
        )
        nn.init.trunc_normal_(self.positions, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.attention = nn.TransformerEncoder(
            layer, num_layers=2,
            enable_nested_tensor=False,
        )

        self.norm = nn.LayerNorm(dim)
        self.classifier = nn.Conv2d(dim, 2, 1)

    def forward(self, x):
        z = self.projection(x)
        b, c, h, w = z.shape

        if (h, w) != (self.patch_height, self.patch_width):
            raise ValueError(f"Wrong patch grid: {(h, w)}")

        # Feature grid -> token sequence
        z = z.flatten(2).transpose(1, 2)
        z = z + self.positions

        z = self.norm(self.attention(z))

        # Token sequence -> feature grid
        z = z.transpose(1, 2).reshape(b, c, h, w)

        return self.classifier(z)


class DecoderModel(nn.Module):
    """Shared interface for all four decoders."""

    def __init__(self, in_dim, kind):
        super().__init__()

        decoders = {
            "mlp": PointwiseMLP,
            "cnn": SpatialCNN,
            "unet": FeatureUNet,
            "transformer": TransformerDecoder,
        }

        self.decoder = decoders[kind](in_dim)

    def forward(self, features):
        logits = self.decoder(features)

        return F.interpolate(
            logits,
            size=(HEIGHT, WIDTH),
            mode="bilinear",
            align_corners=False,
        )


def safe_div(a, b):
    return float(a) / float(b) if b else 0.0


def metrics(tn, fp, fn, tp):
    ma = safe_div(tp, tp + fp + fn)
    bg = safe_div(tn, tn + fp + fn)

    return {
        "ma_iou": ma,
        "background_iou": bg,
        "miou": (ma + bg) / 2,
        "precision": safe_div(tp, tp + fp),
        "recall": safe_div(tp, tp + fn),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


@torch.inference_mode()
def evaluate_thresholds(
    model, val_ds, device, batch_size, num_workers
):
    """Evaluate 19 thresholds in one validation pass."""

    model.eval().to(device)

    thresholds = np.arange(1, 20, dtype=np.float64) / 20.0
    hist = np.zeros(
        (2, len(thresholds) + 1), dtype=np.int64
    )

    loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    for features, target in loader:
        logits = model(
            features.to(device, non_blocking=True)
        )

        p_ma = torch.sigmoid(
            logits[:, 1] - logits[:, 0]
        ).cpu().numpy().reshape(-1)

        y = target.numpy().reshape(-1)
        valid = (y == 0) | (y == 1)

        if not np.isfinite(p_ma[valid]).all():
            raise RuntimeError("Nonfinite predictions")

        bins = np.searchsorted(
            thresholds, p_ma[valid], side="right"
        )

        width = len(thresholds) + 1
        encoded = y[valid].astype(np.int64) * width + bins

        hist += np.bincount(
            encoded, minlength=2 * width
        ).reshape(2, width)

    cumulative = np.cumsum(
        hist[:, ::-1], axis=1
    )[:, ::-1]

    nbg = int(hist[0].sum())
    nma = int(hist[1].sum())

    results = []

    for i, threshold in enumerate(thresholds):
        fp = int(cumulative[0, i + 1])
        tp = int(cumulative[1, i + 1])

        results.append({
            "threshold": round(float(threshold), 2),
            **metrics(nbg - fp, fp, nma - tp, tp),
        })

    return results


def save_csv(path, rows):
    with Path(path).open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0])
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument(
        "--cache-dir", type=Path, default=DEFAULT_CACHE
    )
    p.add_argument(
        "--manifest", type=Path, default=DEFAULT_MANIFEST
    )
    p.add_argument(
        "--output-dir", type=Path,
        default=Path(
            "outputs/aachen_heerlen/runs/decoder_comparison"
        ),
    )
    p.add_argument(
        "--models", nargs="+",
        choices=("mlp", "cnn", "unet", "transformer"),
        default=["mlp", "cnn", "unet", "transformer"],
    )

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--allow-cpu", action="store_true")

    args = p.parse_args()

    if len(set(args.models)) != len(args.models):
        p.error("Duplicate model names")

    if args.limit is not None and args.limit < 2:
        p.error("--limit must be at least 2")

    if (
        args.batch_size < 2
        or args.epochs < 1
        or args.patience < 1
        or args.num_workers < 0
    ):
        p.error("Invalid training parameters")

    if not torch.cuda.is_available() and not args.allow_cpu:
        p.error("CUDA unavailable. Run on a CUDA host.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    cache_dir = args.cache_dir.resolve()
    manifest = args.manifest.resolve()

    # Check provenance and dataset integrity
    cache_meta = json.loads(
        (cache_dir / "cache_metadata.json").read_text(
            encoding="utf-8"
        )
    )

    manifest_hash = sha256_file(manifest)

    if cache_meta.get("manifest_sha256") != manifest_hash:
        raise RuntimeError("Cache/manifest hash mismatch")

    if cache_meta.get("preprocessing_file_sha256") != (
        sha256_file(Path("src/preprocessing_v2.py"))
    ):
        raise RuntimeError("Preprocessing has changed")

    if cache_meta.get("input_shape_per_image") != [
        3, HEIGHT, WIDTH
    ]:
        raise RuntimeError("Wrong image geometry")

    if cache_meta.get("encoder") != "dinov2_vits14":
        raise RuntimeError("Expected ViT-S/14 features")

    if cache_meta.get("limit_per_split") is not None:
        raise RuntimeError("Partial cache is not allowed")

    manifest_rows = load_rows(manifest)

    datasets = {
        split: DenseCachedSegmentationDataset(
            cache_dir,
            manifest,
            split,
            manifest_rows,
            limit=args.limit,
        )
        for split in ("train", "val")
    }

    feature_dim = int(
        datasets["train"].features.shape[1]
    )

    print(
        "Device:", device,
        "| Feature dim:", feature_dim,
        "| Train:", len(datasets["train"]),
        "| Val:", len(datasets["val"]),
        flush=True,
    )

    print("Test data will NOT be evaluated.", flush=True)

    all_rows = []

    for kind in args.models:
        output = args.output_dir / kind

        if output.exists() and any(output.iterdir()):
            raise FileExistsError(
                f"Output already exists: {output}"
            )

        print(f"\n=== TRAINING {kind.upper()} ===", flush=True)

        # Seed before model initialization
        set_seed(args.seed)

        model = DecoderModel(feature_dim, kind)

        n_params = sum(
            t.numel()
            for t in model.parameters()
            if t.requires_grad
        )

        print(
            "Trainable parameters:", n_params,
            flush=True
        )

        config = {
            "architecture": (
                f"frozen_dinov2_dense_features+{kind}_decoder"
            ),
            "feature_cache": str(cache_dir),
            "feature_manifest_sha256": manifest_hash,
            "head": kind,
            "output_shape": [2, HEIGHT, WIDTH],
            "limit": args.limit,
        }

        # Shared trainer: CE loss, AdamW, early stopping
        # Checkpoint selected by validation mIoU at 0.50
        summary = train(
            model,
            datasets["train"],
            datasets["val"],
            output=output,
            task="segmentation",
            num_classes=2,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed,
            workers=args.num_workers,
            cached=True,
            config=config,
        )

        # Load best checkpoint
        state = torch.load(
            output / "best_model.pt",
            map_location="cpu",
            weights_only=True,
        )

        model.load_state_dict(
            state["model_state_dict"],
            strict=True,
        )

        # Evaluate on validation only
        rows = evaluate_thresholds(
            model,
            datasets["val"],
            device,
            args.batch_size,
            args.num_workers,
        )

        save_csv(output / "threshold_metrics.csv", rows)

        at_05 = next(
            row for row in rows
            if row["threshold"] == 0.5
        )

        # Select threshold maximizing validation MA-IoU
        best = max(
            rows,
            key=lambda r: (
                r["ma_iou"],
                -abs(r["threshold"] - 0.5),
            ),
        )

        report = {
            "model": kind,
            "trainable_parameters": n_params,
            "best_epoch": summary["best_epoch"],
            "selection_criterion_epoch": (
                "validation_mIoU_at_0.50"
            ),
            "selection_criterion_threshold": (
                "validation_MA_IoU"
            ),
            "validation_at_0_50": at_05,
            "validation_at_best_threshold": best,
            "test_evaluated": False,
        }

        (output / "comparison_metrics.json").write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )

        all_rows.append({
            "model": kind,
            "params": n_params,
            "best_epoch": summary["best_epoch"],
            "miou_0.50": round(at_05["miou"], 5),
            "ma_iou_0.50": round(at_05["ma_iou"], 5),
            "best_threshold": best["threshold"],
            "miou_best_threshold": round(best["miou"], 5),
            "ma_iou_best_threshold": round(
                best["ma_iou"], 5
            ),
            "precision_best_threshold": round(
                best["precision"], 5
            ),
            "recall_best_threshold": round(
                best["recall"], 5
            ),
        })

        # Update comparison after every completed model
        save_csv(
            args.output_dir / "comparison.csv",
            all_rows,
        )

        print(
            f"{kind}: "
            f"MA-IoU @0.50={at_05['ma_iou']:.4f}, "
            f"best MA-IoU={best['ma_iou']:.4f} "
            f"@threshold={best['threshold']:.2f}",
            flush=True,
        )

    print("\n=== COMPARISON (VAL ONLY) ===")

    for row in all_rows:
        print(row)

    print(
        "\nSaved:",
        args.output_dir / "comparison.csv"
    )

    print("No test data used.")


if __name__ == "__main__":
    main()
