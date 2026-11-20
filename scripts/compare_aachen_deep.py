"""Depth ablation for frozen DINOv2 Aachen-Heerlen segmentation.

Save as scripts/compare_aachen_deep.py in the existing repository.
Requires scripts/compare_aachen_decoders.py and the cached segmentation pipeline.

All methods use the SAME frozen 384x24x32 features and train/val group split.
Test split is never loaded. Epoch picked by val mIoU at threshold 0.50,
then operating threshold picked by val MA-IoU (as in prior decoder comparison).
"""

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from scripts.compare_aachen_decoders import evaluate_thresholds, save_csv
from scripts.train_aachen_cached_segmentation import (
    DEFAULT_CACHE, DEFAULT_MANIFEST, HEIGHT, WIDTH,
    DenseCachedSegmentationDataset, sha256_file,
)
from src.downstream.heads import DeepCNN, SpatialResidual
from src.downstream.data import load_rows
from src.downstream.engine import set_seed, train


class MLPResidual(nn.Module):
    """Only 1x1 convolutions: no direct inter-patch communication."""

    def __init__(self, channels, dropout=0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.GELU(), nn.Conv2d(channels, channels, 1),
            nn.GELU(), nn.Dropout(dropout),
            nn.Conv2d(channels, channels, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return x + self.net(x)


class DeepMLP(nn.Module):
    """8 or 16 pointwise linear layers (including projection and output)."""

    def __init__(self, in_channels, layers):
        super().__init__()
        assert layers in (8, 16)
        channels = 192
        self.proj = nn.Conv2d(in_channels, channels, 1)
        self.blocks = nn.Sequential(*[
            MLPResidual(channels) for _ in range((layers - 2) // 2)
        ])
        self.classifier = nn.Conv2d(channels, 2, 1)

    def forward(self, x):
        return self.classifier(self.blocks(self.proj(x)))



def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.GroupNorm(8, out_channels), nn.GELU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.GroupNorm(8, out_channels), nn.GELU(),
    )


class DeepFeatureUNet(nn.Module):
    """28 spatial 3x3 convolutions, all on the frozen feature grid.

    Four scales: 24x32, 12x16, 6x8, 3x4. No raw-image skip features.
    """

    def __init__(self, in_channels):
        super().__init__()
        widths = (64, 96, 128, 160)
        a, b, c, d = widths
        self.e0 = nn.Sequential(double_conv(in_channels, a), SpatialResidual(a))
        self.e1 = nn.Sequential(double_conv(a, b), SpatialResidual(b))
        self.e2 = nn.Sequential(double_conv(b, c), SpatialResidual(c))
        self.bottom = nn.Sequential(double_conv(c, d), SpatialResidual(d))
        self.d2 = nn.Sequential(double_conv(d + c, c), SpatialResidual(c))
        self.d1 = nn.Sequential(double_conv(c + b, b), SpatialResidual(b))
        self.d0 = nn.Sequential(double_conv(b + a, a), SpatialResidual(a))
        self.classifier = nn.Conv2d(a, 2, 1)

    @staticmethod
    def up(x, skip):
        return torch.cat((
            F.interpolate(x, size=skip.shape[-2:],
                          mode="bilinear", align_corners=False),
            skip,
        ), dim=1)

    def forward(self, x):
        s0 = self.e0(x)
        s1 = self.e1(F.avg_pool2d(s0, 2))
        s2 = self.e2(F.avg_pool2d(s1, 2))
        z = self.bottom(F.avg_pool2d(s2, 2))
        z = self.d2(self.up(z, s2))
        z = self.d1(self.up(z, s1))
        z = self.d0(self.up(z, s0))
        return self.classifier(z)


class DeepTransformer(nn.Module):
    """6 task-specific self-attention blocks, 4 heads, dim=128."""

    def __init__(self, in_channels, layers=6):
        super().__init__()
        dim, h, w = 128, HEIGHT // 14, WIDTH // 14
        assert (h, w) == (24, 32)
        self.hw = (h, w)
        self.proj = nn.Conv2d(in_channels, dim, 1)
        self.positions = nn.Parameter(torch.zeros(1, h * w, dim))
        nn.init.trunc_normal_(self.positions, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=4, dim_feedforward=256,
            dropout=0.1, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(
            layer, num_layers=layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(dim)
        self.classifier = nn.Conv2d(dim, 2, 1)

    def forward(self, x):
        x = self.proj(x)
        b, c, h, w = x.shape
        if (h, w) != self.hw:
            raise ValueError(f"Wrong token shape: {(h, w)}")
        tokens = x.flatten(2).transpose(1, 2) + self.positions
        tokens = self.norm(self.blocks(tokens))
        return self.classifier(tokens.transpose(1, 2).reshape(b, c, h, w))


MODELS = ("mlp8", "mlp16", "cnn8", "cnn16", "unet28", "transformer6")


class Model(nn.Module):
    def __init__(self, in_channels, kind):
        super().__init__()
        if kind.startswith("mlp"):
            self.decoder = DeepMLP(in_channels, int(kind[3:]))
        elif kind.startswith("cnn"):
            self.decoder = DeepCNN(in_channels, int(kind[3:]))
        elif kind == "unet28":
            self.decoder = DeepFeatureUNet(in_channels)
        elif kind == "transformer6":
            self.decoder = DeepTransformer(in_channels, 6)
        else:
            raise ValueError(kind)

    def forward(self, x):
        return F.interpolate(self.decoder(x), size=(HEIGHT, WIDTH),
                             mode="bilinear", align_corners=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", choices=MODELS,
                   default=["cnn8", "cnn16", "unet28", "transformer6"])
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--output-dir", type=Path,
                   default=Path("outputs/aachen_heerlen/runs/deep_comparison"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--allow-cpu", action="store_true")
    args = p.parse_args()

    if len(set(args.models)) != len(args.models):
        p.error("Duplicate models")
    if args.limit is not None and args.limit < 2:
        p.error("--limit must be at least 2")
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 2 or args.num_workers < 0:
        p.error("Invalid train arguments")
    if not torch.cuda.is_available() and not args.allow_cpu:
        p.error("CUDA not found; use --allow-cpu for smoke test")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache, manifest = args.cache_dir.resolve(), args.manifest.resolve()
    meta = json.loads((cache / "cache_metadata.json").read_text(encoding="utf-8"))
    manifest_hash = sha256_file(manifest)
    if meta.get("manifest_sha256") != manifest_hash:
        raise RuntimeError("Manifest/cache mismatch")
    if meta.get("preprocessing_file_sha256") != sha256_file(Path("src/preprocessing_v2.py")):
        raise RuntimeError("Preprocessing hash mismatch")
    if meta.get("input_shape_per_image") != [3, HEIGHT, WIDTH]:
        raise RuntimeError("Unexpected image geometry")
    if meta.get("encoder") != "dinov2_vits14" or meta.get("limit_per_split") is not None:
        raise RuntimeError("Unexpected encoder or partial cache")

    rows = load_rows(manifest)
    train_ds = DenseCachedSegmentationDataset(cache, manifest, "train", rows, limit=args.limit)
    val_ds = DenseCachedSegmentationDataset(cache, manifest, "val", rows, limit=args.limit)
    dim = int(train_ds.features.shape[1])
    print(f"Device={device}, Train={len(train_ds)}, Val={len(val_ds)}, Dim={dim}")
    print("NO TEST DATA ARE USED.", flush=True)

    comparison = []
    for kind in args.models:
        output = args.output_dir / kind
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Output exists: {output}. Use --output-dir for a new run")
        set_seed(args.seed)
        model = Model(dim, kind)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n=== {kind}: {params:,} trainable params ===", flush=True)
        config = {
            "architecture": f"frozen_dinov2+{kind}",
            "feature_cache": str(cache),
            "feature_manifest_sha256": manifest_hash,
            "head": kind,
            "output_shape": [2, HEIGHT, WIDTH],
            "limit": args.limit,
        }
        summary = train(
            model, train_ds, val_ds,
            output=output, task="segmentation", num_classes=2,
            epochs=args.epochs, patience=args.patience,
            batch_size=args.batch_size, lr=args.lr,
            weight_decay=args.weight_decay, seed=args.seed,
            workers=args.num_workers, cached=True, config=config,
        )
        state = torch.load(output / "best_model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state["model_state_dict"], strict=True)
        scores = evaluate_thresholds(model, val_ds, device, args.batch_size, args.num_workers)
        save_csv(output / "threshold_metrics.csv", scores)
        t05 = next(r for r in scores if r["threshold"] == 0.5)
        best = max(scores, key=lambda r: (r["ma_iou"], -abs(r["threshold"] - 0.5)))
        report = {
            "model": kind, "trainable_parameters": params,
            "best_epoch": summary["best_epoch"], "lr": args.lr,
            "validation_at_0_50": t05,
            "validation_at_best_threshold": best,
            "test_evaluated": False,
        }
        (output / "comparison_metrics.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        comparison.append({
            "model": kind, "params": params,
            "best_epoch": summary["best_epoch"],
            "miou_0.50": round(t05["miou"], 5),
            "ma_iou_0.50": round(t05["ma_iou"], 5),
            "threshold": best["threshold"],
            "miou": round(best["miou"], 5),
            "ma_iou": round(best["ma_iou"], 5),
            "precision": round(best["precision"], 5),
            "recall": round(best["recall"], 5),
        })
        save_csv(args.output_dir / "comparison.csv", comparison)
        print(f"{kind}: best val MA-IoU={best['ma_iou']:.4f} at t={best['threshold']:.2f}", flush=True)
    print("\n=== COMPLETE VALIDATION COMPARISON ===")
    for row in comparison:
        print(row)
    print("Saved:", args.output_dir / "comparison.csv")
    print("TEST SPLIT NOT EVALUATED")


if __name__ == "__main__":
    main()
