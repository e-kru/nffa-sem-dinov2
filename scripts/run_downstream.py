"""Shared training CLI: classification/segmentation × linear/MLP.

Examples in PIPELINE.md. This command uses TRAIN and VALIDATION only.
"""

import argparse
import json
from pathlib import Path

import joblib

from src.config import DATASET_ROOT
from src.downstream.data import (
    ClassificationImageDataset, SegmentationDataset,
    load_rows, make_cached_classification,
)
from src.downstream.engine import set_seed, train
from src.downstream.heads import make_head
from src.downstream.model import DINO_DIMS, DINOv2Backbone, DownstreamModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["classification", "segmentation"], required=True)
    parser.add_argument("--head", choices=["linear", "mlp"], required=True)
    parser.add_argument("--mode", choices=["cached", "image"], default="cached")
    parser.add_argument("--split", choices=["A", "B"], default="A")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--feature-dir", type=Path, default=Path("outputs/dataset_v2/features"))
    parser.add_argument("--image-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--encoder", choices=list(DINO_DIMS), default="dinov2_vits14")
    parser.add_argument("--encoder-checkpoint", type=Path, default=None)
    parser.add_argument("--train-encoder", action="store_true",
                        help="Supervised fine-tuning, NOT self-supervised posttraining")
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[512, 256, 128])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if args.task == "segmentation" and args.mode != "image":
        parser.error("Segmentation requires --mode image (global caches have no pixel layout).")
    if args.mode == "cached" and (args.train_encoder or args.encoder_checkpoint):
        parser.error("Cached features belong to original encoder. Use --mode image with "
                     "new encoder weights, or regenerate caches.")
    if args.task == "segmentation" and args.manifest is None:
        parser.error("Segmentation requires --manifest with pixel-mask labels.")
    if args.num_classes is None:
        if args.task == "segmentation":
            parser.error("Segmentation requires explicit --num-classes.")
        args.num_classes = 10
    if args.num_classes < 2:
        parser.error("--num-classes must be >= 2")
    if args.batch_size is None:
        args.batch_size = 256 if args.mode == "cached" else 8

    # Seed BEFORE model/initialization (important for reproducibility).
    set_seed(args.seed)
    if args.task == "classification":
        manifest = args.manifest or Path(
            f"outputs/dataset_v3/splits/split_{args.split}_"
            + ("exact.csv" if args.split == "A" else "full.csv")
        )
        rows = load_rows(manifest)
        if args.mode == "cached":
            train_data, val_data, scaler = make_cached_classification(rows, args.feature_dir)
            input_dim = train_data.X.shape[1]
            if input_dim != DINO_DIMS[args.encoder]:
                parser.error(f"Cached feature dimension {input_dim} != {args.encoder} "
                             f"dimension {DINO_DIMS[args.encoder]}")
            model = make_head("classification", args.head, input_dim,
                              args.num_classes, args.hidden_dims, args.dropout)
        else:
            train_data = ClassificationImageDataset(rows, "train", args.image_root)
            val_data = ClassificationImageDataset(rows, "val", args.image_root)
    else:
        train_data = SegmentationDataset(args.manifest, "train", args.ignore_index)
        val_data = SegmentationDataset(args.manifest, "val", args.ignore_index)

    if args.mode == "image":
        backbone = DINOv2Backbone(args.encoder, frozen=not args.train_encoder,
                                  checkpoint=args.encoder_checkpoint)
        model = DownstreamModel(backbone, args.task, args.head, args.num_classes,
                                args.hidden_dims, args.dropout)

    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    print(f"Task={args.task} Head={args.head} Mode={args.mode} "
          f"Train={len(train_data)} Val={len(val_data)}")
    summary = train(
        model, train_data, val_data, output=args.output, task=args.task,
        num_classes=args.num_classes, epochs=args.epochs, patience=args.patience,
        batch_size=args.batch_size, lr=args.lr, encoder_lr=args.encoder_lr,
        weight_decay=args.weight_decay, ignore_index=args.ignore_index,
        seed=args.seed, workers=args.num_workers, cached=(args.mode == "cached"),
        train_encoder=args.train_encoder, config=config,
    )
    if args.mode == "cached":
        joblib.dump(scaler, args.output / "scaler.joblib")
    print("\nBest validation:", json.dumps(summary, indent=2))
    print("Test partition not accessed.")


if __name__ == "__main__":
    main()
