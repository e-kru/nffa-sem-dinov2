"""Extract dense DINOv2 patch features for Aachen-Heerlen segmentation.

Run from the nffa-sem-dinov2 repository root after placing this script in scripts/:

    CUDA_VISIBLE_DEVICES=4 python -m scripts.extract_aachen_dense_features \
        --manifest outputs/aachen_heerlen/splits/group_split_seed42.csv \
        --output-dir outputs/aachen_heerlen/features/vits14_448x336_float32 \
        --batch-size 16 --num-workers 4

The pipeline uses the repository's SegmentationDataset to guarantee that
cached features use EXACTLY the same 448x336 image preprocessing as training.
Only the frozen backbone is computed; segmentation labels are never used to
compute features. This script does not train or evaluate a model.

Each split is a .npy memory-mappable tensor (N,384,24,32) in the same order as
<split>_index.csv. Files are written atomically (temp then rename).
"""

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.downstream.data import SegmentationDataset, load_rows
from src.downstream.model import DINOv2Backbone


SPLITS = ("train", "val", "test")
HEIGHT, WIDTH = 336, 448
FEATURE_HEIGHT, FEATURE_WIDTH = HEIGHT // 14, WIDTH // 14


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for part in iter(lambda: file.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=Path("outputs/aachen_heerlen/splits/group_split_seed42.csv"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/aachen_heerlen/features/vits14_448x336_float32"))
    parser.add_argument("--encoder", default="dinov2_vits14",
                        choices=("dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"))
    parser.add_argument("--encoder-checkpoint", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--limit", type=int, default=None,
                        help="Smoke test only: process up to N images per split, use separate output directory")
    args = parser.parse_args()

    if args.batch_size <= 0 or args.num_workers < 0:
        parser.error("Invalid batch size or worker count")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Use a CUDA environment.")

    manifest = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = load_rows(manifest)
    print("CUDA device:", torch.cuda.get_device_name(0), flush=True)
    print("Manifest:", manifest, flush=True)
    print("Cache destination:", output_dir, flush=True)

    # Match the model used by src/downstream/model.py, frozen for a valid probe.
    backbone = DINOv2Backbone(args.encoder, frozen=True, checkpoint=args.encoder_checkpoint)
    backbone = backbone.to("cuda:0").eval()
    feature_dim = backbone.feature_dim

    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "encoder": args.encoder,
        "encoder_checkpoint": str(args.encoder_checkpoint.resolve()) if args.encoder_checkpoint else None,
        "checkpoint_sha256": sha256_file(args.encoder_checkpoint) if args.encoder_checkpoint else None,
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "preprocessing_file_sha256": sha256_file(Path("src/preprocessing_v2.py")),
        "tensor_shape_per_image": [feature_dim, FEATURE_HEIGHT, FEATURE_WIDTH],
        "input_shape_per_image": [3, HEIGHT, WIDTH],
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "splits": {},
        "limit_per_split": args.limit,
        "torch_version": torch.__version__,
    }

    # The dataset never uses test labels for fitting: it merely loads matched
    # pairs; cached features depend only on image tensors, not mask tensors.
    for split in args.splits:
        dataset = SegmentationDataset(manifest, split)
        if args.limit is not None:
            dataset.rows = dataset.rows[:args.limit]
        count = len(dataset)
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
            persistent_workers=(args.num_workers > 0),
        )

        npy_final = output_dir / f"{split}.npy"
        index_final = output_dir / f"{split}_index.csv"
        npy_temp = output_dir / f"{split}.partial.npy"
        index_temp = output_dir / f"{split}_index.partial.csv"
        if npy_final.exists() or index_final.exists():
            raise FileExistsError(
                f"Existing cache output for {split}: {npy_final} or {index_final}. "
                "Use a new output directory to avoid overwriting a previous cache."
            )
        if npy_temp.exists() or index_temp.exists():
            raise FileExistsError(f"Incomplete cache for {split} exists; inspect/delete it: {npy_temp}")

        dtype = np.float32 if args.dtype == "float32" else np.float16
        array = np.lib.format.open_memmap(
            npy_temp, mode="w+", dtype=dtype,
            shape=(count, feature_dim, FEATURE_HEIGHT, FEATURE_WIDTH),
        )

        seen = 0
        with torch.inference_mode():
            for images, masks in loader:
                if images.shape[1:] != (3, HEIGHT, WIDTH):
                    raise ValueError(f"Unexpected image dimensions: {tuple(images.shape)}")
                if masks.shape[1:] != (HEIGHT, WIDTH):
                    raise ValueError(f"Unexpected mask dimensions: {tuple(masks.shape)}")

                features = backbone.forward_dense(images.to("cuda:0", non_blocking=True))
                expected = (len(images), feature_dim, FEATURE_HEIGHT, FEATURE_WIDTH)
                if tuple(features.shape) != expected:
                    raise ValueError(f"Unexpected feature shape: {tuple(features.shape)} vs {expected}")
                if not torch.isfinite(features).all().item():
                    raise ValueError(f"Nonfinite feature values in {split}, sample index {seen}")
                batch = features.cpu().numpy().astype(dtype, copy=False)
                array[seen:seen + len(batch)] = batch
                seen += len(batch)
                if seen % 160 < len(batch) or seen == count:
                    print(f"{split}: {seen}/{count} images", flush=True)

        assert seen == count
        array.flush()
        del array

        with index_temp.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=(
                "cache_index", "image_id", "image_path", "mask_path", "split", "group_id"
            ))
            writer.writeheader()
            for i, row in enumerate(dataset.rows):
                writer.writerow({
                    "cache_index": i,
                    "image_id": row["image_id"],
                    "image_path": row["image_path"],
                    "mask_path": row["mask_path"],
                    "split": split,
                    "group_id": row.get("group_id", ""),
                })

        # Verify the on-disk array can be reopened before publishing.
        reopened = np.load(npy_temp, mmap_mode="r")
        assert reopened.shape == (count, feature_dim, FEATURE_HEIGHT, FEATURE_WIDTH)
        del reopened
        os.replace(npy_temp, npy_final)
        os.replace(index_temp, index_final)
        provenance["splits"][split] = {"count": count,
                                       "features": npy_final.name,
                                       "index": index_final.name}
        print(f"Saved {npy_final} and {index_final}", flush=True)

    metadata_tmp = output_dir / "cache_metadata.partial.json"
    metadata_final = output_dir / "cache_metadata.json"
    if metadata_final.exists():
        raise FileExistsError(f"Metadata already exists: {metadata_final}")
    metadata_tmp.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    os.replace(metadata_tmp, metadata_final)
    print("\nCACHE COMPLETE", flush=True)
    for split, info in provenance["splits"].items():
        print(f"{split:5s}: {info['count']:4d} images", flush=True)
    print("Metadata:", metadata_final, flush=True)


if __name__ == "__main__":
    main()
