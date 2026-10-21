
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

from src.dataset import DATASET_ROOT
from src.dinov2_encoder import get_dinov2, get_dinov2_transform



PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MANIFEST = Path("outputs/dataset_v2/manifest.csv")
DEFAULT_OUTPUT = Path("outputs/dataset_v2/features")


class SEMDatasetV2(Dataset):
    """Dataset with explicit V2 cropping and DINOv2 preprocessing."""

    def __init__(self, records):
        self.samples = [
            (
                Path(DATASET_ROOT) / row["image_id"],
                int(row["label"]),
            )
            for row in records
        ]

        self.transform = get_dinov2_transform()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]

        with Image.open(path) as img:
            image = img.convert("RGB")
        
        # Crop V2 and DINOv2 preprocessing
        image = self.transform(image)

        return image, label


def load_included_records(manifest_path):
    """Load only images accepted by the green-overlay filter."""

    with open(manifest_path, newline="") as f:
        records = list(csv.DictReader(f))

    image_ids = [row["image_id"] for row in records]

    if len(image_ids) != len(set(image_ids)):
        raise ValueError("Duplicate image IDs in manifest")

    included = [
        row for row in records
        if row["included"] == "1"
    ]

    if not included:
        raise ValueError("No included images found")

    return included


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    args = parser.parse_args()

    if args.batch_size <= 0 or args.chunk_size <= 0:
        raise ValueError("Batch and chunk sizes must be positive")

    if args.num_workers < 0:
        raise ValueError("num-workers must be nonnegative")

    if args.chunk_size % args.batch_size != 0:
        raise ValueError(
            "chunk-size must be divisible by batch-size"
        )

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    device = torch.device(args.device)

    print(f"Using device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --------------------------------------------------
    # 1. Load Dataset V2 manifest
    # --------------------------------------------------

    records = load_included_records(args.dataset_manifest)

    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        records = records[:args.limit]

    dataset = SEMDatasetV2(records)
    n = len(dataset)

    print(f"Selected images: {n}")
    print("Preprocessing: crop_metadata_strip V2")
    print("Model: dinov2_vits14")

    # --------------------------------------------------
    # 2. Prepare output directory and fingerprint
    # --------------------------------------------------

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    source_files = [
        PROJECT_ROOT / "src/preprocessing_v2.py",
        PROJECT_ROOT / "src/dinov2_encoder.py",
    ]

    fingerprint = hashlib.sha256()

    # Fingerprint selected image paths and labels
    for row in records:
        fingerprint.update(
            f"{row['image_id']}:{row['label']}\n".encode()
        )

    # Fingerprint preprocessing/model source code
    for source in source_files:
        fingerprint.update(source.read_bytes())

    # Fingerprint explicit transform configuration
    fingerprint.update(
        repr(dataset.transform).encode()
    )

    manifest = {
        "model": "dinov2_vits14",
        "dataset": "nffa_sem_v2",
        "num_images": n,
        "preprocessing": "crop_metadata_strip_v2",
        "chunk_size": args.chunk_size,
        "fingerprint": fingerprint.hexdigest(),
    }

    manifest_path = output_dir / "manifest.json"

    if manifest_path.exists():
        with open(manifest_path) as f:
            previous = json.load(f)

        if previous != manifest:
            raise RuntimeError(
                "Dataset, preprocessing or chunk configuration changed. "
                "Use a new output directory."
            )
    else:
        temporary_manifest = manifest_path.with_suffix(".json.tmp")
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2)
        )
        os.replace(temporary_manifest, manifest_path)

    # --------------------------------------------------
    # 3. Load frozen DINOv2
    # --------------------------------------------------

    model = get_dinov2().to(device)
    model.eval()

    # --------------------------------------------------
    # 4. Extract features in resumable chunks
    # --------------------------------------------------

    for start in range(0, n, args.chunk_size):
        end = min(start + args.chunk_size, n)

        output_file = output_dir / (
            f"chunk_{start:06d}_{end:06d}.npz"
        )

        indices = range(start, end)

        expected_paths = [
            records[i]["image_id"]
            for i in indices
        ]

        expected_labels = np.array([
            int(records[i]["label"])
            for i in indices
        ])

        # Resume: skip previously completed chunks
        if output_file.exists():
            with np.load(output_file, allow_pickle=False) as saved:
                valid = (
                    saved["X"].shape == (end - start, 384)
                    and np.array_equal(
                        saved["paths"],
                        np.asarray(expected_paths),
                    )
                    and np.array_equal(
                        saved["y"],
                        expected_labels,
                    )
                )

            if not valid:
                raise RuntimeError(
                    f"Invalid checkpoint: {output_file}"
                )

            print(f"Skipping completed chunk {start}:{end}")
            continue

        loader = DataLoader(
            Subset(dataset, indices),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )

        features = []
        labels = []

        with torch.inference_mode():
            for images, y in tqdm(
                loader,
                desc=f"Images {start}:{end}",
            ):
                images = images.to(
                    device,
                    non_blocking=(device.type == "cuda"),
                )

                z = model(images)

                features.append(z.cpu().numpy())
                labels.append(y.numpy())

        X = np.concatenate(features, axis=0)
        y = np.concatenate(labels, axis=0)

        if X.shape != (end - start, 384):
            raise RuntimeError(
                f"Unexpected feature shape: {X.shape}"
            )

        if not np.array_equal(y, expected_labels):
            raise RuntimeError("Label mismatch during extraction")

        # Save atomically
        temporary = output_file.with_suffix(".npz.tmp")

        with open(temporary, "wb") as f:
            np.savez(
                f,
                X=X,
                y=y,
                paths=np.asarray(expected_paths),
            )

        os.replace(temporary, output_file)

        print(f"Saved {len(X)} features to {output_file}")

    print("\nFeature extraction completed!")
    print(f"Images processed: {n}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
