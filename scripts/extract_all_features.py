
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.dataset import get_dataset, DATASET_ROOT
from src.dinov2_encoder import (
    get_dinov2,
    get_dinov2_transform,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device", choices=["cpu", "cuda"], default="cpu"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/features_dinov2_vits14")
    )

    args = parser.parse_args()

    if args.batch_size <= 0 or args.chunk_size <= 0:
        raise ValueError("Batch and chunk sizes must be positive")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this environment")

    device = torch.device(args.device)
    print("Using device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # Load dataset and preprocessing
    dataset = get_dataset(
        transform=get_dinov2_transform()
    )

    n = len(dataset)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        n = min(n, args.limit)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fingerprint dataset order and preprocessing code
    source_files = [
        PROJECT_ROOT / "src/preprocessing.py",
        PROJECT_ROOT / "src/dinov2_encoder.py",
    ]

    fingerprint = hashlib.sha256()

    for path, label in dataset.samples[:n]:
        relative = Path(path).relative_to(DATASET_ROOT)
        fingerprint.update(
            f"{relative.as_posix()}:{label}\n".encode()
        )

    for source in source_files:
        fingerprint.update(source.read_bytes())

    manifest = {
        "model": "dinov2_vits14",
        "num_images": n,
        "fingerprint": fingerprint.hexdigest(),
    }

    manifest_path = output_dir / "manifest.json"

    if manifest_path.exists():
        with open(manifest_path) as f:
            previous = json.load(f)

        if previous != manifest:
            raise RuntimeError(
                "Dataset or preprocessing changed. "
                "Use a new output directory."
            )
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2))

    # Load frozen DINOv2
    model = get_dinov2().to(device)
    model.eval()

    for start in range(0, n, args.chunk_size):
        end = min(start + args.chunk_size, n)

        output_file = output_dir / (
            f"chunk_{start:06d}_{end:06d}.npz"
        )

        # Resume: skip complete chunks
        if output_file.exists():
            with np.load(output_file, allow_pickle=False) as saved:
                if saved["X"].shape != (end - start, 384):
                    raise RuntimeError(
                        f"Invalid checkpoint: {output_file}"
                    )
            print(f"Skipping completed chunk {start}:{end}")
            continue

        indices = range(start, end)

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
                loader, desc=f"Images {start}:{end}"
            ):
                images = images.to(
                    device,
                    non_blocking=(device.type == "cuda")
                )

                z = model(images)

                features.append(z.cpu().numpy())
                labels.append(y.numpy())

        X = np.concatenate(features, axis=0)
        y = np.concatenate(labels, axis=0)

        paths = [
            Path(dataset.samples[i][0])
            .relative_to(DATASET_ROOT)
            .as_posix()
            for i in indices
        ]

        # Atomic save: incomplete chunks are never marked done
        temporary = output_file.with_suffix(".npz.tmp")

        with open(temporary, "wb") as f:
            np.savez(
                f,
                X=X,
                y=y,
                paths=np.asarray(paths)
            )

        os.replace(temporary, output_file)

        print(f"Saved {len(X)} features to {output_file}")

    print("Feature extraction completed!")


if __name__ == "__main__":
    main()
