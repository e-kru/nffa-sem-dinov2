
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


ROOT = Path("outputs/dataset_v2")
FEATURE_DIR = ROOT / "features"
MANIFEST = ROOT / "manifest.csv"
HASH_FILE = ROOT / "image_hashes.csv"
OUTPUT = ROOT / "dinov2_neighbors.csv"


def load_features():
    """Load all Dataset V2 feature chunks."""

    files = sorted(FEATURE_DIR.glob("chunk_*.npz"))

    if not files:
        raise FileNotFoundError("No feature chunks found")

    features = []
    labels = []
    paths = []

    for file in files:
        with np.load(file, allow_pickle=False) as data:
            features.append(data["X"])
            labels.append(data["y"])
            paths.append(data["paths"])

    X = np.concatenate(features).astype(np.float32)
    y = np.concatenate(labels)
    paths = np.concatenate(paths).astype(str)

    if X.shape[1] != 384:
        raise ValueError(f"Unexpected feature shape: {X.shape}")

    if not np.isfinite(X).all():
        raise ValueError("Features contain NaN or infinity")

    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate image paths in features")

    return X, y, paths


def load_manifest():
    """Load included image paths and labels."""

    with open(MANIFEST, newline="") as f:
        rows = list(csv.DictReader(f))

    included = {
        row["image_id"]: int(row["label"])
        for row in rows
        if row["included"] == "1"
    }

    return included


def load_hashes():
    """Load pHash and exact pixel hash for each image."""

    with open(HASH_FILE, newline="") as f:
        rows = list(csv.DictReader(f))

    hashes = {}

    for row in rows:
        path = row["image_id"]

        if path in hashes:
            raise ValueError(f"Duplicate hash entry: {path}")

        hashes[path] = {
            "phash": int(row["phash"], 16),
            "pixel_sha256": row["pixel_sha256"],
        }

    return hashes


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=10)

    args = parser.parse_args()

    if args.batch_size <= 0 or args.top_k <= 0:
        raise ValueError("Batch size and top-k must be positive")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    device = torch.device(args.device)

    # --------------------------------------------------
    # 1. Load all features and metadata
    # --------------------------------------------------

    X, y, paths = load_features()
    included = load_manifest()
    hashes = load_hashes()

    n = len(paths)

    if set(paths) != set(included):
        raise ValueError("Features do not match Dataset V2 manifest")

    if set(paths) != set(hashes):
        raise ValueError("Hashes do not match Dataset V2 features")

    for path, label in zip(paths, y):
        if int(label) != included[path]:
            raise ValueError(f"Label mismatch: {path}")

    if n <= 1:
        raise ValueError("Need at least two images")

    k = min(args.top_k, n - 1)

    print(f"Images: {n}")
    print(f"Feature matrix: {X.shape}")
    print(f"Device: {device}")
    print(f"Top-k: {k}")

    # --------------------------------------------------
    # 2. Normalize DINOv2 features
    # --------------------------------------------------

    Z = F.normalize(
        torch.from_numpy(X).to(device),
        p=2,
        dim=1,
    )

    top1_scores = []
    cross_class_pairs = 0
    total_pairs = 0

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # 3. Global nearest-neighbor search
    # --------------------------------------------------

    with open(OUTPUT, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "image_a",
            "image_b",
            "label_a",
            "label_b",
            "rank",
            "cosine_similarity",
            "phash_distance",
            "pixel_sha256_identical",
        ])

        with torch.inference_mode():

            for start in tqdm(
                range(0, n, args.batch_size),
                desc="DINOv2 neighbors",
            ):
                end = min(start + args.batch_size, n)

                # Similarity to every image in the dataset
                similarities = Z[start:end] @ Z.T

                # Exclude self-comparisons
                local_indices = torch.arange(
                    end - start, device=device
                )
                global_indices = torch.arange(
                    start, end, device=device
                )

                similarities[
                    local_indices,
                    global_indices,
                ] = -float("inf")

                scores, indices = torch.topk(
                    similarities,
                    k=k,
                    dim=1,
                )

                scores = scores.cpu().numpy()
                indices = indices.cpu().numpy()

                # Save the neighbors for each image
                for local_i in range(end - start):
                    i = start + local_i

                    path_a = str(paths[i])
                    label_a = int(y[i])

                    top1_scores.append(float(scores[local_i, 0]))

                    for rank in range(k):
                        j = int(indices[local_i, rank])

                        path_b = str(paths[j])
                        label_b = int(y[j])

                        similarity = float(scores[local_i, rank])

                        phash_distance = (
                            hashes[path_a]["phash"]
                            ^ hashes[path_b]["phash"]
                        ).bit_count()

                        same_pixels = (
                            hashes[path_a]["pixel_sha256"]
                            == hashes[path_b]["pixel_sha256"]
                        )

                        if label_a != label_b:
                            cross_class_pairs += 1

                        total_pairs += 1

                        writer.writerow([
                            path_a,
                            path_b,
                            label_a,
                            label_b,
                            rank + 1,
                            similarity,
                            phash_distance,
                            int(same_pixels),
                        ])

    # --------------------------------------------------
    # 4. Summary
    # --------------------------------------------------

    similarities = np.asarray(top1_scores)

    print("\n--- Global DINOv2 Audit ---")
    print(f"Images: {n}")
    print(f"Neighbor records: {total_pairs}")
    print(f"Cross-class neighbor records: {cross_class_pairs}")
    print(f"Median top-1 similarity: {np.median(similarities):.4f}")

    for threshold in [0.95, 0.98, 0.99, 0.995]:
        count = int((similarities >= threshold).sum())

        print(
            f"Top-1 similarity >= {threshold}: "
            f"{count} images"
        )

    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
