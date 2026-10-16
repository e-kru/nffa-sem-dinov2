
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from scripts.train_classifier import load_features, get_splits


OUTPUT_DIR = Path("outputs/audit")
SPLIT_FILE = Path("outputs/splits.json")
PHASH_CACHE = OUTPUT_DIR / "phash_cache.npz"
SHA_REPORT = OUTPUT_DIR / "sha256_duplicates.csv"


def load_phashes():
    """Load previously computed perceptual hashes."""
    if not PHASH_CACHE.exists():
        return {}

    with np.load(PHASH_CACHE, allow_pickle=False) as data:
        return {
            str(path): int(h)
            for path, h in zip(data["paths"], data["hashes"])
        }


def load_sha_groups():
    """Map paths of exact duplicate files to their SHA-256."""
    result = {}

    if SHA_REPORT.exists():
        with open(SHA_REPORT) as f:
            for row in csv.DictReader(f):
                result[row["path"]] = row["sha256"]

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(SPLIT_FILE) as f:
        split_info = json.load(f)

    # Load already extracted DINOv2 features
    X, y, paths = load_features()
    splits = get_splits(X, y, paths)

    X_train, _ = splits["train"]
    X_val, _ = splits["val"]

    train_records = split_info["train"]
    val_records = split_info["val"]

    phashes = load_phashes()
    sha_groups = load_sha_groups()

    device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    # Convert NumPy arrays to tensors and normalize vectors
    train = F.normalize(
        torch.from_numpy(X_train).to(device),
        dim=1
    )
    val = F.normalize(
        torch.from_numpy(X_val).to(device),
        dim=1
    )

    results = []

    # Compute similarities in batches
    with torch.inference_mode():
        for start in range(0, len(val), args.batch_size):
            end = min(start + args.batch_size, len(val))

            similarities = val[start:end] @ train.T

            scores, indices = torch.topk(
                similarities,
                k=args.top_k,
                dim=1
            )

            scores = scores.cpu().numpy()
            indices = indices.cpu().numpy()

            for i in range(end - start):
                val_record = val_records[start + i]
                val_path = val_record["path"]

                for rank in range(args.top_k):
                    train_idx = int(indices[i, rank])
                    train_record = train_records[train_idx]
                    train_path = train_record["path"]

                    # Compare pHashes for this exact image pair
                    phash_distance = ""

                    if val_path in phashes and train_path in phashes:
                        phash_distance = (
                            phashes[val_path] ^ phashes[train_path]
                        ).bit_count()

                    # Check whether the files are exactly identical
                    sha_identical = (
                        val_path in sha_groups
                        and train_path in sha_groups
                        and sha_groups[val_path] == sha_groups[train_path]
                    )

                    results.append({
                        "val_path": val_path,
                        "train_path": train_path,
                        "val_label": val_record["label"],
                        "train_label": train_record["label"],
                        "rank": rank + 1,
                        "cosine_similarity": float(scores[i, rank]),
                        "phash_distance": phash_distance,
                        "sha256_identical": sha_identical,
                    })

            print(f"Processed {end}/{len(val)} validation images")

    # Sort by cosine similarity, highest first
    results.sort(
        key=lambda r: r["cosine_similarity"],
        reverse=True
    )

    output = OUTPUT_DIR / "dinov2_neighbors.csv"

    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=results[0].keys()
        )
        writer.writeheader()
        writer.writerows(results)

    # Summarize nearest-neighbor similarity
    top1 = [
        row for row in results
        if row["rank"] == 1
    ]

    similarities = np.array([
        row["cosine_similarity"] for row in top1
    ])

    print("\n--- DINOv2 Nearest-Neighbor Audit ---")
    print(f"Validation images: {len(top1)}")
    print(f"Total neighbor pairs: {len(results)}")
    print(f"Median top-1 similarity: {np.median(similarities):.4f}")

    for threshold in [0.95, 0.98, 0.99, 0.995]:
        count = int((similarities >= threshold).sum())
        print(f"Cosine similarity >= {threshold}: {count}")

    print(f"\nReport: {output}")


if __name__ == "__main__":
    main()
