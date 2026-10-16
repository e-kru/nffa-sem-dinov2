
import csv
import json
from collections import defaultdict
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.preprocessing import crop_metadata_strip


DATASET_ROOT = Path("/mnt/data/stud-ueqio/datasets/nffa_sem_100")
SPLIT_FILE = Path("outputs/splits.json")
OUTPUT_DIR = Path("outputs/audit")
CACHE_FILE = OUTPUT_DIR / "phash_cache.npz"


def compute_phash(path):
    """Compute a 64-bit perceptual hash after metadata cropping."""

    with Image.open(DATASET_ROOT / path) as image:
        image = image.convert("RGB")
        image = crop_metadata_strip(image)

        h = imagehash.phash(image, hash_size=8)

    return int(str(h), 16)


def load_or_compute_hashes(paths):
    """Compute hashes once and reuse them on later runs."""

    if CACHE_FILE.exists():
        with np.load(CACHE_FILE, allow_pickle=False) as data:
            cached_paths = data["paths"].tolist()

            if cached_paths != paths:
                raise ValueError("Cached paths do not match dataset.")

            return data["hashes"]

    hashes = []

    for path in tqdm(paths, desc="Computing pHashes"):
        hashes.append(compute_phash(path))

    hashes = np.array(hashes, dtype=np.uint64)

    np.savez_compressed(
        CACHE_FILE,
        paths=np.array(paths),
        hashes=hashes,
    )

    return hashes


def load_known_duplicates():
    """Find validation images already duplicated in training."""

    groups = defaultdict(list)

    with open(OUTPUT_DIR / "sha256_duplicates.csv") as f:
        for row in csv.DictReader(f):
            groups[row["sha256"]].append(
                (row["split"], row["path"])
            )

    known = set()

    for images in groups.values():
        if any(split == "train" for split, _ in images):
            known.update(
                path for split, path in images
                if split == "val"
            )

    return known


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(SPLIT_FILE) as f:
        splits = json.load(f)

    paths = [
        record["path"]
        for name in ["train", "val", "test"]
        for record in splits[name]
    ]

    hashes = load_or_compute_hashes(paths)
    path_to_hash = dict(zip(paths, hashes))

    train_records = splits["train"]
    val_records = splits["val"]

    train_hashes = np.array(
        [path_to_hash[r["path"]] for r in train_records],
        dtype=np.uint64,
    )

    known_duplicates = load_known_duplicates()
    results = []

    for record in tqdm(val_records, desc="Finding nearest neighbors"):
        val_path = record["path"]
        val_hash = path_to_hash[val_path]

        # Hamming distance to every training image
        distances = np.bitwise_count(
            np.bitwise_xor(train_hashes, val_hash)
        )

        nearest_idx = int(np.argmin(distances))
        nearest_train = train_records[nearest_idx]

        results.append({
            "val_path": val_path,
            "val_label": record["label"],
            "train_path": nearest_train["path"],
            "train_label": nearest_train["label"],
            "distance": int(distances[nearest_idx]),
            "known_sha_duplicate": val_path in known_duplicates,
        })

    results.sort(key=lambda r: r["distance"])

    output_file = OUTPUT_DIR / "phash_nearest_neighbors.csv"

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\n--- pHash Audit ---")
    print(f"Validation images: {len(results)}")
    print(f"Known SHA duplicates: {len(known_duplicates)}")

    for threshold in [0, 4, 8]:
        candidates = [
            r for r in results
            if r["distance"] <= threshold
        ]

        new_candidates = [
            r for r in candidates
            if not r["known_sha_duplicate"]
        ]

        print(f"\nHamming distance <= {threshold}:")
        print(f"  All validation images: {len(candidates)}")
        print(f"  Excluding known SHA duplicates: {len(new_candidates)}")

    print(f"\nReport: {output_file}")


if __name__ == "__main__":
    main()
