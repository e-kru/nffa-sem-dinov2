
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm


DATASET_ROOT = Path("/mnt/data/stud-ueqio/datasets/nffa_sem_100")
SPLIT_FILE = Path("outputs/splits.json")
OUTPUT_DIR = Path("outputs/audit")


def hash_file(path):
    """Calculate the SHA-256 hash of an image file."""
    hasher = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def audit_duplicates():
    with open(SPLIT_FILE) as f:
        splits = json.load(f)

    # Collect all image paths and their split assignments
    records = []

    for split_name in ["train", "val", "test"]:
        for record in splits[split_name]:
            records.append((split_name, record["path"]))

    # Group images with identical file contents
    hash_to_images = defaultdict(list)

    for split_name, relative_path in tqdm(records, desc="Hashing images"):
        image_path = DATASET_ROOT / relative_path

        image_hash = hash_file(image_path)
        hash_to_images[image_hash].append(
            (split_name, relative_path)
        )

    # Find groups of identical images
    duplicates = {
        h: images
        for h, images in hash_to_images.items()
        if len(images) > 1
    }

    cross_split = {
        h: images
        for h, images in duplicates.items()
        if len({split for split, _ in images}) > 1
    }

    # Save report
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_DIR / "sha256_duplicates.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sha256", "cross_split", "split", "path"])

        for h, images in duplicates.items():
            is_cross_split = h in cross_split

            for split, path in images:
                writer.writerow([h, is_cross_split, split, path])

    print("\n--- Duplicate Audit ---")
    print(f"Images checked: {len(records)}")
    print(f"Duplicate groups: {len(duplicates)}")
    print(f"Cross-split duplicate groups: {len(cross_split)}")
    print(f"Report: {OUTPUT_DIR / 'sha256_duplicates.csv'}")


if __name__ == "__main__":
    audit_duplicates()
