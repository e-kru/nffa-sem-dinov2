
import csv
import hashlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import imagehash
from PIL import Image
from tqdm import tqdm

from src.preprocessing_v2 import crop_metadata_strip


ROOT = Path("/mnt/data/stud-ueqio/datasets/nffa_sem_100")
MANIFEST = Path("outputs/dataset_v2/manifest.csv")
OUTPUT = Path("outputs/dataset_v2/image_hashes.csv")


def process_image(path):
    with Image.open(ROOT / path) as img:
        image = crop_metadata_strip(img.convert("RGB"))

    # Exact hash of cropped RGB pixels and image dimensions
    hasher = hashlib.sha256()
    hasher.update(image.width.to_bytes(4, "big"))
    hasher.update(image.height.to_bytes(4, "big"))
    hasher.update(image.tobytes())

    return {
        "image_id": path,
        "pixel_sha256": hasher.hexdigest(),
        "phash": str(imagehash.phash(image)),
    }


def main():
    with open(MANIFEST, newline="") as f:
        rows = list(csv.DictReader(f))

    paths = [
        row["image_id"]
        for row in rows
        if row["included"] == "1"
    ]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    pixel_counts = Counter()
    phash_counts = Counter()

    with ProcessPoolExecutor(max_workers=8) as executor:
        results = executor.map(
            process_image,
            paths,
            chunksize=16,
        )

        with open(OUTPUT, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "image_id",
                    "pixel_sha256",
                    "phash",
                ],
            )
            writer.writeheader()

            for result in tqdm(results, total=len(paths)):
                writer.writerow(result)
                pixel_counts[result["pixel_sha256"]] += 1
                phash_counts[result["phash"]] += 1

    exact_groups = sum(n > 1 for n in pixel_counts.values())
    phash_groups = sum(n > 1 for n in phash_counts.values())

    print("\n--- Dataset V2 Hash Audit ---")
    print(f"Images: {len(paths)}")
    print(f"Exact pixel-duplicate groups: {exact_groups}")
    print(f"Groups with identical pHash: {phash_groups}")
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
