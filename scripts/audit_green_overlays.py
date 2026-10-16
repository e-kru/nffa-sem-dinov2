
import argparse
import csv
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.preprocessing_v2 import crop_metadata_strip


DATASET_ROOT = Path("/mnt/data/stud-ueqio/datasets/nffa_sem_100")
OUTPUT = Path("outputs/audit/green_overlays.csv")

MIN_GREEN_PIXELS = 15


def count_green_pixels(image):
    """Count clearly green pixels in an RGB image."""

    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)

    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    green_mask = (
        (g > r + 30)
        & (g > b + 20)
        & (g > 60)
    )

    return int(green_mask.sum())


def process_image(task):
    """Process one image. Called independently by CPU workers."""

    split, path = task

    with Image.open(DATASET_ROOT / path) as img:
        image = img.convert("RGB")

    image = crop_metadata_strip(image)

    green_pixels = count_green_pixels(image)

    return {
        "split": split,
        "path": path,
        "green_pixels": green_pixels,
        "has_green": green_pixels >= MIN_GREEN_PIXELS,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    with open("outputs/splits.json") as f:
        splits = json.load(f)

    tasks = [
        (split, record["path"])
        for split in ["train", "val", "test"]
        for record in splits[split]
    ]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    counts = Counter()
    totals = Counter()

    # Process images in parallel
    with ProcessPoolExecutor(max_workers=args.workers) as executor:

        results = executor.map(
            process_image,
            tasks,
            chunksize=16,
        )

        with open(OUTPUT, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "split",
                    "path",
                    "green_pixels",
                    "has_green",
                ],
            )
            writer.writeheader()

            for result in tqdm(results, total=len(tasks)):
                writer.writerow(result)

                split = result["split"]
                totals[split] += 1

                if result["has_green"]:
                    counts[split] += 1

    print("\n--- Green Overlay Audit ---")

    for split in ["train", "val", "test"]:
        print(
            f"{split}: {counts[split]}/{totals[split]} "
            f"({counts[split] / totals[split]:.2%})"
        )

    total_green = sum(counts.values())
    total_images = sum(totals.values())

    print(f"\nTotal images: {total_images}")
    print(f"Images with green pixels: {total_green}")
    print(f"Percentage: {total_green / total_images:.2%}")
    print(f"Report: {OUTPUT}")


if __name__ == "__main__":
    main()
