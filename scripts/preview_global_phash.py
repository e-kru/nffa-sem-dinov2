
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from src.preprocessing_v2 import crop_metadata_strip


ROOT = Path("/mnt/data/stud-ueqio/datasets/nffa_sem_100")
INPUT = Path("outputs/dataset_v2/phash_candidate_pairs.csv")
OUTPUT = Path("outputs/dataset_v2/phash_examples")

DISTANCES = [0, 2, 4, 6, 8]
PAIRS_PER_DISTANCE = 6


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    with open(INPUT, newline="") as f:
        pairs = list(csv.DictReader(f))

    rng = random.Random(42)

    for distance in DISTANCES:
        candidates = [
            row for row in pairs
            if int(row["phash_distance"]) == distance
            and row["same_pixels"] == "0"
        ]

        selected = rng.sample(
            candidates,
            min(PAIRS_PER_DISTANCE, len(candidates))
        )

        if not selected:
            continue

        fig, axes = plt.subplots(
            len(selected), 2,
            figsize=(12, 3.5 * len(selected)),
            squeeze=False
        )

        for i, pair in enumerate(selected):
            for j, key in enumerate(["image_a", "image_b"]):
                path = pair[key]

                with Image.open(ROOT / path) as img:
                    image = crop_metadata_strip(
                        img.convert("RGB")
                    )

                axes[i, j].imshow(image)
                axes[i, j].set_title(
                    f"{Path(path).parent.name}\n"
                    f"{Path(path).stem[:24]}",
                    fontsize=9
                )
                axes[i, j].axis("off")

        fig.suptitle(
            f"pHash distance = {distance}",
            fontsize=15
        )
        fig.tight_layout()

        path = OUTPUT / f"distance_{distance}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)

        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
