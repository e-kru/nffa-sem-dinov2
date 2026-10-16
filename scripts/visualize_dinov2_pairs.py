
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from src.preprocessing_v2 import crop_metadata_strip


DATASET_ROOT = Path("/mnt/data/stud-ueqio/datasets/nffa_sem_100")
REPORT = Path("outputs/dataset_v2/dinov2_neighbors.csv")
OUTPUT_DIR = Path("outputs/dataset_v2/dinov2_pairs")

PAIRS_PER_CATEGORY = 12
PAIRS_PER_FIGURE = 6
SEED = 42


# Categories for visual inspection
CATEGORIES = {
    "A_phash_0_4_dino_099": lambda d, s: (
        d <= 4 and s >= 0.99
    ),
    "B_phash_5_8_dino_098": lambda d, s: (
        5 <= d <= 8 and s >= 0.98
    ),
    "C_phash_above_8_dino_0995": lambda d, s: (
        d > 8 and s >= 0.995
    ),
}


def load_unique_pairs():
    """Load DINOv2 neighbors and remove reversed duplicates."""

    with open(REPORT, newline="") as f:
        rows = list(csv.DictReader(f))

    unique = {}

    for row in rows:
        a = row["image_a"]
        b = row["image_b"]

        if a == b:
            continue

        # Exact pixel duplicates are already confirmed
        if row["pixel_sha256_identical"] == "1":
            continue

        key = tuple(sorted((a, b)))

        if key not in unique:
            unique[key] = row

    return list(unique.values())


def select_examples(candidates, rng):
    """
    Select strong and random candidates
    to inspect both obvious and difficult cases.
    """

    candidates = sorted(
        candidates,
        key=lambda r: float(r["cosine_similarity"]),
        reverse=True,
    )

    # Four strongest candidates
    strongest = candidates[:4]

    # Random selection from the remaining candidates
    remaining = candidates[4:]

    n_random = min(
        PAIRS_PER_CATEGORY - len(strongest),
        len(remaining),
    )

    selected = strongest + rng.sample(
        remaining, n_random
    )

    return selected


def plot_examples(category, selected):
    """Save contact sheets comparing cropped SEM images."""

    for start in range(0, len(selected), PAIRS_PER_FIGURE):
        batch = selected[start:start + PAIRS_PER_FIGURE]

        fig, axes = plt.subplots(
            len(batch),
            2,
            figsize=(12, 3.6 * len(batch)),
            squeeze=False,
        )

        for i, row in enumerate(batch):
            similarity = float(row["cosine_similarity"])
            distance = int(row["phash_distance"])

            for j, key in enumerate(["image_a", "image_b"]):
                path = row[key]
                label = row[f"label_{'a' if j == 0 else 'b'}"]

                with Image.open(DATASET_ROOT / path) as img:
                    image = crop_metadata_strip(
                        img.convert("RGB")
                    )

                axes[i, j].imshow(image)
                axes[i, j].set_title(
                    f"{Path(path).parent.name} | label {label}\n"
                    f"Cosine={similarity:.4f} | pHash={distance}",
                    fontsize=9,
                )
                axes[i, j].axis("off")

        fig.tight_layout()

        output = OUTPUT_DIR / f"{category}_{start:03d}.png"
        fig.savefig(output, dpi=120)
        plt.close(fig)

        print(f"Saved: {output}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pairs = load_unique_pairs()

    print(f"Unique DINOv2 candidate pairs: {len(pairs)}")

    rng = random.Random(SEED)

    for category, rule in CATEGORIES.items():

        candidates = [
            row for row in pairs
            if rule(
                int(row["phash_distance"]),
                float(row["cosine_similarity"]),
            )
        ]

        print(f"\n{category}")
        print(f"Candidate pairs: {len(candidates)}")

        cross_class = sum(
            row["label_a"] != row["label_b"]
            for row in candidates
        )

        print(f"Cross-class pairs: {cross_class}")

        selected = select_examples(candidates, rng)

        if not selected:
            print("No candidates.")
            continue

        plot_examples(category, selected)


if __name__ == "__main__":
    main()
