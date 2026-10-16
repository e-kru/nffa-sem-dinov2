
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from src.preprocessing import crop_metadata_strip


DATASET_ROOT = Path("/mnt/data/stud-ueqio/datasets/nffa_sem_100")
REPORT = Path("outputs/audit/phash_nearest_neighbors.csv")
OUTPUT_DIR = Path("outputs/audit/phash_pairs")


def visualize_group(rows, group_name):
    """Create contact sheets containing 8 image pairs each."""

    group_dir = OUTPUT_DIR / group_name
    group_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{group_name}: {len(rows)} pairs")

    for start in range(0, len(rows), 8):
        batch = rows[start:start + 8]

        fig, axes = plt.subplots(
            len(batch),
            2,
            figsize=(12, 3 * len(batch)),
            squeeze=False,
        )

        for i, row in enumerate(batch):
            distance = int(row["distance"])

            for j, split in enumerate(["val", "train"]):
                path = row[f"{split}_path"]

                with Image.open(DATASET_ROOT / path) as img:
                    image = crop_metadata_strip(
                        img.convert("RGB")
                    )

                axes[i, j].imshow(image)
                axes[i, j].set_title(
                    f"{split.upper()} | class {row[f'{split}_label']}"
                    f" | pHash distance: {distance}\n"
                    f"{Path(path).name[:35]}",
                    fontsize=9,
                )
                axes[i, j].axis("off")

        fig.tight_layout()

        output = group_dir / f"pairs_{start:03d}.png"
        fig.savefig(output, dpi=120)
        plt.close(fig)

        print(f"Saved: {output}")


def main():
    with open(REPORT) as f:
        rows = list(csv.DictReader(f))

    # Exclude known exact SHA-256 duplicates
    rows = [
        row for row in rows
        if row["known_sha_duplicate"].lower() == "false"
    ]

    # Group A: distance 1–4
    group_a = [
        row for row in rows
        if 1 <= int(row["distance"]) <= 4
    ]

    # Group B: distance 5–8
    group_b = [
        row for row in rows
        if 5 <= int(row["distance"]) <= 8
    ]

    visualize_group(group_a, "distance_1_4")
    visualize_group(group_b, "distance_5_8")


if __name__ == "__main__":
    main()
