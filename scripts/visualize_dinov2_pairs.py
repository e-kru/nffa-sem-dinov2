
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from src.preprocessing import crop_metadata_strip


DATASET_ROOT = Path("/mnt/data/stud-ueqio/datasets/nffa_sem_100")
REPORT = Path("outputs/audit/dinov2_neighbors.csv")
OUTPUT_DIR = Path("outputs/audit/dinov2_pairs")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Find high-similarity pairs missed by pHash
with open(REPORT) as f:
    rows = list(csv.DictReader(f))

candidates = [
    row for row in rows
    if float(row["cosine_similarity"]) >= 0.95
    and row["phash_distance"] != ""
    and int(row["phash_distance"]) > 8
    and row["sha256_identical"].lower() == "false"
]

# Highest DINOv2 similarity first
candidates.sort(
    key=lambda row: float(row["cosine_similarity"]),
    reverse=True
)

# Keep only one candidate per validation image
selected = []
seen_val = set()

for row in candidates:
    if row["val_path"] in seen_val:
        continue

    selected.append(row)
    seen_val.add(row["val_path"])

print(f"New candidates: {len(selected)}")

# Visualize the 40 highest-similarity candidates
selected = selected[:40]

for start in range(0, len(selected), 8):
    batch = selected[start:start + 8]

    fig, axes = plt.subplots(
        len(batch), 2,
        figsize=(12, 3 * len(batch)),
        squeeze=False
    )

    for i, row in enumerate(batch):
        similarity = float(row["cosine_similarity"])
        distance = row["phash_distance"]

        for j, split in enumerate(["val", "train"]):
            path = row[f"{split}_path"]

            with Image.open(DATASET_ROOT / path) as img:
                image = crop_metadata_strip(
                    img.convert("RGB")
                )

            axes[i, j].imshow(image)
            axes[i, j].set_title(
                f"{split.upper()} | class {row[f'{split}_label']}\n"
                f"Cosine: {similarity:.4f} | pHash: {distance}",
                fontsize=9
            )
            axes[i, j].axis("off")

    fig.tight_layout()

    output = OUTPUT_DIR / f"pairs_{start:03d}.png"
    fig.savefig(output, dpi=120)
    plt.close(fig)

    print(f"Saved: {output}")
