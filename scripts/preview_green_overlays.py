
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from src.preprocessing_v2 import crop_metadata_strip


ROOT = Path("/mnt/data/stud-ueqio/datasets/nffa_sem_100")
REPORT = Path("outputs/audit/green_overlays.csv")
OUTPUT = Path("outputs/audit/green_overlay_examples.png")

with open(REPORT) as f:
    rows = list(csv.DictReader(f))

positive = [
    row for row in rows
    if int(row["green_pixels"]) >= 15
]

# 12 images with the most green pixels
most_green = sorted(
    positive,
    key=lambda row: int(row["green_pixels"]),
    reverse=True
)[:12]

# 12 random flagged images
rng = random.Random(42)
remaining = [
    row for row in positive
    if row not in most_green
]
random_examples = rng.sample(remaining, 12)

selected = most_green + random_examples

fig, axes = plt.subplots(6, 4, figsize=(16, 20))

for ax, row in zip(axes.flat, selected):
    with Image.open(ROOT / row["path"]) as img:
        image = crop_metadata_strip(img.convert("RGB"))

    ax.imshow(image)
    ax.set_title(
        f"{row['path'].split('/')[0]}\n"
        f"Green pixels: {row['green_pixels']}",
        fontsize=9
    )
    ax.axis("off")

fig.tight_layout()
fig.savefig(OUTPUT, dpi=120)
plt.close(fig)

print(f"Saved: {OUTPUT}")
