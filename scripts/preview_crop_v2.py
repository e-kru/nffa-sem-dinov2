
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from src.preprocessing import crop_metadata_strip as crop_v1
from src.preprocessing_v2 import (
    crop_metadata_strip as crop_v2,
    detect_metadata_start,
)


DATASET_ROOT = Path("/mnt/data/stud-ueqio/datasets/nffa_sem_100")
OUTPUT = Path("outputs/audit/crop_v2_comparison.png")

with open("outputs/splits.json") as f:
    splits = json.load(f)

paths = [
    record["path"]
    for split in ["train", "val", "test"]
    for record in splits[split]
]

# Known problematic images from our previous audit
problematic_ids = [
    "L2_276ece5f845de7cf395bce8a1a0995cf",
    "L2_3379b8ba1d4ee1a37f47f73238a0a4d3",
    "L6_2eadd4c15e506c8d6ef614f0d05df812",
]

selected = [
    path for path in paths
    if Path(path).stem in problematic_ids
]

# Add random images to check for false positives
rng = random.Random(42)
selected += rng.sample(
    [p for p in paths if p not in selected],
    8
)

fig, axes = plt.subplots(
    len(selected), 3,
    figsize=(14, 3 * len(selected)),
    squeeze=False
)

for i, path in enumerate(selected):
    with Image.open(DATASET_ROOT / path) as img:
        original = img.convert("RGB")

    old = crop_v1(original)
    new = crop_v2(original)
    crop_y = detect_metadata_start(original)

    for j, (title, image) in enumerate([
        ("Original", original),
        ("Old crop", old),
        ("New crop V2", new),
    ]):
        axes[i, j].imshow(image)
        axes[i, j].set_title(title)
        axes[i, j].axis("off")

    if crop_y is not None:
        axes[i, 0].axhline(
            crop_y, color="red", linewidth=2
        )

    print(
        f"{Path(path).name[:30]} | "
        f"V1 removed: {original.height - old.height}px | "
        f"V2 removed: {original.height - new.height}px"
    )

fig.tight_layout()
fig.savefig(OUTPUT, dpi=130)
plt.close(fig)

print(f"\nSaved: {OUTPUT}")
