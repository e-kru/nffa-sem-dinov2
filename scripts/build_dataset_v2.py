
import csv
import json
from collections import Counter
from pathlib import Path


SPLITS_PATH = Path("outputs/splits.json")
GREEN_PATH = Path("outputs/audit/green_overlays.csv")

OUTPUT_DIR = Path("outputs/dataset_v2")
MANIFEST_PATH = OUTPUT_DIR / "manifest.csv"

GREEN_THRESHOLD = 15


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load the original dataset inventory
    with open(SPLITS_PATH) as f:
        splits = json.load(f)

    images = {}

    for split in ("train", "val", "test"):
        for record in splits[split]:
            path = record["path"]

            if path in images:
                raise ValueError(f"Duplicate path: {path}")

            images[path] = {
                "label": int(record["label"]),
                "original_split": split,
            }

    # 2. Load green-overlay audit
    green_counts = {}

    with open(GREEN_PATH, newline="") as f:
        for row in csv.DictReader(f):
            path = row["path"]

            if path in green_counts:
                raise ValueError(f"Duplicate audit entry: {path}")

            green_counts[path] = int(row["green_pixels"])

    # 3. Check that both inventories match
    if set(images) != set(green_counts):
        missing = set(images) - set(green_counts)
        extra = set(green_counts) - set(images)

        raise ValueError(
            f"Audit mismatch: {len(missing)} missing, "
            f"{len(extra)} extra"
        )

    # 4. Build central manifest
    fieldnames = [
        "image_id",
        "label",
        "class_name",
        "original_split",
        "green_pixels",
        "included",
        "exclusion_reason",
    ]

    counts = Counter()
    removed_per_class = Counter()
    remaining_per_class = Counter()

    with open(MANIFEST_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for path in sorted(images):
            info = images[path]
            green_pixels = green_counts[path]

            included = green_pixels < GREEN_THRESHOLD
            class_name = Path(path).parts[0]

            writer.writerow({
                "image_id": path,
                "label": info["label"],
                "class_name": class_name,
                "original_split": info["original_split"],
                "green_pixels": green_pixels,
                "included": int(included),
                "exclusion_reason": (
                    "" if included else "green_overlay"
                ),
            })

            counts["total"] += 1

            if included:
                counts["included"] += 1
                remaining_per_class[class_name] += 1
            else:
                counts["excluded"] += 1
                removed_per_class[class_name] += 1

    # 5. Save configuration for reproducibility
    config = {
        "source_inventory": str(SPLITS_PATH),
        "green_audit": str(GREEN_PATH),
        "green_threshold": GREEN_THRESHOLD,
        "crop_version": "preprocessing_v2",
        "original_splits_used_for_new_split": False,
    }

    with open(OUTPUT_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # 6. Report
    assert counts["total"] == 21169
    assert counts["included"] + counts["excluded"] == counts["total"]

    print("\n--- Dataset V2 ---")
    print(f"Original images: {counts['total']}")
    print(f"Included:        {counts['included']}")
    print(f"Excluded:        {counts['excluded']}")

    print("\nClass distribution:")
    for class_name in sorted(
        set(remaining_per_class) | set(removed_per_class)
    ):
        kept = remaining_per_class[class_name]
        removed = removed_per_class[class_name]
        total = kept + removed

        print(
            f"{class_name:32s} "
            f"kept={kept:5d} "
            f"removed={removed:5d} "
            f"({removed / total:.1%})"
        )

    print(f"\nSaved: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
