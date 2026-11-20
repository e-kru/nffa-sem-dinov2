"""
Reconstructed V3 manifest builder.

Input:
  outputs/dataset_v2/manifest.csv
  outputs/dataset_v2/box_detector/audit_v2/detections.csv

YOLO detections were generated after preprocessing_v2.
YOLO is used only to exclude images, not to mask or crop regions.
All retained images use Crop V2, without additional cropping.

This script reconstructs the final filtering rules and should
be validated against the original V3 manifest.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

V2 = Path("outputs/dataset_v2")
THRESHOLD = 0.50

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", type=Path, required=True)
args = parser.parse_args()

out = args.output_dir
out.mkdir(parents=True, exist_ok=True)

manifest_out = out / "manifest.csv"
config_out = out / "config.json"

for path in (manifest_out, config_out):
    if path.exists():
        raise FileExistsError(path)

# 1. Read YOLO detections after Crop V2
boxes = set()
strips = set()

with open(
    V2 / "box_detector/audit_v2/detections.csv",
    newline=""
) as f:
    for row in csv.DictReader(f):
        if float(row["confidence"]) < THRESHOLD:
            continue

        image_id = row["image_id"]

        if row["class_name"] == "measurement_box":
            boxes.add(image_id)
        elif row["class_name"] == "metadata_strip":
            strips.add(image_id)

# 2. Read V2 manifest
with open(V2 / "manifest.csv", newline="") as f:
    reader = csv.DictReader(f)
    fields = list(reader.fieldnames)
    rows = list(reader)

if "crop_mode" not in fields:
    fields.append("crop_mode")

counts = Counter()

# 3. Apply final exclusion rules
for row in rows:
    image_id = row["image_id"]

    if row["included"] != "1":
        row["crop_mode"] = "excluded"
        counts["green_overlay"] += 1

    elif image_id in boxes:
        row["included"] = "0"
        row["exclusion_reason"] = "measurement_box_yolo_v2"
        row["crop_mode"] = "excluded"
        counts["measurement_box"] += 1

    elif image_id in strips:
        row["included"] = "0"
        row["exclusion_reason"] = "metadata_strip_yolo_v2"
        row["crop_mode"] = "excluded"
        counts["metadata_strip"] += 1

    else:
        row["crop_mode"] = "crop_v2"
        counts["included"] += 1

# 4. Validate counts
assert len(rows) == 21169
assert counts["green_overlay"] == 2555, counts
assert counts["measurement_box"] == 210, counts
assert counts["metadata_strip"] == 118, counts
assert counts["included"] == 18286, counts

# 5. Save final manifest
with open(manifest_out, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

# 6. Save reconstruction configuration
config = {
    "source_manifest": str(V2 / "manifest.csv"),
    "source_detections": str(
        V2 / "box_detector/audit_v2/detections.csv"
    ),
    "yolo_weights": str(
        V2 / "box_detector/runs/v2/weights/best.pt"
    ),
    "normal_crop": "src.preprocessing_v2.crop_metadata_strip",
    "measurement_box_threshold": THRESHOLD,
    "metadata_strip_threshold": THRESHOLD,
    "measurement_box_action": "exclude",
    "metadata_strip_action": "exclude",
    "extra_crop": False,
}

config_out.write_text(json.dumps(config, indent=2))

print("\n--- DATASET V3 ---")
print("Excluded green:", counts["green_overlay"])
print("Excluded boxes:", counts["measurement_box"])
print("Excluded strips:", counts["metadata_strip"])
print("Included:", counts["included"])
print("Saved:", manifest_out)
print("Saved:", config_out)
