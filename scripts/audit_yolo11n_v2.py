"""Reconstructed YOLO11n inference over V2 accepted SEM images.

Workflow: outputs/dataset_v2/manifest.csv (18,614 accepted images)
  -> load raw image from DATASET_ROOT -> RGB -> crop_metadata_strip (V2)
  -> fine-tuned YOLO11n best.pt inference -> detection-level CSV and per-image CSV.

YOLO predictions do NOT mask, crop, or modify images. Final V3 manifest
excludes original V2-accepted images with measurement_box confidence >= 0.50;
then excludes remaining images with metadata_strip confidence >= 0.50.

This is a reconstruction, not a recovered original inference script.
It MUST NOT overwrite outputs/dataset_v2/box_detector/audit_v2/.
Different Ultralytics/PyTorch versions may yield slight prediction differences.
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

from src.config import DATASET_ROOT
from src.preprocessing_v2 import crop_metadata_strip

V2 = Path("outputs/dataset_v2")
WEIGHTS = V2 / "box_detector/runs/v2/weights/best.pt"
MANIFEST = V2 / "manifest.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu", help="cpu or CUDA index, e.g. 0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.05,
                        help="Detection audit threshold; final exclusion uses 0.50")
    parser.add_argument("--limit", type=int, default=None,
                        help="Smoke-test on a subset; not a complete audit")
    args = parser.parse_args()

    if not WEIGHTS.is_file():
        raise FileNotFoundError(WEIGHTS)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")

    with MANIFEST.open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["included"] == "1"]
    if args.limit is None:
        assert len(rows) == 18614, len(rows)
    else:
        rows = rows[:args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(WEIGHTS))
    detections_csv = args.output_dir / "detections.csv"
    summary_csv = args.output_dir / "image_summary.csv"
    counts = Counter()

    with detections_csv.open("w", newline="") as df, summary_csv.open("w", newline="") as sf:
        detections = csv.DictWriter(df, fieldnames=[
            "image_id", "class_name", "confidence", "x1", "y1", "x2", "y2"
        ])
        summaries = csv.DictWriter(sf, fieldnames=[
            "image_id", "num_detections", "measurement_box_max_confidence",
            "metadata_strip_max_confidence"
        ])
        detections.writeheader()
        summaries.writeheader()

        for index, row in enumerate(rows, start=1):
            image_id = row["image_id"]
            path = Path(DATASET_ROOT) / image_id
            with Image.open(path) as im:
                cropped = crop_metadata_strip(im.convert("RGB"))

            result = model.predict(
                source=cropped, imgsz=args.imgsz, conf=args.conf,
                device=args.device, verbose=False
            )[0]
            maximum = {"measurement_box": 0.0, "metadata_strip": 0.0}
            count = 0

            for box in result.boxes:
                class_name = str(result.names[int(box.cls.item())])
                conf = float(box.conf.item())
                x1, y1, x2, y2 = (float(x) for x in box.xyxy[0].tolist())
                detections.writerow({
                    "image_id": image_id,
                    "class_name": class_name,
                    "confidence": conf,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                })
                maximum[class_name] = max(maximum.get(class_name, 0.0), conf)
                count += 1

            summaries.writerow({
                "image_id": image_id,
                "num_detections": count,
                "measurement_box_max_confidence": maximum["measurement_box"],
                "metadata_strip_max_confidence": maximum["metadata_strip"],
            })
            if maximum["measurement_box"] >= 0.50:
                counts["measurement_box"] += 1
            elif maximum["metadata_strip"] >= 0.50:
                counts["metadata_strip"] += 1
            else:
                counts["retained"] += 1

            if index % 500 == 0:
                print(f"Processed {index}/{len(rows)} images", flush=True)

    print("Saved:", detections_csv, summary_csv)
    print("Exclusion counts at 0.50:", dict(counts))
    if args.limit is None:
        print("Original audit expected: 210 box, 118 strip, 18,286 retained")
        if (counts["measurement_box"], counts["metadata_strip"], counts["retained"]) != (210, 118, 18286):
            print("WARNING: Reconstructed audit differs; inspect versions and model inference settings.")


if __name__ == "__main__":
    main()
