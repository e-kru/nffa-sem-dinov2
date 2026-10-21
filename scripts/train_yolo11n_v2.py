"""Reconstructed YOLO11n training entry point for SEM annotation filtering.

Workflow: existing yolo_v2/images/{train,val} and labels/{train,val}
  -> ultralytics YOLO11n fine-tuning -> runs/v2_reproduction/weights/best.pt.

Input training images/labels are a PREPARED dataset; this script does not
reconstruct how those image files were selected/annotated or prepare them.
No `src.preprocessing_v2` cropping occurs at training time here: the files
in yolo_v2/images are used as stored. Whether/how they were preprocessed
before creation of that folder needs separate provenance verification.
YOLO is later used only to DETECT and EXCLUDE images (no masking or ROI crop).

Training arguments are reconstructed from the saved original
`outputs/dataset_v2/box_detector/runs/v2/args.yaml`; package versions and
other original defaults may differ. This is NOT the original training script.
The original weights and experimental results are not overwritten.
"""

from pathlib import Path

from ultralytics import YOLO

BASE = Path("outputs/dataset_v2/box_detector")
DATA = BASE / "yolo_v2/data.yaml"
PROJECT = BASE / "runs"
NAME = "v2_reproduction"


def main():
    if not DATA.is_file():
        raise FileNotFoundError(DATA)
    for split in ("train", "val"):
        imgdir = BASE / f"yolo_v2/images/{split}"
        labeldir = BASE / f"yolo_v2/labels/{split}"
        if not imgdir.is_dir() or not labeldir.is_dir():
            raise FileNotFoundError(f"Missing {split} data: {imgdir}, {labeldir}")
    if (PROJECT / NAME).exists():
        raise FileExistsError(f"Refusing to overwrite: {PROJECT / NAME}")

    model = YOLO("yolo11n.pt")
    model.train(
        data=str(DATA.resolve()),
        epochs=60,
        batch=16,
        imgsz=640,
        device=0,
        optimizer="auto",
        seed=42,
        lr0=0.01,
        project=str(PROJECT.resolve()),
        name=NAME,
        exist_ok=False,
    )


if __name__ == "__main__":
    main()
