"""Reconstructed test evaluation for Split A; original test command not recovered.

Inputs: outputs/dataset_v3/splits/split_A_exact.csv,
  outputs/dataset_v2/features/*.npz (precomputed with preprocessing_v2),
  outputs/dataset_v3/baselines/split_A/model.joblib.
No image preprocessing is performed here: only cached DINOv2 feature vectors.
Writes a separate test_results_reproduction.json; never overwrites original results.
"""

import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

V2 = Path("outputs/dataset_v2")
V3 = Path("outputs/dataset_v3")
OUT = V3 / "baselines/split_A"
DEST = OUT / "test_results_reproduction.json"


def main():
    if DEST.exists():
        raise FileExistsError(DEST)

    with (V3 / "splits/split_A_exact.csv").open(newline="") as f:
        rows = [row for row in csv.DictReader(f) if row["split"] == "test"]
    assert len(rows) == 2743
    needed = {row["image_id"] for row in rows}

    features = {}
    for file in sorted((V2 / "features").glob("*.npz")):
        with np.load(file, allow_pickle=True) as data:
            for path, vector in zip(data["paths"], data["X"]):
                path = str(path)
                if "/nffa_sem_100/" in path:
                    path = path.split("/nffa_sem_100/", 1)[1]
                if path in needed:
                    if path in features:
                        raise ValueError(f"Duplicate feature for {path}")
                    features[path] = vector
    assert len(features) == len(needed)

    X = np.stack([features[r["image_id"]] for r in rows]).astype(np.float32)
    y = np.array([int(r["label"]) for r in rows])
    saved = joblib.load(OUT / "model.joblib")
    pred = saved["classifier"].predict(saved["scaler"].transform(X))

    accuracy = float(accuracy_score(y, pred))
    macro_f1 = float(f1_score(y, pred, average="macro"))
    names = {int(r["label"]): r["class_name"] for r in rows}
    labels = sorted(names)

    print("Split A test accuracy:", accuracy)
    print("Split A test macro-F1:", macro_f1)
    print(classification_report(
        y, pred, labels=labels, target_names=[names[i] for i in labels],
        digits=3, zero_division=0
    ))

    original = json.loads((OUT / "test_results.json").read_text())
    print("Original accuracy:", original["accuracy"])
    print("Original macro-F1:", original["macro_f1"])
    print("Accuracy matches:", abs(accuracy - original["accuracy"]) < 1e-12)
    print("Macro-F1 matches:", abs(macro_f1 - original["macro_f1"]) < 1e-12)

    DEST.write_text(json.dumps({
        "split": "A_exact",
        "test_images": len(rows),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "classification_report": classification_report(
            y, pred, labels=labels, target_names=[names[i] for i in labels],
            output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
        "class_names": [names[i] for i in labels],
    }, indent=2))
    print("Saved:", DEST)


if __name__ == "__main__":
    main()
