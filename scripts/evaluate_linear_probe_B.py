import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

V2 = Path("outputs/dataset_v2")
V3 = Path("outputs/dataset_v3")
OUT = V3 / "baselines/split_B"

result_file = OUT / "test_results.json"
if result_file.exists():
    raise FileExistsError(result_file)

# 1. Load test images from Split B
with open(V3 / "splits/split_B_full.csv", newline="") as f:
    rows = [
        r for r in csv.DictReader(f)
        if r["split"] == "test"
    ]

assert len(rows) == 2744
needed = {r["image_id"] for r in rows}

# 2. Load existing DINOv2 features
features = {}

for file in sorted((V2 / "features").glob("*.npz")):
    with np.load(file, allow_pickle=True) as data:
        for path, vector in zip(data["paths"], data["X"]):
            path = str(path)

            if "/nffa_sem_100/" in path:
                path = path.split("/nffa_sem_100/", 1)[1]

            if path in needed:
                assert path not in features
                features[path] = vector

assert len(features) == len(needed)

X_test = np.stack([
    features[r["image_id"]] for r in rows
]).astype(np.float32)

y_test = np.array([
    int(r["label"]) for r in rows
])

# 3. Load trained Split B classifier
saved = joblib.load(OUT / "model.joblib")
scaler = saved["scaler"]
classifier = saved["classifier"]

X_test = scaler.transform(X_test)
pred = classifier.predict(X_test)

# 4. Evaluate
accuracy = accuracy_score(y_test, pred)
macro_f1 = f1_score(y_test, pred, average="macro")

names = {
    int(r["label"]): r["class_name"]
    for r in rows
}
labels = sorted(names)

report = classification_report(
    y_test,
    pred,
    labels=labels,
    target_names=[names[i] for i in labels],
    digits=3,
    zero_division=0,
)

print("\n--- SPLIT B: FINAL TEST RESULTS ---")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Macro-F1: {macro_f1:.4f}")
print("\n", report)

# 5. Save results
results = {
    "split": "B_full",
    "test_images": len(rows),
    "accuracy": float(accuracy),
    "macro_f1": float(macro_f1),
    "classification_report": classification_report(
        y_test,
        pred,
        labels=labels,
        target_names=[names[i] for i in labels],
        output_dict=True,
        zero_division=0,
    ),
    "confusion_matrix": confusion_matrix(
        y_test, pred, labels=labels
    ).tolist(),
    "class_names": [names[i] for i in labels],
}

result_file.write_text(json.dumps(results, indent=2))
print("Saved:", result_file)