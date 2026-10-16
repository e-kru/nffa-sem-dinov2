
import csv
import json
from collections import defaultdict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from scripts.train_classifier import (
    load_features,
    get_splits,
    train_classifier,
)


# Load the existing SHA-256 audit
groups = defaultdict(list)

with open("outputs/audit/sha256_duplicates.csv") as f:
    for row in csv.DictReader(f):
        groups[row["sha256"]].append(
            (row["split"], row["path"])
        )


# Find validation images with an identical training image
leaked_val_paths = set()

for images in groups.values():
    splits_present = {split for split, _ in images}

    if "train" in splits_present:
        for split, path in images:
            if split == "val":
                leaked_val_paths.add(path)


# Load features and train the existing baseline
X, y, paths = load_features()
splits = get_splits(X, y, paths)

X_train, y_train = splits["train"]
X_val, y_val = splits["val"]

classifier = train_classifier(X_train, y_train)
predictions = classifier.predict(X_val)


# Match the JSON validation order
with open("outputs/splits.json") as f:
    split_info = json.load(f)

val_paths = [r["path"] for r in split_info["val"]]

mask = np.array([
    path not in leaked_val_paths
    for path in val_paths
])

assert len(mask) == len(y_val)


print(f"Excluded validation images: {(~mask).sum()}")
print(f"Remaining validation images: {mask.sum()}")

print("\nOriginal validation:")
print("Accuracy:", accuracy_score(y_val, predictions))
print("Macro-F1:", f1_score(y_val, predictions, average="macro"))

print("\nWithout exact train duplicates:")
print("Accuracy:", accuracy_score(y_val[mask], predictions[mask]))
print("Macro-F1:", f1_score(
    y_val[mask], predictions[mask], average="macro"
))
