import csv
import json
from pathlib import Path

import joblib
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

V2 = Path("outputs/dataset_v2")
V3 = Path("outputs/dataset_v3")
OUT = V3 / "baselines/split_A"
OUT.mkdir(parents=True, exist_ok=True)

# 1. Load Split A
with open(V3 / "splits/split_A_exact.csv", newline="") as f:
    rows = list(csv.DictReader(f))

assert len(rows) == 18286

# 2. Load existing DINOv2 features
features = {}

for file in sorted((V2 / "features").glob("*.npz")):
    with np.load(file, allow_pickle=True) as data:
        for path, vector in zip(data["paths"], data["X"]):
            path = str(path)

            if "/nffa_sem_100/" in path:
                path = path.split("/nffa_sem_100/", 1)[1]

            if path in features:
                raise ValueError(f"Duplicate feature: {path}")

            features[path] = vector

# 3. Assemble features and labels
missing = [
    r["image_id"] for r in rows
    if r["image_id"] not in features
]
assert not missing, f"Missing features: {missing[:5]}"

X = np.stack([
    features[r["image_id"]] for r in rows
]).astype(np.float32)

y = np.array([int(r["label"]) for r in rows])
splits = np.array([r["split"] for r in rows])

class_names = {
    int(r["label"]): r["class_name"]
    for r in rows
}

assert X.shape == (18286, 384)
assert np.isfinite(X).all()

# 4. Separate Train / Validation
train = splits == "train"
val = splits == "val"
test = splits == "test"

assert (train.sum(), val.sum(), test.sum()) == (
    12801, 2742, 2743
)

X_train, y_train = X[train], y[train]
X_val, y_val = X[val], y[val]

# 5. Fit scaler ONLY on training data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# 6. Train and tune logistic regression
C_VALUES = [0.001, 0.01, 0.1, 1, 10, 100]
results = []

best_score = -1
best_model = None
best_C = None

for C in C_VALUES:
    model = LogisticRegression(
        C=C,
        solver="lbfgs",
        max_iter=3000,
        random_state=42,
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_val)

    accuracy = accuracy_score(y_val, pred)
    macro_f1 = f1_score(y_val, pred, average="macro")

    results.append({
        "C": C,
        "val_accuracy": float(accuracy),
        "val_macro_f1": float(macro_f1),
    })

    print(
        f"C={C:<8g} "
        f"Val Accuracy={accuracy:.4f} "
        f"Macro-F1={macro_f1:.4f}"
    )

    if macro_f1 > best_score:
        best_score = macro_f1
        best_model = model
        best_C = C

# 7. Save model and validation results
joblib.dump(
    {"scaler": scaler, "classifier": best_model},
    OUT / "model.joblib",
)

(OUT / "validation_results.json").write_text(
    json.dumps({
        "split": "A_exact",
        "best_C": best_C,
        "selection_metric": "val_macro_f1",
        "results": results,
    }, indent=2)
)

print("\n--- SPLIT A RESULTS ---")
print("Best C:", best_C)
print("Best Validation Macro-F1:", round(best_score, 4))

final_pred = best_model.predict(X_val)

print("\nPer-class validation results:")
labels = sorted(class_names)

print(classification_report(
    y_val,
    final_pred,
    labels=labels,
    target_names=[class_names[i] for i in labels],
    digits=3,
    zero_division=0,
))

print("Test set not evaluated.")
print("Saved:", OUT)