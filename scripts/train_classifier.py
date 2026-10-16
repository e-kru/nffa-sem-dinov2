import json
from pathlib import Path

import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report


FEATURE_DIR = Path("outputs/features_dinov2_vits14")
SPLIT_FILE = Path("outputs/splits.json")


def load_features():
    """Load all DINOv2 feature chunks."""

    files = sorted(FEATURE_DIR.glob("chunk_*.npz"))

    features = []
    labels = []
    paths = []

    for file in files:
        with np.load(file, allow_pickle=False) as data:
            features.append(data["X"])
            labels.append(data["y"])
            paths.extend(data["paths"].tolist())

    X = np.concatenate(features, axis=0)
    y = np.concatenate(labels, axis=0)

    assert X.shape == (21169, 384)
    assert len(set(paths)) == len(paths)

    return X, y, paths


def get_splits(X, y, paths):
    """Match feature vectors to train, validation and test splits."""

    with open(SPLIT_FILE) as f:
        splits = json.load(f)

    path_to_idx = {
        path: i for i, path in enumerate(paths)
    }

    result = {}

    used_paths = set()

    for name in ["train", "val", "test"]:
        records = splits[name]

        indices = []

        for record in records:
            path = record["path"]
            label = record["label"]

            if path in used_paths:
                raise ValueError(f"Duplicate split assignment: {path}")

            i = path_to_idx[path]

            if int(y[i]) != label:
                raise ValueError(f"Label mismatch: {path}")

            used_paths.add(path)
            indices.append(i)

        result[name] = (X[indices], y[indices])

    assert len(used_paths) == len(paths)

    return result

def train_classifier(X_train, y_train):
    """Train a linear classifier on frozen DINOv2 features."""

    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        )
    )

    classifier.fit(X_train, y_train)

    return classifier


if __name__ == "__main__":

    X, y, paths = load_features()
    splits = get_splits(X, y, paths)

    for name, (X_split, y_split) in splits.items():
        print(f"{name}: X={X_split.shape}, y={y_split.shape}")

        X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]

    # Train classifier
    classifier = train_classifier(X_train, y_train)

    # Predict validation labels
    y_pred = classifier.predict(X_val)

    # Evaluate
    accuracy = accuracy_score(y_val, y_pred)
    macro_f1 = f1_score(y_val, y_pred, average="macro")

    print("\n--- Validation Results ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, zero_division=0))

        # Evaluate on training data
    y_train_pred = classifier.predict(X_train)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_macro_f1 = f1_score(y_train, y_train_pred, average="macro")

    print("\n--- Training Results ---")
    print(f"Accuracy: {train_accuracy:.4f}")
    print(f"Macro-F1: {train_macro_f1:.4f}")

    print("\nTraining Classification Report:")
    print(classification_report(
        y_train,
        y_train_pred,
        zero_division=0
    ))