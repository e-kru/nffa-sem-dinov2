
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from src.dataset import get_dataset, DATASET_ROOT
from src.dinov2_encoder import (
    get_dinov2,
    get_dinov2_transform,
)


def extract_features(model, dataset, indices):
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=4,
        shuffle=False,
        num_workers=0
    )

    features = []
    labels = []

    with torch.inference_mode():
        for images, y in loader:
            z = model(images)
            features.append(z.cpu())
            labels.append(y.cpu())

    X = torch.cat(features).numpy()
    y = torch.cat(labels).numpy()

    return X, y


def main():
    with open("outputs/splits.json") as f:
        splits = json.load(f)

    dataset = get_dataset(
        transform=get_dinov2_transform()
    )

    # Map relative image paths to ImageFolder indices
    path_to_idx = {
        Path(path).relative_to(DATASET_ROOT).as_posix(): i
        for i, (path, _) in enumerate(dataset.samples)
    }

    model = get_dinov2()
    model.eval()

    # Small balanced experiment: 90 images total
    sample_sizes = {
        "train": 5,
        "val": 2,
        "test": 2
    }

    results = {}

    for split_name, n_per_class in sample_sizes.items():
        records = []

        for class_idx in range(len(dataset.classes)):
            candidates = [
                r for r in splits[split_name]
                if r["label"] == class_idx
            ]
            records.extend(candidates[:n_per_class])

        indices = [path_to_idx[r["path"]] for r in records]

        print(f"Extracting {split_name}: {len(indices)} images")

        X, y = extract_features(model, dataset, indices)
        results[split_name] = (X, y)

        print("Features:", X.shape)

    # Save features so DINOv2 need not run again
    Path("outputs").mkdir(exist_ok=True)

    np.savez(
        "outputs/features_smoke.npz",
        **{
            f"{name}_{key}": value
            for name, (X, y) in results.items()
            for key, value in [("X", X), ("y", y)]
        }
    )

    X_train, y_train = results["train"]

    # Multinomial linear classifier
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )
    )

    classifier.fit(X_train, y_train)

    for name in ["val", "test"]:
        X, y = results[name]
        predictions = classifier.predict(X)

        print(f"\n{name.upper()}")
        print("Accuracy:", accuracy_score(y, predictions))
        print(
            "Macro-F1:",
            f1_score(y, predictions, average="macro")
        )


if __name__ == "__main__":
    main()
