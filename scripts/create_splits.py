import json
from pathlib import Path
from collections import Counter

from sklearn.model_selection import train_test_split

from src.dataset import get_dataset, DATASET_ROOT


SEED = 42
OUTPUT = Path("outputs/splits.json")


def main():
    dataset = get_dataset()

    labels = [label for _, label in dataset.samples]
    indices = list(range(len(dataset)))

    # 70% train, 30% remaining
    train_idx, remaining_idx = train_test_split(
        indices,
        test_size=0.30,
        random_state=SEED,
        stratify=labels
    )

    # Split remaining into 15% validation, 15% test
    val_idx, test_idx = train_test_split(
        remaining_idx,
        test_size=0.50,
        random_state=SEED,
        stratify=[labels[i] for i in remaining_idx]
    )

    def make_records(indices):
        return [
            {
                "path": str(
                    Path(dataset.samples[i][0]).relative_to(DATASET_ROOT)
                ),
                "label": int(dataset.samples[i][1])
            }
            for i in indices
        ]

    splits = {
        "seed": SEED,
        "class_to_idx": dataset.class_to_idx,
        "train": make_records(train_idx),
        "val": make_records(val_idx),
        "test": make_records(test_idx)
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT, "w") as f:
        json.dump(splits, f, indent=2)

    for name in ["train", "val", "test"]:
        records = splits[name]
        counts = Counter(item["label"] for item in records)

        print(f"\n{name}: {len(records)} images")

        for idx, class_name in enumerate(dataset.classes):
            print(f"  {class_name}: {counts[idx]}")


if __name__ == "__main__":
    main()