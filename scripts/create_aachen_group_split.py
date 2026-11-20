"""Build reproducible, leakage-aware Aachen-Heerlen segmentation split.

Groups are (type, temperature, direction, distance), without detector angle.
These are *metadata groups*, NOT verified distinct microscope locations.

Example (run from repository root):
    python scripts/create_aachen_group_split.py
"""

import argparse
import csv
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src.config import AACHEN_ROOT


DEFAULT_ROOT = AACHEN_ROOT
DEFAULT_MANIFEST = Path("outputs/aachen_heerlen/splits/group_split_seed42.csv")
SPLITS = ("train", "val", "test")
TARGET = np.array([0.70, 0.15, 0.15], dtype=float)
FIELDS = (
    "image_id", "image_path", "mask_path", "split", "group_id",
    "type", "temperature", "direction", "distance", "angle",
)


def image_number(name):
    return int(Path(name).stem.rsplit("_", 1)[-1])


def make_group(row):
    # Deliberately omit angle; tilted/perpendicular images may show the same area.
    return (row["type"], row["temperature"], row["direction"], row["distance"])


def load_metadata(root):
    images = root / "images"
    masks = root / "masks"
    metadata = root / "raw" / "nature_scidata_steel_metadata.csv"
    image_names = {p.name for p in images.glob("*.png")}
    mask_names = {p.name for p in masks.glob("*.png")}
    if not image_names:
        raise RuntimeError(f"No PNG images found under {images}")
    if image_names != mask_names:
        raise RuntimeError(
            f"Image/mask filename mismatch: images only={sorted(image_names-mask_names)[:5]}, "
            f"masks only={sorted(mask_names-image_names)[:5]}"
        )
    with metadata.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"image_url", "type", "temperature", "direction", "distance", "angle"}
        if not required.issubset(reader.fieldnames or []):
            raise RuntimeError(f"Missing metadata columns: {required-set(reader.fieldnames or [])}")
        rows = [r for r in reader if r["image_url"] in image_names]
    names = [r["image_url"] for r in rows]
    if len(rows) != len(image_names) or len(set(names)) != len(names):
        raise RuntimeError(
            f"Metadata mismatch: {len(image_names)} images, "
            f"{len(rows)} metadata rows, {len(set(names))} unique names"
        )
    if any(any(not r[field].strip() for field in make_group_fields()) for r in rows):
        raise RuntimeError("Empty group metadata; cannot make trustworthy groups")
    return sorted(rows, key=lambda r: image_number(r["image_url"]))


def make_group_fields():
    return ("type", "temperature", "direction", "distance")


def split_rows(rows, seed, trials):
    groups = defaultdict(list)
    for row in rows:
        groups[make_group(row)].append(row)
    keys = sorted(groups)
    n_groups = len(keys)
    if n_groups < 10:
        raise RuntimeError(f"Too few groups for this split: {n_groups}")

    # Stratification is approximate because whole groups cannot be divided.
    category_fields = ("type", "temperature", "direction", "angle")
    features = []
    for field in category_fields:
        cats = sorted({r[field] for r in rows})
        vectors = np.zeros((n_groups, len(cats)), dtype=np.float64)
        for i, key in enumerate(keys):
            counts = Counter(r[field] for r in groups[key])
            vectors[i] = [counts[c] for c in cats]
        features.append(vectors)

    sizes = np.array([len(groups[k]) for k in keys], dtype=np.int64)
    N = sizes.sum()
    rng = np.random.default_rng(seed)
    count_train = round(n_groups * TARGET[0])
    count_val = round(n_groups * TARGET[1])
    count_test = n_groups - count_train - count_val
    if min(count_train, count_val, count_test) == 0:
        raise RuntimeError("One split would have no groups")
    cuts = (count_train, count_train + count_val)

    reference = [v.sum(axis=0) / N for v in features]
    feature_weights = (0.6, 0.25, 0.15, 0.1)
    best = None
    for _ in range(trials):
        permutation = rng.permutation(n_groups)
        partitions = np.split(permutation, cuts)
        counts = np.array([int(sizes[p].sum()) for p in partitions])
        if np.any(counts == 0):
            continue
        fractions = counts / N
        # Prioritize the target sample fractions, not just the number of groups.
        size_penalty = 16.0 * np.abs(fractions - TARGET).sum()
        balance_penalty = 0.0
        for vectors, ref, weight in zip(features, reference, feature_weights):
            for p, count in zip(partitions, counts):
                distribution = vectors[p].sum(axis=0) / count
                balance_penalty += weight * np.abs(distribution - ref).sum()
        score = size_penalty + balance_penalty
        if best is None or score < best[0]:
            best = (score, [p.copy() for p in partitions], counts.copy())

    if best is None:
        raise RuntimeError("No valid partition found")
    _, partitions, counts = best
    assignment = {
        keys[i]: label for label, inds in zip(SPLITS, partitions) for i in inds
    }
    assert len(assignment) == n_groups
    return assignment, groups, counts


def write_manifest(rows, assignment, root, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for row in rows:
        name = row["image_url"]
        key = make_group(row)
        records.append({
            "image_id": name,
            "image_path": os.path.relpath(root / "images" / name, output.parent),
            "mask_path": os.path.relpath(root / "masks" / name, output.parent),
            "split": assignment[key],
            "group_id": "|".join(key),
            "type": row["type"],
            "temperature": row["temperature"],
            "direction": row["direction"],
            "distance": row["distance"],
            "angle": row["angle"],
        })
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)
    return records


def validate(records, output, n_groups):
    unique_images = {r["image_id"] for r in records}
    if len(unique_images) != len(records):
        raise RuntimeError("Duplicate image IDs in output")
    by_split = {}
    for split in SPLITS:
        subset = [r for r in records if r["split"] == split]
        if not subset:
            raise RuntimeError(f"Empty split: {split}")
        by_split[split] = {r["group_id"] for r in subset}
        for row in subset:
            for key in ("image_path", "mask_path"):
                if not (output.parent / row[key]).is_file():
                    raise RuntimeError(f"Missing file for {row['image_id']}: {row[key]}")
    for i, s in enumerate(SPLITS):
        for t in SPLITS[i + 1:]:
            if by_split[s] & by_split[t]:
                raise RuntimeError(f"Group leakage between {s} and {t}")
    if len(set.union(*by_split.values())) != n_groups:
        raise RuntimeError("Some metadata groups are missing")
    print("\n=== SPLIT ===")
    for split in SPLITS:
        selected = [r for r in records if r["split"] == split]
        print(f"{split:5s}: {len(selected):4d} images "
              f"({100 * len(selected) / len(records):5.2f}%), "
              f"{len(by_split[split]):3d} groups")
    print("\n=== MATERIAL TYPES PER SPLIT ===")
    types = sorted({r["type"] for r in records})
    print(f"{'type':>8s} {'train':>7s} {'val':>7s} {'test':>7s}")
    for typ in types:
        nums = [sum(r["type"] == typ and r["split"] == s for r in records)
                for s in SPLITS]
        print(f"{typ:>8s} {nums[0]:7d} {nums[1]:7d} {nums[2]:7d}")
    print("\nChecks: image IDs unique; all files exist; group overlap = 0")
    print("NOTE: These are metadata groups, not proven independent sample locations.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=5000)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be >= 1")
    root = args.root.resolve()
    output = args.output.resolve()
    rows = load_metadata(root)
    assignment, groups, _ = split_rows(rows, args.seed, args.trials)
    records = write_manifest(rows, assignment, root, output)
    validate(records, output, len(groups))
    print(f"\nManifest: {output}")
    print(f"Seed={args.seed}, candidate splits={args.trials}")


if __name__ == "__main__":
    main()
