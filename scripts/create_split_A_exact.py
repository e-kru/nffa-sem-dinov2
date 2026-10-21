import csv
import json
import random
from pathlib import Path
from collections import defaultdict, Counter

BASE = Path("outputs/dataset_v3")
OUT = BASE / "splits"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

CSV_OUT = OUT / "split_A_exact.csv"
JSON_OUT = OUT / "split_A_exact.json"
CONFIG_OUT = OUT / "split_A_exact_config.json"

# Never silently overwrite an existing split
for path in (CSV_OUT, JSON_OUT, CONFIG_OUT):
    if path.exists():
        raise FileExistsError(f"Already exists: {path}")

# 1. Load cleaned Dataset V3
with open(BASE / "manifest.csv", newline="") as f:
    manifest = {
        r["image_id"]: r
        for r in csv.DictReader(f)
        if r["included"] == "1"
    }

assert len(manifest) == 18286

# 2. Union-Find for exact duplicates only
parent = {p: p for p in manifest}
size = {p: 1 for p in manifest}

def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def union(a, b):
    a, b = find(a), find(b)

    if a == b:
        return

    if size[a] < size[b]:
        a, b = b, a

    parent[b] = a
    size[a] += size[b]

exact_edges = 0

with open(
    BASE / "duplicates/candidate_pairs_v3.csv",
    newline=""
) as f:
    for r in csv.DictReader(f):

        # ONLY exact SHA or pixel matches.
        # pHash and DINOv2 play no role here.
        if not (
            r["original_sha_identical"] == "1"
            or r["pixel_identical"] == "1"
        ):
            continue

        a, b = r["image_a"], r["image_b"]

        assert a in manifest and b in manifest

        assert (
            manifest[a]["class_name"]
            == manifest[b]["class_name"]
        ), f"Conflicting labels: {a}, {b}"

        union(a, b)
        exact_edges += 1

# 3. Connected components
components = defaultdict(list)

for image_id in sorted(manifest):
    components[find(image_id)].append(image_id)

component_list = sorted(
    components.values(),
    key=lambda members: members[0],
)

multi = [g for g in component_list if len(g) >= 2]

assert len(component_list) == 18011
assert len(multi) == 266
assert sum(map(len, multi)) == 541

groups = []

for i, members in enumerate(component_list):
    cls = manifest[members[0]]["class_name"]

    groups.append({
        "group_id": f"A_{i:05d}",
        "class_name": cls,
        "members": members,
    })

print("--- EXACT DUPLICATES ---")
print("Exact pair edges:", exact_edges)
print("Total groups:", len(groups))
print("Groups with >= 2 images:", len(multi))
print("Images in duplicate groups:", 541)

# 4. Class-stratified group-aware split
rng = random.Random(SEED)
by_class = defaultdict(list)

for group in groups:
    by_class[group["class_name"]].append(group)

assignments = {}

for cls in sorted(by_class):
    class_groups = by_class[cls]

    # Reproducible randomized order
    rng.shuffle(class_groups)

    total = sum(len(g["members"]) for g in class_groups)

    targets = {
        "train": round(total * 0.70),
        "val": round(total * 0.15),
    }
    targets["test"] = total - targets["train"] - targets["val"]

    counts = Counter()

    for group in class_groups:
        # Place the whole group into the split
        # with the largest remaining target deficit.
        split = max(
            RATIOS,
            key=lambda s: targets[s] - counts[s]
        )

        assignments[group["group_id"]] = split
        counts[split] += len(group["members"])

# 5. Build output rows
rows = []
splits = {s: [] for s in RATIOS}
group_splits = defaultdict(set)

for group in groups:
    split = assignments[group["group_id"]]

    for image_id in group["members"]:
        rows.append({
            "image_id": image_id,
            "label": manifest[image_id]["label"],
            "class_name": group["class_name"],
            "group_id": group["group_id"],
            "split": split,
        })

        splits[split].append(image_id)
        group_splits[group["group_id"]].add(split)

# 6. Verify
assert len(rows) == 18286
assert len(set(r["image_id"] for r in rows)) == 18286
assert all(len(s) == 1 for s in group_splits.values())

# Independently verify all exact duplicate edges
image_split = {r["image_id"]: r["split"] for r in rows}

with open(
    BASE / "duplicates/candidate_pairs_v3.csv",
    newline=""
) as f:
    for r in csv.DictReader(f):
        if (
            r["original_sha_identical"] == "1"
            or r["pixel_identical"] == "1"
        ):
            assert (
                image_split[r["image_a"]]
                == image_split[r["image_b"]]
            ), "Exact duplicate leaked across splits"

# 7. Save
with open(CSV_OUT, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "image_id", "label", "class_name",
            "group_id", "split"
        ]
    )
    writer.writeheader()
    writer.writerows(rows)

for split in splits:
    splits[split].sort()

JSON_OUT.write_text(json.dumps(splits, indent=2))

CONFIG_OUT.write_text(json.dumps({
    "seed": SEED,
    "ratios": RATIOS,
    "grouping": "Original SHA-256 OR post-Crop-V2 pixel SHA-256",
    "uses_phash": False,
    "uses_dinov2_similarity": False,
    "images": 18286,
    "groups": len(groups),
}, indent=2))

# 8. Report
print("\n--- SPLIT A: EXACT ONLY ---")

for split, paths in splits.items():
    print(
        f"{split:5s}: {len(paths):5d} images "
        f"({len(paths) / 18286:.2%})"
    )

print("\nClass distribution:")
print(f"{'Class':35s} {'Train':>6} {'Val':>6} {'Test':>6}")

for cls in sorted(by_class):
    subset = [
        r for r in rows
        if r["class_name"] == cls
    ]
    c = Counter(r["split"] for r in subset)

    print(
        f"{cls:35s} "
        f"{c['train']:6d} "
        f"{c['val']:6d} "
        f"{c['test']:6d}"
    )

print("\nExact duplicate leakage: 0")
print("Saved:", CSV_OUT)
print("Saved:", JSON_OUT)