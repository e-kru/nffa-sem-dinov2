import csv
import json
import random
from collections import defaultdict, Counter
from pathlib import Path

BASE = Path("outputs/dataset_v3")
OUT = BASE / "splits"
OUT.mkdir(parents=True, exist_ok=True)

RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
NAMES = list(RATIOS)

output = OUT / "split_B_full.csv"
if output.exists():
    raise FileExistsError(output)

with open(BASE / "manifest.csv", newline="") as f:
    manifest = {
        r["image_id"]: r
        for r in csv.DictReader(f)
        if r["included"] == "1"
    }

assert len(manifest) == 18286

# 1. Build connected components using all candidate edges
parent = {p: p for p in manifest}
sizes = {p: 1 for p in manifest}

def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def union(a, b):
    a, b = find(a), find(b)
    if a == b:
        return
    if sizes[a] < sizes[b]:
        a, b = b, a
    parent[b] = a
    sizes[a] += sizes[b]

edges = 0
with open(BASE / "duplicates/candidate_pairs_v3.csv", newline="") as f:
    for r in csv.DictReader(f):
        union(r["image_a"], r["image_b"])
        edges += 1

components = defaultdict(list)
for p in sorted(manifest):
    components[find(p)].append(p)

groups = sorted(components.values(), key=lambda g: g[0])
multi = [g for g in groups if len(g) > 1]

assert len(groups) == 16074
assert len(multi) == 778

# 2. Get label counts for each component
class_totals = Counter(int(r["label"]) for r in manifest.values())
total = len(manifest)

group_data = []
for i, members in enumerate(groups):
    label_counts = Counter(
        int(manifest[p]["label"]) for p in members
    )
    group_data.append((
        f"B_{i:05d}", members, label_counts
    ))

# 3. Group-aware, class-balanced greedy assignment
# Try several deterministic shuffled orders.
best = None

for attempt in range(8):
    rng = random.Random(42 + attempt)
    ordered = group_data.copy()
    rng.shuffle(ordered)
    ordered.sort(key=lambda g: -len(g[1]))

    targets = {
        s: {c: n * RATIOS[s] for c, n in class_totals.items()}
        for s in NAMES
    }
    total_targets = {s: total * RATIOS[s] for s in NAMES}

    counts = {s: Counter() for s in NAMES}
    totals = Counter()
    assignments = {}

    for group_id, members, labels in ordered:
        group_size = len(members)

        def cost(split):
            value = 0.0
            for c, amount in labels.items():
                gap = counts[split][c] - targets[split][c]
                value += (
                    2 * gap * amount + amount**2
                ) / max(targets[split][c], 1)

            gap = totals[split] - total_targets[split]
            value += 0.5 * (
                2 * gap * group_size + group_size**2
            ) / max(total_targets[split], 1)
            return value

        split = min(rng.sample(NAMES, 3), key=cost)
        assignments[group_id] = split
        counts[split].update(labels)
        totals[split] += group_size

    score = sum(
        (counts[s][c] - targets[s][c])**2
        / max(targets[s][c], 1)
        for s in NAMES
        for c in class_totals
    )

    if best is None or score < best[0]:
        best = (score, assignments.copy())

assignments = best[1]

# 4. Save image-level assignments
rows = []
split_lists = {s: [] for s in NAMES}

for group_id, members, _ in group_data:
    split = assignments[group_id]
    for p in members:
        rows.append({
            "image_id": p,
            "label": manifest[p]["label"],
            "class_name": manifest[p]["class_name"],
            "group_id": group_id,
            "split": split,
        })
        split_lists[split].append(p)

assert len(rows) == 18286
image_split = {r["image_id"]: r["split"] for r in rows}

# Verify that no candidate edge crosses splits
with open(BASE / "duplicates/candidate_pairs_v3.csv", newline="") as f:
    for r in csv.DictReader(f):
        assert (
            image_split[r["image_a"]]
            == image_split[r["image_b"]]
        ), "Leakage across Split B"

with open(output, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

json_path = OUT / "split_B_full.json"
if json_path.exists():
    raise FileExistsError(json_path)

json_path.write_text(json.dumps({
    s: sorted(paths) for s, paths in split_lists.items()
}, indent=2))

config_path = OUT / "split_B_full_config.json"
config_path.write_text(json.dumps({
    "method": "All pHash <= 8, exact SHA/pixel, DINO cosine >= 0.995",
    "seed": 42,
    "assignment_attempts": 8,
    "candidate_edges": edges,
    "groups": len(groups),
    "multi_image_groups": len(multi),
    "ratios": RATIOS,
}, indent=2))

print("\n--- SPLIT B: FULL SIMILARITY GROUPING ---")
print("Candidate edges:", edges)
print("Total groups:", len(groups))
print("Multi-image groups:", len(multi))
print("Images in multi-image groups:", sum(map(len, multi)))
print("Largest group:", max(map(len, groups)))

for s in NAMES:
    n = len(split_lists[s])
    print(f"{s}: {n} images ({n / total:.2%})")

print("\nClass distribution:")
print(f"{'Class':35s} {'Train':>6} {'Val':>6} {'Test':>6}")

for c in sorted(class_totals):
    name = next(
        r["class_name"] for r in manifest.values()
        if int(r["label"]) == c
    )
    nums = [
        sum(int(r["label"]) == c for r in rows if r["split"] == s)
        for s in NAMES
    ]
    print(f"{name:35s} {nums[0]:6d} {nums[1]:6d} {nums[2]:6d}")

print("\nCandidate edges crossing splits: 0")
print("Saved:", output)