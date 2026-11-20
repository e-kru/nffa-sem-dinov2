import csv
from pathlib import Path
from collections import defaultdict, Counter
from itertools import combinations

import numpy as np

V2 = Path("outputs/dataset_v2")
V3 = Path("outputs/dataset_v3")
OUT = V3 / "duplicates"
OUT.mkdir(parents=True, exist_ok=True)

# 1. Final dataset
with open(V3 / "manifest.csv", newline="") as f:
    manifest = {
        r["image_id"]: r
        for r in csv.DictReader(f)
        if r["included"] == "1"
    }

ids = sorted(manifest)
idx = {p: i for i, p in enumerate(ids)}
n = len(ids)
assert n == 18286

# 2. Existing hashes from Crop V2
with open(V2 / "image_hashes.csv", newline="") as f:
    hashes = {
        r["image_id"]: r
        for r in csv.DictReader(f)
    }

assert all(p in hashes for p in ids)

ph = np.array([
    int(hashes[p]["phash"], 16)
    for p in ids
], dtype=np.uint64)

# Evidence flags
SHA = 1
PIXEL = 2
PHASH = 4
DINO = 8

pairs = defaultdict(int)

def add_pair(i, j, flag):
    if i == j:
        return
    if i > j:
        i, j = j, i
    pairs[(i, j)] |= flag

# 3. Original SHA duplicate groups
sha_groups = defaultdict(list)

for p in ids:
    sha_groups[manifest[p]["exact_group_id"]].append(idx[p])

for members in sha_groups.values():
    if len(members) > 1:
        for i, j in combinations(members, 2):
            add_pair(i, j, SHA)

# 4. Exact duplicates after Crop V2
pixel_groups = defaultdict(list)

for p in ids:
    pixel_groups[hashes[p]["pixel_sha256"]].append(idx[p])

for members in pixel_groups.values():
    if len(members) > 1:
        for i, j in combinations(members, 2):
            add_pair(i, j, PIXEL)

# 5. Global pHash search, distance <= 8
for start in range(0, n, 128):
    end = min(start + 128, n)

    distances = np.bitwise_count(
        np.bitwise_xor(ph[start:end, None], ph[None, :])
    )

    for local_i in range(end - start):
        i = start + local_i
        js = np.flatnonzero(distances[local_i, i+1:] <= 8)

        for offset in js:
            add_pair(i, i + 1 + int(offset), PHASH)

# 6. Add new global DINOv2 high-similarity pairs
with open(
    OUT / "dinov2_high_similarity_v3.csv", newline=""
) as f:
    for r in csv.DictReader(f):
        add_pair(
            idx[r["image_a"]],
            idx[r["image_b"]],
            DINO
        )

print(f"Unique candidate pairs: {len(pairs)}")

# 7. Load existing DINOv2 features
X = np.zeros((n, 384), dtype=np.float32)
filled = np.zeros(n, dtype=bool)

for file in sorted((V2 / "features").glob("*.npz")):
    with np.load(file, allow_pickle=True) as data:
        for p, x in zip(data["paths"], data["X"]):
            p = str(p)

            if "/nffa_sem_100/" in p:
                p = p.split("/nffa_sem_100/", 1)[1]

            if p in idx:
                i = idx[p]
                assert not filled[i], f"Duplicate feature: {p}"
                X[i] = x
                filled[i] = True

assert filled.all(), f"Missing {sum(~filled)} features"

X /= np.maximum(
    np.linalg.norm(X, axis=1, keepdims=True), 1e-12
)

# 8. Measure similarity for every candidate
pair_keys = sorted(pairs)
ii = np.array([p[0] for p in pair_keys])
jj = np.array([p[1] for p in pair_keys])

cosines = np.einsum(
    "ij,ij->i",
    X[ii],
    X[jj]
)

output = OUT / "candidate_pairs_v3.csv"
counts = Counter()

fields = [
    "image_a", "image_b",
    "class_a", "class_b",
    "original_sha_identical",
    "pixel_identical",
    "phash_distance",
    "cosine_similarity",
    "category",
]

with open(output, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

    for (i, j), cosine in zip(pair_keys, cosines):
        a, b = ids[i], ids[j]
        flags = pairs[(i, j)]
        distance = int((int(ph[i]) ^ int(ph[j])).bit_count())

        if flags & (SHA | PIXEL):
            category = "exact"
        elif distance <= 4 and cosine >= 0.99:
            category = "strong_near_duplicate"
        elif distance <= 8 and cosine >= 0.98:
            category = "possible_near_duplicate"
        elif cosine >= 0.995:
            category = "dino_candidate"
        else:
            category = "other_phash_candidate"

        counts[category] += 1

        if manifest[a]["label"] != manifest[b]["label"]:
            counts["cross_class"] += 1

        writer.writerow({
            "image_a": a,
            "image_b": b,
            "class_a": manifest[a]["class_name"],
            "class_b": manifest[b]["class_name"],
            "original_sha_identical": int(bool(flags & SHA)),
            "pixel_identical": int(bool(flags & PIXEL)),
            "phash_distance": distance,
            "cosine_similarity": round(float(cosine), 6),
            "category": category,
        })

print("\n--- V3 DUPLICATE CANDIDATES ---")
print("Images:", n)
print("Candidate pairs:", len(pair_keys))

for category, count in counts.items():
    print(f"{category}: {count}")

print("\nSaved:", output)