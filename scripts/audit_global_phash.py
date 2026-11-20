
import argparse
import csv
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm


ROOT = Path("outputs/dataset_v2")
INPUT = ROOT / "image_hashes.csv"
MANIFEST = ROOT / "manifest.csv"
OUTPUT = ROOT / "phash_candidate_pairs.csv"

BLOCK_SIZE = 128
_HASHES = None
_MAX_DISTANCE = None


def init_worker(hashes, max_distance):
    global _HASHES, _MAX_DISTANCE
    _HASHES = hashes
    _MAX_DISTANCE = max_distance


def scan_block(start):
    """Find similar pairs in one block of rows."""

    n = len(_HASHES)
    end = min(start + BLOCK_SIZE, n)

    # Compare this block with all images.
    xor = np.bitwise_xor(
        _HASHES[start:end, None],
        _HASHES[None, :]
    )

    # Hamming distance of 64-bit pHashes.
    distances = np.bitwise_count(xor)

    # Only retain pairs (i, j) with i < j.
    row_ids = np.arange(start, end)
    col_ids = np.arange(n)

    matches = (
        (distances <= _MAX_DISTANCE)
        & (row_ids[:, None] < col_ids[None, :])
    )

    rows, cols = np.nonzero(matches)

    return [
        (start + int(i), int(j), int(distances[i, j]))
        for i, j in zip(rows, cols)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-distance", type=int, default=4)
    args = parser.parse_args()

    if not 0 <= args.max_distance <= 64:
        parser.error("--max-distance must be between 0 and 64")

    with open(INPUT, newline="") as f:
        rows = list(csv.DictReader(f))

    with open(MANIFEST, newline="") as f:
        manifest = {
            row["image_id"]: row
            for row in csv.DictReader(f)
        }

    paths = [row["image_id"] for row in rows]

    included = {
        path for path, row in manifest.items()
        if row["included"] == "1"
    }

    if len(paths) != len(set(paths)) or set(paths) != included:
        raise ValueError("Hash file does not match included images")

    # Convert hexadecimal pHashes to uint64.
    hashes = np.array(
        [int(row["phash"], 16) for row in rows],
        dtype=np.uint64
    )

    pixel_hashes = [
        row["pixel_sha256"] for row in rows
    ]
    labels = [
        manifest[path]["label"] for path in paths
    ]

    starts = list(range(0, len(paths), BLOCK_SIZE))
    counts = Counter()
    affected = set()

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=init_worker,
        initargs=(hashes, args.max_distance),
    ) as executor:

        results = executor.map(scan_block, starts)

        with open(OUTPUT, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "image_a",
                "image_b",
                "phash_distance",
                "same_class",
                "same_pixels",
            ])

            for pairs in tqdm(results, total=len(starts)):
                for i, j, distance in pairs:
                    same_class = labels[i] == labels[j]
                    same_pixels = (
                        pixel_hashes[i] == pixel_hashes[j]
                    )

                    writer.writerow([
                        paths[i],
                        paths[j],
                        distance,
                        int(same_class),
                        int(same_pixels),
                    ])

                    counts[distance] += 1
                    affected.update((i, j))

                    if not same_class:
                        counts["cross_class"] += 1

                    if same_pixels:
                        counts["same_pixels"] += 1

    print("\n--- Global pHash Audit ---")
    print(f"Images: {len(paths)}")
    print(f"Candidate pairs: {sum(counts[d] for d in range(args.max_distance + 1))}")
    print(f"Images involved: {len(affected)}")

    for d in range(args.max_distance + 1):
        print(f"Distance {d}: {counts[d]} pairs")

    print(f"Exact pixel pairs: {counts['same_pixels']}")
    print(f"Cross-class pairs: {counts['cross_class']}")
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
