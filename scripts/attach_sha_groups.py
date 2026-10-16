
import csv
import os
from collections import defaultdict
from pathlib import Path


MANIFEST = Path("outputs/dataset_v2/manifest.csv")
SHA_REPORT = Path("outputs/audit/sha256_duplicates.csv")


def main():
    # Load manifest
    with open(MANIFEST, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = reader.fieldnames

    image_ids = {row["image_id"] for row in rows}
    assert len(image_ids) == len(rows)

    # Load previously identified SHA duplicates
    sha_by_path = {}

    with open(SHA_REPORT, newline="") as f:
        for row in csv.DictReader(f):
            path = row["path"]
            sha = row["sha256"]

            if path not in image_ids:
                raise ValueError(f"Unknown image: {path}")

            if path in sha_by_path:
                raise ValueError(f"Duplicate SHA entry: {path}")

            sha_by_path[path] = sha

    # Attach exact-duplicate groups to every image
    for row in rows:
        path = row["image_id"]
        sha = sha_by_path.get(path, "")

        row["sha256"] = sha

        if sha:
            row["exact_group_id"] = f"sha:{sha}"
        else:
            row["exact_group_id"] = f"img:{path}"

    # Save updated manifest atomically
    fields = [
        field for field in fields
        if field not in ("sha256", "exact_group_id")
    ] + ["sha256", "exact_group_id"]

    temp_path = MANIFEST.with_suffix(".tmp")

    with open(temp_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    os.replace(temp_path, MANIFEST)

    # Count duplicate groups among included images
    groups = defaultdict(list)

    for row in rows:
        if row["included"] == "1":
            groups[row["exact_group_id"]].append(
                row["image_id"]
            )

    duplicates = [
        members for members in groups.values()
        if len(members) > 1
    ]

    print("\n--- SHA Groups ---")
    print(f"All images: {len(rows)}")
    print(f"Included images: {sum(r['included'] == '1' for r in rows)}")
    print(f"Images in original SHA report: {len(sha_by_path)}")
    print(f"Duplicate groups after filtering: {len(duplicates)}")
    print(f"Images in those groups: {sum(map(len, duplicates))}")
    print(f"Total initial groups: {len(groups)}")
    print(f"\nUpdated: {MANIFEST}")


if __name__ == "__main__":
    main()
