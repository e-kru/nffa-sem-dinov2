"""Strict group manifest validation. Test records are checked, never opened."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.downstream.data import load_rows
from .geometry import content_box, regions


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def resolve(manifest, value):
    return (Path(manifest).parent / value).resolve()


def manifest_rows(manifest, limit=None):
    rows = load_rows(manifest)
    required = {"image_id", "image_path", "mask_path", "group_id", "split"}
    if not required <= rows[0].keys():
        raise ValueError(f"Manifest requires {sorted(required)}")
    groups, images, masks = {}, set(), set()
    for row in rows:
        if any(not row[k] for k in required):
            raise ValueError("Empty identity/group/path in manifest")
        group, split = row["group_id"], row["split"]
        if group in groups and groups[group] != split:
            raise ValueError(f"Group leakage: {group}")
        groups[group] = split
        for key, seen in (("image_path", images), ("mask_path", masks)):
            path = resolve(manifest, row[key])
            if path in seen:
                raise ValueError(f"Duplicate {key}: {path}")
            seen.add(path)
    selected = []
    for split in ("train", "val"):
        part = [r for r in rows if r["split"] == split]
        if not part:
            raise ValueError(f"Empty {split} split")
        selected.extend(part if limit is None else part[:limit])
    return selected


def load_mask(manifest, row):
    """Load native labels only; cached training never decodes original RGB."""
    if row["split"] not in ("train", "val"):
        raise ValueError("Test data are forbidden in this pipeline")
    with Image.open(resolve(manifest, row["mask_path"])) as handle:
        if handle.mode not in ("L", "P"):
            raise ValueError("Mask must contain indexed labels (L/P)")
        labels = np.array(handle, dtype=np.int64, copy=True)
    expected = row.get("original_size")
    if expected is not None and labels.shape != (expected[1], expected[0]):
        raise ValueError(f"Image/mask mismatch: {row['image_id']}")
    if not np.isin(labels, (0, 1, 255)).all():
        raise ValueError(f"Invalid labels: {row['image_id']}")
    if not np.any(labels != 255):
        raise ValueError(f"All-ignore original: {row['image_id']}")
    return torch.from_numpy(labels)


def load_original(manifest, row):
    if row["split"] not in ("train", "val"):
        raise ValueError("Test data are forbidden in this pipeline")
    with Image.open(resolve(manifest, row["image_path"])) as handle:
        image = handle.convert("RGB")
    labels = load_mask(manifest, row)
    if tuple(labels.shape) != (image.height, image.width):
        raise ValueError(f"Image/mask mismatch: {row['image_id']}")
    return image, labels


def inspect_data(manifest, rows, variants):
    records, sizes, padded = [], Counter(), Counter()
    for row in rows:
        image, _ = load_original(manifest, row)
        sizes[f"{image.width}x{image.height}"] += 1
        record = dict(row, original_size=list(image.size),
                      image_sha256=sha256_file(resolve(manifest, row["image_path"])),
                      mask_sha256=sha256_file(resolve(manifest, row["mask_path"])))
        records.append(record)
        for v in variants:
            for x0, y0, x1, y1 in regions(*image.size, v.quadrants):
                if content_box(x1-x0, y1-y0, v) != (0, 0, v.width, v.height):
                    padded[v.name] += 1
    return records, {"counts": dict(Counter(r["split"] for r in rows)),
                     "original_sizes": dict(sizes), "letterboxed_regions": dict(padded),
                     "test_files_opened": False,
                     "historical_baseline_comparable": False}
