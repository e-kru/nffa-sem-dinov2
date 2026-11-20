"""Train-only manifest selection and Meta's original augmentation/masking."""

import csv
from functools import partial
import hashlib
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from src.preprocessing_v2 import crop_metadata_strip


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def train_records(manifest: Path, split: Path, holdout_splits=()) -> list[dict]:
    """Honor curated inclusion and a supplied group-aware split, never labels."""
    with manifest.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    with split.open(newline="") as stream:
        splits = list(csv.DictReader(stream))
    if not rows or not splits:
        raise ValueError("Empty manifest or split")
    for records in (rows, splits):
        ids = [r["image_id"] for r in records]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate image_id in manifest or split")
    if any(r["split"] not in {"train", "val", "test"} for r in splits):
        raise ValueError("Unknown split name")
    # A conservative group crossing any partition is an error, including groups
    # spanning validation/test: the split must remain a valid group-aware split.
    group_splits = {}
    for row in splits:
        group = row.get("group_id") or row.get("exact_group_id")
        if not group:
            raise ValueError("Split requires a nonempty group_id or exact_group_id")
        previous = group_splits.setdefault(group, row["split"])
        if previous != row["split"]:
            raise ValueError(f"Group crosses partitions: {group}")
    by_id = {r["image_id"]: r for r in rows}
    if any(r["image_id"] not in by_id for r in splits):
        raise ValueError("Split contains unknown manifest image IDs")
    selected = []
    for assignment in splits:
        row = by_id[assignment["image_id"]]
        if assignment["split"] == "train" and row["included"] == "1":
            if row.get("crop_mode", "crop_v2") != "crop_v2":
                raise ValueError("Pilot supports only curated crop_v2 images")
            selected.append({"image_id": row["image_id"],
                             "group_id": assignment.get("group_id") or assignment["exact_group_id"]})
    if not selected:
        raise ValueError("No included training images")
    # Retain whole primary groups only when all selected members are also train
    # in every additional evaluation protocol (e.g. split A as well as split B).
    for holdout in holdout_splits:
        allowed = {r["image_id"] for r in train_records(manifest, holdout)}
        blocked = {r["group_id"] for r in selected if r["image_id"] not in allowed}
        selected = [r for r in selected if r["group_id"] not in blocked]
    if not selected:
        raise ValueError("No training groups survive all held-out partitions")
    return selected


class SEMTrainDataset(Dataset):
    def __init__(self, root: Path, records: list[dict], transform, *, validate_paths=True):
        self.root = root.resolve()
        self.records = records
        self.transform = transform
        self.profile = False
        self.decoded_cache = None
        for row in records if validate_paths else ():
            path = (self.root / row["image_id"]).resolve()
            if not path.is_relative_to(self.root) or not path.is_file():
                raise ValueError(f"Missing image or path outside root: {row['image_id']}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        from src.ssl.loading import load_image
        import time
        timings = {} if self.profile else None
        if self.decoded_cache is None:
            image = load_image(self.root / self.records[index]['image_id'], timings)
        else:
            begun = time.perf_counter()
            image = self.decoded_cache.image(self.records[index]['image_id'])
            if timings is not None:
                timings['cache_fetch_seconds'] = time.perf_counter() - begun
        result = self.transform(image)
        if timings is not None:
            result.setdefault('_loader_profile', {}).update(timings)
        return result, ()


def augmentation_and_collate(cfg):
    # activate() must run before this import.
    from dinov2.data import DataAugmentationDINO, MaskingGenerator, collate_data_and_cast
    crops = cfg.crops
    transform = DataAugmentationDINO(
        crops.global_crops_scale, crops.local_crops_scale, crops.local_crops_number,
        global_crops_size=crops.global_crops_size, local_crops_size=crops.local_crops_size,
    )
    grid = crops.global_crops_size // cfg.student.patch_size
    generator = MaskingGenerator((grid, grid), max_num_patches=int(0.5 * grid * grid))
    collate = partial(
        collate_data_and_cast, mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
        mask_probability=cfg.ibot.mask_sample_probability, dtype=torch.float32,
        n_tokens=grid * grid, mask_generator=generator,
    )
    return transform, collate


def seed_worker(_worker_id):
    torch.set_num_threads(1)
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
