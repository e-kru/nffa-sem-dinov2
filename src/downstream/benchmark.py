"""Versioned train/validation-only feature caches for frozen encoder benchmarks."""
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

from src.aachen_ablation.cache import atomic_json, encoder_fingerprint, make_pinned_encoder
from src.aachen_ablation.data import sha256_file
from .data import ClassificationImageDataset, load_rows

ROOT = Path(__file__).resolve().parents[2]


def sources():
    paths = [*sorted((ROOT / "src/downstream").glob("*.py")),
             ROOT / "src/aachen_ablation/cache.py",
             ROOT / "src/dinov2_encoder.py", ROOT / "src/preprocessing_v2.py",
             ROOT / "scripts/evaluate_sem_encoder.py", ROOT / "scripts/verify_frozen_encoder.py"]
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in paths}


def classification_records(manifest, split_file, image_root, workers=8, smoke=False, *, progress=False):
    """Read partition identity for all rows; interpret labels/open files only in train/val."""
    rows = load_rows(split_file)
    groups = {}
    for row in rows:
        group = row.get("group_id")
        if not group:
            raise ValueError("Missing classification group_id")
        if group in groups and groups[group] != row["split"]:
            raise ValueError(f"Group leakage: {group}")
        groups[group] = row["split"]
    with Path(manifest).open(newline="") as handle:
        inventory = list(csv.DictReader(handle))
    indexed = {r["image_id"]: r for r in inventory}
    if len(indexed) != len(inventory):
        raise ValueError("Duplicate image IDs in classification manifest")
    root, selected = Path(image_root).resolve(), []
    for partition in ("train", "val"):
        part = [r for r in rows if r["split"] == partition]
        if not part:
            raise ValueError(f"Empty classification {partition}")
        if partition == "train" and len(part) < 2:
            raise ValueError("Classification training requires at least two samples")
        chosen = part[:8 if partition == "train" else 4] if smoke else part
        for row in chosen:
            item = indexed.get(row["image_id"])
            if item is None or item["included"] != "1":
                raise ValueError(f"Split/manifest mismatch: {row['image_id']}")
            if int(row["label"]) != int(item["label"]) or not 0 <= int(row["label"]) < 10:
                raise ValueError(f"Classification label mismatch: {row['image_id']}")
            selected.append({"image_id": row["image_id"], "label": row["label"],
                             "group_id": row["group_id"], "split": partition})
    def inspect_record(row):
        # Resolve symlinks, inspect and hash in workers: metadata syscalls on the
        # shared filesystem are expensive too. Never submit test rows.
        path = (root / row["image_id"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing/outside-root training/validation image: {path}")
        return dict(row, image_sha256=sha256_file(path))
    records = []
    if progress:
        print(f"Inspecting {len(selected)} train/val images with {max(1, workers)} workers: {split_file}", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for record in pool.map(inspect_record, selected):
            records.append(record)
            if progress and (len(records) % 2000 == 0 or len(records) == len(selected)):
                print(f"Train/val file fingerprints: {len(records)}/{len(selected)}", flush=True)
    return records


def feature_contract(records, manifest, split_file, image_root, identity, smoke):
    return {"schema": "sem-classification-cache-v1", "encoder": identity,
            "feature_shape": [identity.get("feature_dim", 384)], "dtype": "float32", "records": records,
            "manifest_sha256": sha256_file(manifest), "split_sha256": sha256_file(split_file),
            "image_root": str(Path(image_root).resolve()),
            "preprocessing": "original V2 crop -> Resize(256) -> CenterCrop(224) -> ImageNet normalization",
            "sources": sources(), "nonscientific_smoke_test": smoke}


def validate_features(path, contract, complete=True):
    path = Path(path)
    meta = json.loads((path / "cache_metadata.json").read_text())
    if meta["contract"] != contract:
        raise ValueError(f"Classification cache identity mismatch: {path}")
    if complete and not meta["complete"]:
        raise ValueError("Classification cache incomplete; resume extraction")
    count = 0
    for chunk in meta["chunks"]:
        file = path / chunk["file"]
        if file.name != f"chunk_{count:06d}_{count+chunk['count']:06d}.npz":
            raise ValueError("Invalid classification cache chunk order")
        with np.load(file, allow_pickle=False) as data:
            expected = contract["records"][count:count+chunk["count"]]
            if (data["X"].shape != (len(expected), contract["feature_shape"][0]) or data["X"].dtype != np.float32
                    or not np.isfinite(data["X"]).all()
                    or data["paths"].tolist() != [r["image_id"] for r in expected]
                    or data["y"].tolist() != [int(r["label"]) for r in expected]
                    or sha256_file(file) != chunk["sha256"]):
                raise ValueError(f"Corrupt/misaligned classification chunk: {file}")
        count += chunk["count"]
    if count > len(contract["records"]) or (meta["complete"] and count != len(contract["records"])):
        raise ValueError("Classification cache count mismatch")
    return meta


class SyntheticGlobal(torch.nn.Module):
    """Deterministic test double; no model weights or scientific claims."""
    def forward_global(self, x):
        if self.training or not torch.is_inference_mode_enabled():
            raise RuntimeError("Frozen eval/inference mode required")
        return x.mean((2, 3)).repeat(1, 128)


def extract_features(path, contract, checkpoint, device, workers=8, resume=False, smoke=False):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    metadata = path / "cache_metadata.json"
    if metadata.exists():
        meta = validate_features(path, contract, complete=False)
        if meta["complete"]:
            return meta
        if not resume:
            raise FileExistsError(f"Incomplete classification cache; use --resume: {path}")
    else:
        if any(path.iterdir()):
            raise FileExistsError(f"Unrecognized classification cache: {path}")
        meta = {"contract": contract, "complete": False, "chunks": []}
        atomic_json(metadata, meta)
    encoder = SyntheticGlobal().eval() if smoke else make_pinned_encoder(device, checkpoint, name=contract["encoder"].get("model", "dinov2_vits14"))
    encoder.eval().requires_grad_(False).to(device)
    fingerprint = encoder_fingerprint(encoder)
    if meta.get("encoder_state_sha256", fingerprint) != fingerprint:
        raise ValueError("Frozen encoder changed during extraction resume")
    meta["encoder_state_sha256"] = fingerprint
    atomic_json(metadata, meta)
    rows = contract["records"]
    dataset = ConcatDataset([ClassificationImageDataset(rows, s, Path(contract["image_root"]))
                             for s in ("train", "val")])
    start = sum(c["count"] for c in meta["chunks"])
    for begin in range(start, len(dataset), 512):
        end = min(begin+512, len(dataset))
        file = path / f"chunk_{begin:06d}_{end:06d}.npz"
        if file.exists():
            # A crash may have published a chunk before its metadata; never replace it.
            raise FileExistsError(f"Unindexed chunk requires inspection: {file}")
        loader = DataLoader(Subset(dataset, range(begin, end)), batch_size=64,
                            shuffle=False, num_workers=workers, pin_memory=device.type == "cuda")
        values = []
        with torch.inference_mode():
            for images, _ in loader:
                features = encoder.forward_global(images.to(device, non_blocking=True))
                if features.shape != (len(images), contract["feature_shape"][0]) or not torch.isfinite(features).all():
                    raise ValueError("Invalid global DINOv2 features")
                values.append(features.cpu().numpy().astype(np.float32))
        temp = file.with_suffix(".partial")
        with temp.open("wb") as handle:
            np.savez(handle, X=np.concatenate(values),
                     paths=np.asarray([r["image_id"] for r in rows[begin:end]]),
                     y=np.asarray([int(r["label"]) for r in rows[begin:end]]))
        os.replace(temp, file)
        meta["chunks"].append({"file": file.name, "count": end-begin, "sha256": sha256_file(file)})
        atomic_json(metadata, meta)
        print(f"Classification features: {end}/{len(dataset)}", flush=True)
    meta["complete"] = True
    atomic_json(metadata, meta)
    return validate_features(path, contract)


def arrays(path, contract):
    """Keep baseline float32 ordering; validate all chunks before reading any features."""
    meta = validate_features(path, contract)
    values = []
    for chunk in meta["chunks"]:
        with np.load(Path(path) / chunk["file"], allow_pickle=False) as data:
            values.append(data["X"])
    X = np.concatenate(values)
    records = contract["records"]
    train = np.array([r["split"] == "train" for r in records])
    y = np.array([int(r["label"]) for r in records], dtype=np.int64)
    return X[train], y[train], X[~train], y[~train]
