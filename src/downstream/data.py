"""Data loading for cached classification and image-based downstream tasks."""

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from src.dinov2_encoder import get_dinov2_transform
from src.preprocessing_v2 import preprocess_segmentation_pair



def load_rows(manifest: Path) -> list[dict[str, str]]:
    with Path(manifest).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "split" not in rows[0]:
        raise ValueError(f"Empty or invalid manifest: {manifest}")
    splits = [r["split"] for r in rows]
    if set(splits) - {"train", "val", "test"}:
        raise ValueError(f"Invalid split values: {set(splits)}")
    if "image_id" in rows[0]:
        ids = [r["image_id"] for r in rows]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate image_id entries in split manifest")
    return rows


def load_feature_cache(feature_dir: Path) -> dict[str, np.ndarray]:
    files = sorted(Path(feature_dir).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files in {feature_dir}")
    mapping = {}
    for file in files:
        with np.load(file, allow_pickle=True) as data:
            for path, vector in zip(data["paths"], data["X"]):
                image_id = str(path)
                if "/nffa_sem_100/" in image_id:
                    image_id = image_id.split("/nffa_sem_100/", 1)[1]
                if image_id in mapping:
                    raise ValueError(f"Duplicate cached image: {image_id}")
                mapping[image_id] = np.asarray(vector, dtype=np.float32)
    return mapping


class CachedClassificationDataset(Dataset):
    def __init__(self, rows: list[dict], features: dict[str, np.ndarray],
                 split: str, scaler: StandardScaler):
        selected = [r for r in rows if r["split"] == split]
        if not selected:
            raise ValueError(f"No samples for split {split}")
        missing = [r["image_id"] for r in selected if r["image_id"] not in features]
        if missing:
            raise ValueError(f"Missing features, e.g. {missing[:3]}")
        X = np.stack([features[r["image_id"]] for r in selected])
        y = np.array([int(r["label"]) for r in selected], dtype=np.int64)
        self.X = torch.from_numpy(scaler.transform(X).astype(np.float32))
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def make_cached_classification(rows: list[dict], feature_dir: Path):
    features = load_feature_cache(feature_dir)
    selected = [r for r in rows if r["split"] == "train"]
    if not selected:
        raise ValueError("Training partition is empty")
    missing = [r["image_id"] for r in selected if r["image_id"] not in features]
    if missing:
        raise ValueError(f"Missing train features, e.g. {missing[:3]}")
    scaler = StandardScaler().fit(
        np.stack([features[r["image_id"]] for r in selected]))
    train = CachedClassificationDataset(rows, features, "train", scaler)
    val = CachedClassificationDataset(rows, features, "val", scaler)
    return train, val, scaler


class ClassificationImageDataset(Dataset):
    def __init__(self, rows: list[dict], split: str, image_root: Path):
        self.rows = [r for r in rows if r["split"] == split]
        if not self.rows:
            raise ValueError(f"No {split} samples")
        self.image_root = Path(image_root)
        self.transform = get_dinov2_transform()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        with Image.open(self.image_root / row["image_id"]) as im:
            image = im.convert("RGB")
        return self.transform(image), int(row["label"])


class SegmentationDataset(Dataset):
    """CSV with columns image_path,mask_path,split; 0..K-1 or ignore_index pixels.

    Relative paths are resolved relative to the CSV location.
    Pixel masks MUST use palette ('P') or single channel ('L') images,
    not RGB color-coded masks, and must align to uncropped raw images.
    """

    def __init__(self, manifest: Path, split: str, ignore_index: int = 255):
        self.manifest = Path(manifest)
        rows = load_rows(manifest)
        if not {"image_path", "mask_path"}.issubset(rows[0]):
            raise ValueError("Segmentation CSV needs image_path,mask_path,split")
        identifiers = [r["image_path"] for r in rows]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Duplicate image_path entries in segmentation manifest; possible leakage")
        self.rows = [r for r in rows if r["split"] == split]
        if not self.rows:
            raise ValueError(f"No segmentation samples for split {split}")
        self.ignore_index = ignore_index

    def __len__(self):
        return len(self.rows)

    def _resolve(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.manifest.parent / path

    def __getitem__(self, i):
        row = self.rows[i]
        with Image.open(self._resolve(row["image_path"])) as file:
            image = file.convert("RGB")
        with Image.open(self._resolve(row["mask_path"])) as file:
            if file.mode not in {"L", "P"}:
                raise ValueError(f"Mask must be label indices, not RGB: {file.mode}")
            mask = file.copy()
        if image.size != mask.size:
            raise ValueError(f"Image/mask sizes differ: {row['image_path']}")

        return preprocess_segmentation_pair(image, mask)
