"""Canonical SEM preprocessing (V2) for feature extraction and downstream tasks.

This is the ONLY active SEM preprocessing implementation. Never apply
metadata cropping twice. Keep image/mask geometry aligned.
"""

import numpy as np
from PIL import Image

SEM_RESIZE_SHORT_SIDE = 256
SEM_INPUT_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def detect_metadata_start(image):
    """Detect the upper boundary of the white SEM metadata strip (V2)."""
    width, height = image.size
    if height < 80:
        return None

    small = image.convert("RGB").resize(
        (min(width, 512), height), Image.Resampling.BILINEAR
    )
    pixels = np.asarray(small)
    white_fraction = (np.min(pixels, axis=2) >= 230).mean(axis=1)

    window = max(4, round(0.007 * height))
    band_height = max(8, round(0.014 * height))

    for y in range(int(0.75 * height), height - band_height):
        above = white_fraction[y - window:y].mean()
        below = white_fraction[y:y + window].mean()
        band = white_fraction[y:y + band_height]
        if above > 0.45:
            continue
        if below < 0.75:
            continue
        if below - above < 0.50:
            continue
        if (band >= 0.65).mean() < 0.65:
            continue
        return max(1, y - 2)
    return None


def crop_metadata_strip(image):
    """Original V2 crop; its decision rule is kept unchanged."""
    crop_y = detect_metadata_start(image)
    return image if crop_y is None else image.crop((0, 0, image.width, crop_y))


def apply_sem_geometry(image, mask=None):
    """Run the shared V2 crop, resize, center-crop on image and optional mask.

    Images use bilinear resizing; discrete label masks use nearest-neighbor.
    Returns PIL image, PIL mask (or None). This is the single place defining
    the downstream spatial preprocessing.
    """
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as TF

    if mask is not None and image.size != mask.size:
        raise ValueError(f"Image/mask dimensions mismatch: {image.size}, {mask.size}")

    crop_y = detect_metadata_start(image)
    if crop_y is not None:
        box = (0, 0, image.width, crop_y)
        image = image.crop(box)
        if mask is not None:
            mask = mask.crop(box)

    image = TF.resize(image, SEM_RESIZE_SHORT_SIDE,
                      interpolation=InterpolationMode.BILINEAR)
    image = TF.center_crop(image, SEM_INPUT_SIZE)
    if mask is not None:
        mask = TF.resize(mask, SEM_RESIZE_SHORT_SIDE,
                         interpolation=InterpolationMode.NEAREST)
        mask = TF.center_crop(mask, SEM_INPUT_SIZE)
    return image, mask


def preprocess_sem_image(image):
    """PIL image -> normalized DINOv2 tensor, through canonical V2 geometry."""
    from torchvision.transforms import functional as TF
    image, _ = apply_sem_geometry(image)
    return TF.normalize(TF.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)


def preprocess_segmentation_pair(image, mask):
    """Aligned PIL image and label-mask -> image tensor, integer label tensor.

    Mask pixels should be 0..C-1, optionally 255 for ignored pixels.
    """
    import torch
    from torchvision.transforms import functional as TF
    image, mask = apply_sem_geometry(image, mask)
    return (TF.normalize(TF.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD),
            torch.from_numpy(np.array(mask, dtype=np.int64, copy=True)))
