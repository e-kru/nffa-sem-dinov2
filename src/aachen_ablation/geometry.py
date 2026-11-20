"""Lossless original regions and reversible, aspect-preserving image geometry."""
from dataclasses import asdict, dataclass

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from src.preprocessing_v2 import IMAGENET_MEAN, IMAGENET_STD

PREPROCESSING_VERSION = "aachen-original-mask-letterbox-v1"


@dataclass(frozen=True)
class Variant:
    name: str
    width: int
    height: int
    quadrants: bool = False

    @property
    def grid(self):
        return self.height // 14, self.width // 14

    @property
    def tokens(self):
        return np.prod(self.grid).item() * (4 if self.quadrants else 1)

    def config(self):
        return asdict(self)


VARIANTS = {
    "A1": Variant("A1", 448, 336),
    "A2": Variant("A2", 672, 504),
    "A3": Variant("A3", 896, 672),
    "B1": Variant("B1", 448, 336, True),
}


def regions(width, height, quadrants):
    if width < 2 or height < 2:
        raise ValueError("Original dimensions must be at least 2x2")
    if not quadrants:
        return [(0, 0, width, height)]
    x, y = width // 2, height // 2
    return [(0, 0, x, y), (x, 0, width, y),
            (0, y, x, height), (x, y, width, height)]


def content_box(width, height, variant):
    """Integer letterbox (PIL x0,y0,x1,y1); at most one pixel rounding error."""
    scale = min(variant.width / width, variant.height / height)
    w = min(variant.width, max(1, round(width * scale)))
    h = min(variant.height, max(1, round(height * scale)))
    x, y = (variant.width - w) // 2, (variant.height - h) // 2
    return x, y, x + w, y + h


def prepare_region(image, box, variant):
    crop = image.crop(box).convert("RGB")
    inner = content_box(*crop.size, variant)
    x0, y0, x1, y1 = inner
    canvas = Image.new("RGB", (variant.width, variant.height), (0, 0, 0))
    canvas.paste(crop.resize((x1-x0, y1-y0), Image.Resampling.BILINEAR), (x0, y0))
    pixels = torch.from_numpy(np.array(canvas, copy=True)).permute(2, 0, 1).float() / 255
    mean = torch.tensor(IMAGENET_MEAN)[:, None, None]
    std = torch.tensor(IMAGENET_STD)[:, None, None]
    return (pixels - mean) / std


def project_logits(logits, size, variant):
    """Grid logits C,h,w -> ORIGINAL region C,H,W, before softmax/threshold."""
    width, height = size
    box = content_box(width, height, variant)
    z = logits.unsqueeze(0)
    if box != (0, 0, variant.width, variant.height):
        z = F.interpolate(z, (variant.height, variant.width), mode="bilinear", align_corners=False)
        x0, y0, x1, y1 = box
        z = z[:, :, y0:y1, x0:x1]
    return F.interpolate(z, (height, width), mode="bilinear", align_corners=False)[0]


def reconstruct(logits, size, variant):
    width, height = size
    boxes = regions(width, height, variant.quadrants)
    if len(logits) != len(boxes):
        raise ValueError("Wrong number of crops")
    out = logits.new_empty((2, height, width))
    for z, (x0, y0, x1, y1) in zip(logits, boxes):
        out[:, y0:y1, x0:x1] = project_logits(z, (x1-x0, y1-y0), variant)
    return out
