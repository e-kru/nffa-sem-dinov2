"""Non-destructive 448x336 no-padding segmentation smoke test.

Run in repository:
    python -m scripts.test_aachen_rect
Optional local/mock test:
    python -m scripts.test_aachen_rect --mock
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from src.downstream.model import DINOv2Backbone, DownstreamModel
from src.preprocessing_v2 import IMAGENET_MEAN, IMAGENET_STD

from src.config import AACHEN_ROOT

HEIGHT, WIDTH = 336, 448


def get_pair(root: Path, name: str | None):
    if root.exists() and (root / "images").exists():
        image_files = sorted((root / "images").glob("*.png"))
        if image_files:
            image_file = root / "images" / name if name else image_files[0]
            mask_file = root / "masks" / image_file.name
            with Image.open(image_file) as f:
                image = f.convert("RGB")
            with Image.open(mask_file) as f:
                mask = f.convert("L")
            return image_file.name, image, mask

    # Synthetic 4:3 image used only when dataset isn't on this machine.
    image = Image.new("RGB", (1024, 768), (90, 100, 110))
    mask = Image.new("L", (1024, 768), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle((4, 15, 95, 150), fill=1)
    draw.ellipse((865, 600, 1015, 745), fill=1)
    return "SYNTHETIC", image, mask


class FakeDINO(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 384)

    def get_intermediate_layers(self, images, n=1, reshape=True, norm=True):
        pooled = torch.nn.functional.avg_pool2d(images, kernel_size=14)
        return (torch.einsum("bchw,dc->bdhw", pooled, self.fc.weight),)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="Avoid loading pretrained DINOv2 weights")
    parser.add_argument("--root", type=Path, default=AACHEN_ROOT)
    parser.add_argument("--image-name", default=None)
    args = parser.parse_args()

    name, image, mask = get_pair(args.root, args.image_name)
    assert image.size == mask.size
    assert image.size == (1024, 768), f"Unexpected source size: {image.size}"
    assert (WIDTH % 14 == HEIGHT % 14 == 0)
    assert abs(image.width / image.height - WIDTH / HEIGHT) < 1e-8

    # Both are resized on exactly the same geometry; NO metadata crop or center crop.
    image_small = TF.resize(image, [HEIGHT, WIDTH], interpolation=InterpolationMode.BILINEAR)
    mask_small = TF.resize(mask, [HEIGHT, WIDTH], interpolation=InterpolationMode.NEAREST)
    values = np.unique(np.asarray(mask_small))
    assert set(values.tolist()) <= {0, 1}, f"Unexpected mask labels: {values}"
    assert np.count_nonzero(np.asarray(mask_small)) > 0
    x = TF.normalize(TF.to_tensor(image_small), IMAGENET_MEAN, IMAGENET_STD).unsqueeze(0)
    y = torch.from_numpy(np.array(mask_small, dtype=np.int64, copy=True)).unsqueeze(0)
    assert x.shape == (1, 3, 336, 448), x.shape
    assert y.shape == (1, 336, 448), y.shape

    print("Image:", name)
    print("Original / Resized:", image.size, "/", image_small.size)
    print("Tensor image/mask:", tuple(x.shape), tuple(y.shape))
    print("Mask unique values:", values.tolist())
    print("Positive pixels before/after:", int(np.count_nonzero(np.asarray(mask))),
          int(np.count_nonzero(np.asarray(mask_small))))

    if args.mock:
        backbone = DINOv2Backbone(model=FakeDINO(), frozen=True)
        print("Backbone: FakeDINO (interface test, not real model weights)")
    else:
        backbone = DINOv2Backbone(name="dinov2_vits14", frozen=True)
        print("Backbone: real pretrained dinov2_vits14")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DownstreamModel(backbone, "segmentation", "linear", 2).to(device)
    model.train()
    x, y = x.to(device), y.to(device)
    with torch.no_grad():
        patch_features = model.backbone.forward_dense(x)
    print("Patch feature grid:", tuple(patch_features.shape))
    assert patch_features.shape == (1, 384, 24, 32), patch_features.shape

    prediction = model(x)
    print("Pixel logits:", tuple(prediction.shape))
    assert prediction.shape == (1, 2, 336, 448), prediction.shape
    loss = torch.nn.functional.cross_entropy(prediction, y)
    assert torch.isfinite(loss).item()
    loss.backward()
    assert model.head.net.weight.grad is not None
    print("Cross-entropy:", round(float(loss.detach()), 6))
    print("Head gradients:", model.head.net.weight.grad is not None)
    print("GPU:", device)
    print("RESULT: PASS (448x336, no crop, no padding)")


if __name__ == "__main__":
    main()
