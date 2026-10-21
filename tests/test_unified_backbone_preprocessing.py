"""Regression tests: canonical preprocessing vs historical V2 extraction."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

from src.preprocessing_v2 import (
    apply_sem_geometry, crop_metadata_strip, preprocess_segmentation_pair,
)
from src.dinov2_encoder import (
    DINO_DIMS, get_dinov2, get_dinov2_transform,
    load_dinov2_checkpoint,
)
from src.downstream.data import ClassificationImageDataset
from src.downstream.model import DINOv2Backbone


class DummyDINO(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(384))

    def forward(self, x):
        return x.mean(dim=(1, 2, 3)).unsqueeze(1) * self.weight


class UnificationTest(unittest.TestCase):
    def _images(self):
        rng = np.random.default_rng(31)
        for width, height, add_strip in [(310, 230, False), (1024, 768, True),
                                         (280, 360, False)]:
            arr = rng.integers(0, 155, size=(height, width, 3), dtype=np.uint8)
            if add_strip:
                arr[int(.88 * height):] = 250
            yield Image.fromarray(arr)

    def test_exact_same_transform_as_historical_feature_extraction(self):
        old = T.Compose([
            T.Lambda(crop_metadata_strip),
            T.Resize(256), T.CenterCrop(224), T.ToTensor(),
            T.Normalize((.485, .456, .406), (.229, .224, .225)),
        ])
        unified = get_dinov2_transform()
        for image in self._images():
            with self.subTest(size=image.size):
                torch.testing.assert_close(unified(image), old(image), rtol=0, atol=0)

    def test_segmentation_image_uses_identical_geometry_and_mask_labels(self):
        for image in self._images():
            mask = Image.fromarray(np.pad(
                np.ones((image.height, image.width // 2), dtype=np.uint8),
                ((0, 0), (0, image.width - image.width // 2)),
                constant_values=255,
            ))
            tensor, label = preprocess_segmentation_pair(image, mask)
            torch.testing.assert_close(tensor, get_dinov2_transform()(image), rtol=0, atol=0)
            self.assertEqual(tuple(label.shape), (224, 224))
            self.assertTrue(set(label.unique().tolist()).issubset({1, 255}))
            geom_im, geom_mask = apply_sem_geometry(image, mask)
            expected_mask = TF.center_crop(TF.resize(
                mask if crop_metadata_strip(image).size == image.size else
                mask.crop((0, 0, image.width, crop_metadata_strip(image).height)),
                256, interpolation=InterpolationMode.NEAREST), 224)
            np.testing.assert_array_equal(np.asarray(geom_mask), np.asarray(expected_mask))

    def test_legacy_loader_and_downstream_load_same_backbone(self):
        with patch("torch.hub.load", side_effect=lambda *args: DummyDINO()) as loader:
            old_model = get_dinov2()
            new_model = DINOv2Backbone()
        self.assertEqual(loader.call_count, 2)
        self.assertEqual(loader.call_args.args, ("facebookresearch/dinov2", "dinov2_vits14"))
        self.assertFalse(any(p.requires_grad for p in old_model.parameters()))
        self.assertFalse(any(p.requires_grad for p in new_model.parameters()))
        self.assertEqual(DINO_DIMS["dinov2_vits14"], 384)

    def test_checkpoint_loading_for_both_entrypoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "backbone.pt"
            expected = DummyDINO()
            expected.weight.data.fill_(2.5)
            torch.save({"encoder_state_dict": expected.state_dict()}, ckpt)
            direct = load_dinov2_checkpoint(DummyDINO(), ckpt)
            torch.testing.assert_close(direct.weight, expected.weight)
            with patch("torch.hub.load", return_value=DummyDINO()):
                model = DINOv2Backbone(checkpoint=ckpt)
            torch.testing.assert_close(model.model.weight, expected.weight)


if __name__ == "__main__":
    unittest.main()
