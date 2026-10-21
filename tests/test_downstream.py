"""Unit and integration smoke tests. No DINOv2 download needed."""

import csv
import tempfile
import sys
from unittest.mock import patch
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from src.downstream.data import SegmentationDataset, load_rows, make_cached_classification
from src.downstream.engine import train
from src.downstream.heads import make_head
from src.downstream.metrics import PixelConfusion
from src.downstream.model import DINOv2Backbone, DownstreamModel


class FakeDINO(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 384)

    def forward(self, images):
        return self.fc(images.mean(dim=(2, 3)))

    def get_intermediate_layers(self, images, n=1, reshape=True, norm=True):
        x = nn.functional.avg_pool2d(images, kernel_size=14)
        # deterministic feature map B,384,H/14,W/14
        return (torch.einsum("bchw,dc->bdhw", x, self.fc.weight),)


class DownstreamTest(unittest.TestCase):
    def test_head_shapes_and_gradients(self):
        x = torch.randn(2, 384)
        maps = torch.randn(2, 384, 16, 16)
        for kind in ("linear", "mlp"):
            cls = make_head("classification", kind, 384, 10)
            seg = make_head("segmentation", kind, 384, 3, (32, 16), .2)
            self.assertEqual(cls(x).shape, (2, 10))
            self.assertEqual(seg(maps).shape, (2, 3, 16, 16))

    def test_backbone_modes(self):
        encoder = DINOv2Backbone(model=FakeDINO(), frozen=True)
        classifier = DownstreamModel(encoder, "classification", "mlp", 10)
        classifier.train()
        images = torch.randn(2, 3, 224, 224)
        output = classifier(images)
        self.assertEqual(output.shape, (2, 10))
        output.sum().backward()
        self.assertIsNone(encoder.model.fc.weight.grad)
        encoder.set_frozen(False)
        classifier.zero_grad(set_to_none=True)
        classifier(images).sum().backward()
        self.assertIsNotNone(encoder.model.fc.weight.grad)
        model = DownstreamModel(encoder, "segmentation", "linear", 3)
        self.assertEqual(model(images).shape, (2, 3, 224, 224))

    def test_segmentation_masks_and_metric(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = Image.fromarray(np.full((230, 310, 3), 70, dtype=np.uint8))
            mask_arr = np.zeros((230, 310), dtype=np.uint8)
            mask_arr[:, 155:] = 1
            image.save(root / "img.png")
            Image.fromarray(mask_arr).save(root / "mask.png")
            with (root / "m.csv").open("w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["image_path", "mask_path", "split"])
                writer.writerow(["img.png", "mask.png", "train"])
            dataset = SegmentationDataset(root / "m.csv", "train")
            x, y = dataset[0]
            self.assertEqual(x.shape, (3, 224, 224))
            self.assertEqual(y.shape, (224, 224))
            self.assertEqual(set(y.unique().tolist()), {0, 1})
            m = PixelConfusion(2)
            labels = torch.tensor([[[0, 1], [255, 1]]])
            logits = torch.tensor([[[[9., 0.], [1., 0.]],
                                   [[0., 9.], [0., 9.]]]])
            m.update(logits, labels)
            self.assertEqual(m.compute()["miou"], 1.0)

    def test_cached_split_and_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            features = root / "features"
            features.mkdir()
            rng = np.random.default_rng(42)
            xs = rng.normal(size=(28, 384)).astype(np.float32)
            labels = (xs[:, 0] > 0).astype(int)
            ids = [f"class{labels[i]}/{i}.jpg" for i in range(len(xs))]
            np.savez(features / "chunk.npz", X=xs, paths=np.array(ids))
            rows = [dict(image_id=ids[i], label=str(labels[i]),
                         split="train" if i < 20 else "val") for i in range(len(xs))]
            train_ds, val_ds, scaler = make_cached_classification(rows, features)
            self.assertEqual(len(train_ds), 20)
            self.assertEqual(len(val_ds), 8)
            self.assertTrue(np.allclose(scaler.mean_, xs[:20].mean(axis=0), atol=1e-6))
            model = make_head("classification", "mlp", 384, 2, (16, 8, 4), 0.0)
            summary = train(model, train_ds, val_ds, output=root / "out",
                            task="classification", num_classes=2,
                            epochs=2, patience=2, batch_size=8)
            self.assertGreaterEqual(summary["best_val_score"], 0)
            self.assertTrue((root / "out" / "best_model.pt").exists())
            self.assertFalse(summary["test_evaluated"])


    def test_cli_cached_and_segmentation(self):
        from scripts.run_downstream import main
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cached = root / "cache"
            cached.mkdir()
            rng = np.random.default_rng(37)
            X = rng.normal(size=(20, 384)).astype(np.float32)
            y = (X[:, 0] > 0).astype(int)
            ids = [f"class{y[i]}/{i}.png" for i in range(20)]
            np.savez(cached / "f.npz", paths=np.array(ids), X=X)
            manifest = root / "classes.csv"
            with manifest.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["image_id", "label", "split"])
                for i in range(20):
                    w.writerow([ids[i], y[i], "train" if i < 16 else "val"])
            cmd = ["--task", "classification", "--head", "linear",
                   "--mode", "cached", "--manifest", str(manifest),
                   "--feature-dir", str(cached), "--num-classes", "2",
                   "--batch-size", "8", "--epochs", "1", "--output", str(root / "c_out")]
            with patch.object(sys, "argv", ["run_downstream"] + cmd):
                main()
            self.assertTrue((root / "c_out/scaler.joblib").exists())
            self.assertTrue((root / "c_out/best_model.pt").exists())

            image = Image.fromarray(np.full((224, 224, 3), 70, dtype=np.uint8))
            mask = Image.fromarray(np.zeros((224, 224), dtype=np.uint8))
            seg_manifest = root / "seg.csv"
            with seg_manifest.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["image_path", "mask_path", "split"])
                for i, partition in enumerate(["train", "train", "val"]):
                    image.save(root / f"im{i}.png")
                    mask.save(root / f"mask{i}.png")
                    w.writerow([f"im{i}.png", f"mask{i}.png", partition])
            cmd = ["--task", "segmentation", "--head", "linear", "--mode", "image",
                   "--manifest", str(seg_manifest), "--num-classes", "2",
                   "--batch-size", "2", "--epochs", "1", "--output", str(root / "s_out")]
            with patch.object(sys, "argv", ["run_downstream"] + cmd), \
                 patch("torch.hub.load", return_value=FakeDINO()):
                main()
            self.assertTrue((root / "s_out/best_model.pt").exists())
            self.assertTrue((root / "s_out/summary.json").exists())



if __name__ == "__main__":
    unittest.main()

# Additional CLI smoke tests are defined above the __main__ guard via a new class.
