"""Downstream adapters. Encoder loading is centralized in src.dinov2_encoder.

Posttrained weights must be loaded in image mode or features re-extracted.
"""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from src.dinov2_encoder import DINO_DIMS, get_dinov2, load_dinov2_checkpoint
from .heads import make_head


class DINOv2Backbone(nn.Module):
    def __init__(self, name: str = "dinov2_vits14", frozen: bool = True,
                 checkpoint: str | Path | None = None, model: nn.Module | None = None):
        super().__init__()
        if name not in DINO_DIMS:
            raise ValueError(f"Unsupported DINOv2 model: {name}")
        self.name = name
        self.feature_dim = DINO_DIMS[name]
        self.frozen = frozen
        # Injection for tests avoids any model download.
        self.model = (get_dinov2(name, frozen=frozen, checkpoint=checkpoint)
                      if model is None else model)
        if model is not None and checkpoint is not None:
            load_dinov2_checkpoint(self.model, checkpoint)
        self.set_frozen(frozen)

    def set_frozen(self, frozen: bool) -> None:
        self.frozen = frozen
        self.model.requires_grad_(not frozen)
        if frozen:
            self.model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            self.model.eval()
        return self

    def forward_global(self, images: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.frozen):
            z = self.model(images)
        if z.ndim != 2 or z.shape[1] != self.feature_dim:
            raise RuntimeError(f"Unexpected global output: {tuple(z.shape)}")
        return z

    def forward_dense(self, images: torch.Tensor) -> torch.Tensor:
        if images.shape[-2] % 14 or images.shape[-1] % 14:
            raise ValueError("DINOv2 spatial input must be divisible by patch size 14.")
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.frozen):
            patches = self.model.get_intermediate_layers(
                images, n=1, reshape=True, norm=True)[0]
        if patches.ndim != 4 or patches.shape[1] != self.feature_dim:
            raise RuntimeError(f"Unexpected patch output: {tuple(patches.shape)}")
        return patches


class DownstreamModel(nn.Module):
    def __init__(self, backbone: nn.Module, task: str, head_kind: str,
                 num_classes: int, hidden_dims=(512, 256, 128), dropout: float = 0.0):
        super().__init__()
        self.backbone = backbone
        self.task = task
        self.head = make_head(task, head_kind, backbone.feature_dim,
                              num_classes, hidden_dims, dropout)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.task == "classification":
            return self.head(self.backbone.forward_global(images))
        if self.task == "segmentation":
            features = self.backbone.forward_dense(images)
            logits = self.head(features)
            return F.interpolate(logits, size=images.shape[-2:],
                                 mode="bilinear", align_corners=False)
        raise ValueError(f"Unknown task: {self.task}")
