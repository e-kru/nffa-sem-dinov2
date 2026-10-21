"""Task heads. Input: (B,D) for classification or (B,D,h,w) for segmentation."""

from collections.abc import Sequence

import torch
from torch import nn


class ClassificationHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, kind: str = "linear",
                 hidden_dims: Sequence[int] = (512, 256, 128), dropout: float = 0.0):
        super().__init__()
        if kind == "linear":
            self.net = nn.Linear(in_dim, num_classes)
        elif kind == "mlp":
            layers = []
            previous = in_dim
            for width in hidden_dims:
                layers += [nn.Linear(previous, width), nn.BatchNorm1d(width),
                           nn.ReLU(), nn.Dropout(dropout)]
                previous = width
            layers.append(nn.Linear(previous, num_classes))
            self.net = nn.Sequential(*layers)
        else:
            raise ValueError(f"Unsupported head: {kind}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Classification expects (B,D), got {tuple(x.shape)}")
        return self.net(x)


class SegmentationHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, kind: str = "linear",
                 hidden_dims: Sequence[int] = (512, 256, 128), dropout: float = 0.0):
        super().__init__()
        if kind == "linear":
            self.net = nn.Conv2d(in_dim, num_classes, kernel_size=1)
        elif kind == "mlp":
            # A shared MLP at each spatial position, implemented with 1x1 convolutions.
            layers = []
            previous = in_dim
            for width in hidden_dims:
                layers += [nn.Conv2d(previous, width, kernel_size=1),
                           nn.BatchNorm2d(width), nn.ReLU(), nn.Dropout2d(dropout)]
                previous = width
            layers.append(nn.Conv2d(previous, num_classes, kernel_size=1))
            self.net = nn.Sequential(*layers)
        else:
            raise ValueError(f"Unsupported head: {kind}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Segmentation expects (B,D,h,w), got {tuple(x.shape)}")
        return self.net(x)


def make_head(task: str, kind: str, in_dim: int, num_classes: int,
              hidden_dims: Sequence[int] = (512, 256, 128),
              dropout: float = 0.0) -> nn.Module:
    if not 0 <= dropout < 1:
        raise ValueError("Dropout must be in [0, 1).")
    if in_dim <= 0 or num_classes <= 1 or any(n <= 0 for n in hidden_dims):
        raise ValueError("Invalid model dimensions.")
    if kind == "mlp" and not hidden_dims:
        raise ValueError("MLP requires at least one hidden layer.")
    if task == "classification":
        return ClassificationHead(in_dim, num_classes, kind, hidden_dims, dropout)
    if task == "segmentation":
        return SegmentationHead(in_dim, num_classes, kind, hidden_dims, dropout)
    raise ValueError(f"Unknown task: {task}")
