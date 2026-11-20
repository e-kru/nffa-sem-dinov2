"""Versioned residual classification head; the historical heads stay intact."""
from collections.abc import Sequence

import torch
from torch import nn


class ResidualMLPBlock(nn.Module):
    """Original hidden-layer operations plus an identity or projected shortcut."""
    def __init__(self, in_dim: int, out_dim: int, dropout: float = .1):
        super().__init__()
        if min(in_dim, out_dim) <= 0 or not 0 <= dropout < 1:
            raise ValueError('Invalid residual block dimensions or dropout')
        self.main = nn.Sequential(nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim),
                                  nn.ReLU(), nn.Dropout(dropout))
        self.shortcut = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shortcut(x) + self.main(x)


class ResidualMLP(nn.Module):
    """Three main hidden layers by default; width changes add linear shortcuts."""
    def __init__(self, in_dim: int, num_classes: int,
                 hidden_dims: Sequence[int] = (512, 256, 128), dropout: float = .1):
        super().__init__()
        if in_dim <= 0 or num_classes <= 1 or not hidden_dims or any(width <= 0 for width in hidden_dims):
            raise ValueError('Invalid residual MLP dimensions')
        widths = (in_dim, *hidden_dims)
        self.blocks = nn.Sequential(*[ResidualMLPBlock(a, b, dropout) for a, b in zip(widths, widths[1:])])
        self.classifier = nn.Linear(widths[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f'Classification expects (B,D), got {tuple(x.shape)}')
        return self.classifier(self.blocks(x))
