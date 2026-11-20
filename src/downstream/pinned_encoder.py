"""Offline pinned ViT-S/14 loading, without changing the historical Hub factory."""
from pathlib import Path

from src.dinov2_encoder import load_dinov2_checkpoint
from src.ssl.upstream import activate


def get_pinned_dinov2(checkpoint: str | Path, name="dinov2_vits14"):
    """Load a compatible release/backbone export strictly; never fetch weights."""
    activate()
    from dinov2.hub import backbones
    if name not in ('dinov2_vits14', 'dinov2_vitl14'):
        raise ValueError('Supported pinned encoders: dinov2_vits14, dinov2_vitl14')
    model = getattr(backbones, name)(pretrained=False)
    load_dinov2_checkpoint(model, checkpoint)
    return model.requires_grad_(False).eval()
