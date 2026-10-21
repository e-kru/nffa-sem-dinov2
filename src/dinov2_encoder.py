"""Single canonical DINOv2 model factory and SEM input transform.

Used by legacy feature extraction scripts AND the shared downstream pipeline.
Keep get_dinov2() and get_dinov2_transform() backward-compatible.
"""

from pathlib import Path

import torch
from torchvision import transforms

from src.preprocessing_v2 import preprocess_sem_image

MODEL_NAME = "dinov2_vits14"
DINO_DIMS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}


def get_dinov2_transform():
    """Original pipeline: V2 crop -> Resize(256) -> CenterCrop(224) -> normalize.

    The operations now live in preprocessing_v2 so segmentation uses the
    same geometry. Signature retained for scripts/extract_all_features.py.
    """
    return transforms.Lambda(preprocess_sem_image)


def load_dinov2_checkpoint(model: torch.nn.Module, checkpoint: str | Path):
    """Load a *compatible backbone-only* state dict into DINOv2.

    Supports a raw backbone state dict, or a wrapper containing either
    encoder_state_dict or state_dict. Rejects shape/key mismatches explicitly.
    """
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("DINOv2 checkpoint must contain a state dict.")
    state = payload.get("encoder_state_dict", payload.get("state_dict", payload))
    if not isinstance(state, dict):
        raise ValueError("Checkpoint state_dict must be a dictionary.")
    model.load_state_dict(state, strict=True)
    return model


def get_dinov2(model_name: str = MODEL_NAME, *, frozen: bool = True,
                checkpoint: str | Path | None = None):
    """Load pretrained DINOv2; frozen=True preserves the historical baseline.

    Using frozen=False is for supervised end-to-end fine-tuning, *not*
    self-supervised SEM posttraining. A modified encoder invalidates old caches.
    """
    if model_name not in DINO_DIMS:
        raise ValueError(f"Unsupported DINOv2 model: {model_name}")
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    if checkpoint is not None:
        load_dinov2_checkpoint(model, checkpoint)
    model.requires_grad_(not frozen)
    if frozen:
        model.eval()
    return model
