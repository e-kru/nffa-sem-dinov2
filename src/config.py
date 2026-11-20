"""Filesystem settings for the SEM data and encoder checkpoints."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = Path(os.environ.get("SEM_DATA_ROOT", PROJECT_ROOT / "data")).expanduser().resolve()
DATASET_ROOT = Path(os.environ.get("NFFA_SEM_ROOT", DATASETS_ROOT / "nffa_sem_100")).expanduser().resolve()
AACHEN_ROOT = Path(os.environ.get("AACHEN_SEM_ROOT", DATASETS_ROOT / "aachen_heerlen")).expanduser().resolve()
VITS14_CHECKPOINT = Path(os.environ.get(
    "DINOV2_VITS14_CHECKPOINT", PROJECT_ROOT / "outputs/pretrained/dinov2_vits14_pretrain.pth",
)).expanduser().resolve()
