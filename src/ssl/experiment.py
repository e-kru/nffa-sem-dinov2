"""Phase-A configuration, original step schedules and train-only dataset plan."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import PIL
import omegaconf
from omegaconf import OmegaConf
import torch
import torchvision

from src.ssl.adapter import load_config
from src.ssl.data import SEMTrainDataset, augmentation_and_collate, sha256, train_records
from src.ssl.upstream import PIN, ROOT, activate

from src.config import VITS14_CHECKPOINT

REFERENCE = VITS14_CHECKPOINT


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def configuration(path):
    cfg = load_config(path)
    exp = cfg.experiment
    horizon = exp.schedule_total_steps
    if type(horizon) is not int or horizon < 2 or not 0 <= exp.run_until_step <= horizon:
        raise ValueError("Invalid schedule_total_steps/run_until_step")
    if not 0 <= exp.warmup_steps < horizon or not 0 <= exp.freeze_last_layer_steps <= horizon:
        raise ValueError("Invalid LR warmup/freeze duration")
    if not 1 <= exp.teacher_temp_warmup_steps <= horizon:
        raise ValueError("Teacher temperature warmup must be within the original horizon")
    if exp.recovery_interval < 1 or exp.keep_recent < 1 or exp.log_interval < 1:
        raise ValueError("Checkpoint retention and intervals must be positive")
    if exp.deterministic not in ('off', 'warn', 'strict'):
        raise ValueError("deterministic must be off, warn or strict")
    if len(set(exp.milestones)) != len(exp.milestones) or any(type(x) is not int or not 0 <= x <= horizon for x in exp.milestones):
        raise ValueError("Invalid milestone steps")
    expected = {"global_crops_size": 224, "local_crops_size": 98, "local_crops_number": 8,
                "global_crops_scale": [0.32, 1.0], "local_crops_scale": [0.05, 0.32]}
    if any(OmegaConf.to_container(cfg.crops)[k] != v for k, v in expected.items()):
        raise ValueError("Phase A must retain the original multicrop settings")
    if (cfg.dino.head_n_prototypes != 65536 or cfg.dino.head_hidden_dim != 2048
            or cfg.dino.head_bottleneck_dim != 256 or cfg.dino.head_nlayers != 3
            or cfg.dino.loss_weight != 1 or cfg.ibot.loss_weight != 1 or cfg.dino.koleo_loss_weight != 0.1
            or cfg.ibot.mask_sample_probability != 0.5 or list(cfg.ibot.mask_ratio_min_max) != [0.1, 0.5]):
        raise ValueError("Phase A requires the original heads, masking and objective weights")
    return cfg


def step_schedules(cfg):
    from dinov2.utils.utils import CosineScheduler
    exp = cfg.experiment
    n = exp.schedule_total_steps
    common = dict(base_value=cfg.optim.lr, final_value=cfg.optim.min_lr, total_iters=n,
                  warmup_iters=exp.warmup_steps, start_warmup_value=0)
    lr, last = CosineScheduler(**common), CosineScheduler(**common)
    last.schedule[:exp.freeze_last_layer_steps] = 0
    temp_n = exp.teacher_temp_warmup_steps
    return dict(lr=lr, last_layer_lr=last,
                weight_decay=CosineScheduler(cfg.optim.weight_decay, cfg.optim.weight_decay_end, n),
                momentum=CosineScheduler(cfg.teacher.momentum_teacher, cfg.teacher.final_momentum_teacher, n),
                teacher_temp=CosineScheduler(cfg.teacher.teacher_temp, cfg.teacher.teacher_temp, n,
                    warmup_iters=temp_n, start_warmup_value=cfg.teacher.warmup_teacher_temp))


def dataset_plan(cfg, workers=8):
    """Provider boundary: trainer consumes (dataset, original collate, identity).

    Later providers may return multi-source records without changing the loop.
    Only this NFFA provider is implemented; validation/test files are not opened.
    """
    if cfg.dataset.provider != 'nffa_v3':
        raise ValueError("Only the audited nffa_v3 dataset provider is implemented")
    if cfg.dataset.get('inventory'):
        from src.ssl_benchmark import dataset_from_inventory
        inventory = json.loads(Path(cfg.dataset.inventory).read_text())
        return (*dataset_from_inventory(cfg, inventory), inventory)
    paths = [Path(cfg.dataset.manifest), Path(cfg.dataset.primary_split),
             *map(Path, cfg.dataset.holdout_splits)]
    records = train_records(*paths[:2], paths[2:])
    root = Path(cfg.dataset.root).resolve()

    def inspect(row):
        path = (root / row['image_id']).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing/outside train image: {row['image_id']}")
        stat = path.stat()
        return {**row, 'bytes': stat.st_size, 'sha256': sha256(path)}

    # Hash only selected training images, never images from held-out partitions.
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        inventory = list(pool.map(inspect, records))
    identity = dict(provider='nffa_v3', root=str(root), count=len(records),
                    manifests=[dict(path=str(p.resolve()), sha256=sha256(p)) for p in paths],
                    records=inventory, records_sha256=digest(inventory),
                    selection='primary '+Path(cfg.dataset.primary_split).stem+' train; additional train-group constraints: '
                              +str(len(cfg.dataset.holdout_splits))+'; included crop_v2 only')
    transform, collate = augmentation_and_collate(cfg)
    dataset = SEMTrainDataset(root, records, transform, validate_paths=False)
    from src.ssl.loading import configure_dataset
    dataset = configure_dataset(cfg, dataset, identity)
    return dataset, collate, identity


def contract(cfg, data, checkpoint, precision):
    full = OmegaConf.to_container(cfg, resolve=True)
    # Runtime stop/IO choices are recorded, but may change during a resume.
    scientific = json.loads(json.dumps(full))
    for key in ('run_until_step', 'milestones', 'recovery_interval', 'keep_recent', 'log_interval'):
        scientific['experiment'].pop(key)
    scientific['train'].pop('num_workers', None)
    source_paths = [ROOT / 'src/preprocessing_v2.py', ROOT / 'scripts/train_sem_ssl.py',
                    *sorted((ROOT / 'src/ssl').glob('*.py'))]
    return dict(schema=1, config=scientific, data_sha256=digest(data),
                initialization_sha256=sha256(checkpoint), upstream_commit=PIN['commit'], precision=precision,
                software=dict(python=platform.python_version(), torch=str(torch.__version__),
                              torchvision=str(torchvision.__version__), numpy=np.__version__,
                              pillow=PIL.__version__, omegaconf=omegaconf.__version__, cuda=torch.version.cuda),
                source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in source_paths})
