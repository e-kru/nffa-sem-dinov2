"""Resolve presentation budgets only from the approved corpus and fixed batch."""
import math
from pathlib import Path

from omegaconf import OmegaConf

from src.ssl.adapter import load_config
from src.ssl.data import sha256
from src.ssl.upstream import PIN, ROOT, activate

DEFAULT_CONFIG = ROOT / 'configs/sem_ssl/sem_cellstyle_vitl14_v1.yaml'


def configuration(path=DEFAULT_CONFIG):
    cfg = load_config(Path(path))
    if cfg.recipe != 'sem_cellstyle_vitl14_v1':
        raise ValueError('Expected the opt-in Large SEM recipe')
    if cfg.runtime.effective_batch != 128 or cfg.runtime.physical_batch not in (None, 16, 32):
        raise ValueError('Effective batch must be 128; physical batch must be unresolved, 16 or 32')
    if cfg.student.get('freeze_first_blocks', 0) != 0:
        raise ValueError('All Large blocks must remain trainable')
    if cfg.runtime.activation_checkpoint_reentrant or cfg.runtime.deterministic_cuda_algorithms:
        raise ValueError('Recipe requires non-reentrant checkpointing and CUDA determinism off')
    if type(cfg.runtime.get('allow_cuda_resume_update_drift', False)) is not bool:
        raise ValueError('allow_cuda_resume_update_drift must be a boolean')
    if cfg.loader.decoded_cache is not None:
        raise ValueError('This recipe does not enable a full-corpus decoded cache')
    if cfg.runtime.seed != 42:
        raise ValueError('The predeclared first experiment uses seed 42')
    splits = list(cfg.evaluation.get('classification_splits', ['A','B']))
    if not splits or len(set(splits)) != len(splits) or set(splits)-{'A','B'}:
        raise ValueError('Classification validation splits must be a nonempty unique subset of A/B')
    if cfg.evaluation.get('head_profile','legacy') not in ('legacy','residual_v1'):
        raise ValueError('Unknown downstream head profile')
    from .regularization import options
    options(cfg)
    return cfg


def budgets(cfg, originals):
    if type(originals) is not int or originals < 2:
        raise ValueError('verified_train_originals must be an integer >= 2')
    def step(passes):
        return math.ceil(passes * originals / cfg.runtime.effective_batch)
    horizon = cfg.budget.full_original_presentation_passes
    stop = cfg.budget.initial_run_until_passes
    if not 0 < stop <= horizon:
        raise ValueError('Initial stage must lie within the full schedule')
    milestones = list(cfg.budget.checkpoints_at_passes)
    if milestones != sorted(set(milestones)) or milestones[0] != 0 or milestones[-1] != horizon:
        raise ValueError('Milestones must be sorted, unique, include zero and full horizon')
    return dict(verified_train_originals=originals, effective_batch=128,
                full_original_presentations=horizon*originals, initial_original_presentations=stop*originals,
                schedule_total_steps=step(horizon), run_until_step=step(stop),
                milestones=sorted(set(step(p) for p in milestones)),
                pass_to_step={str(p): step(p) for p in milestones},
                warmup_steps=step(cfg.budget.warmup_passes),
                freeze_last_layer_steps=step(cfg.budget.freeze_last_layer_passes),
                teacher_temp_warmup_steps=step(cfg.budget.teacher_temperature_warmup_passes))


def resolve(cfg, originals, physical_batch):
    if physical_batch not in (16, 32):
        raise ValueError('Benchmark and fix physical batch at 16 or 32 before scientific training')
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg.runtime.physical_batch = physical_batch
    cfg.runtime.accumulation_steps = 128 // physical_batch
    cfg.train.batch_size_per_gpu = physical_batch
    cfg.experiment = budgets(cfg, originals)
    return cfg


def schedules(cfg):
    activate()
    from dinov2.utils.utils import CosineScheduler
    n, warmup = cfg.experiment.schedule_total_steps, cfg.experiment.warmup_steps
    if not 0 < warmup < n:
        raise ValueError('Warmup must be shorter than the full schedule')
    result = {}
    for module in ('backbone', 'head'):
        result[module + '_lr'] = CosineScheduler(cfg.optim[module + '_lr'], cfg.optim[module + '_min_lr'], n,
                                                warmup_iters=warmup, start_warmup_value=0)
    result['weight_decay'] = CosineScheduler(cfg.optim.weight_decay, cfg.optim.weight_decay_end, n)
    result['momentum'] = CosineScheduler(cfg.teacher.momentum_teacher, cfg.teacher.final_momentum_teacher, n)
    temp = cfg.experiment.teacher_temp_warmup_steps
    result['teacher_temp'] = CosineScheduler(cfg.teacher.teacher_temp, cfg.teacher.teacher_temp, temp,
                                             warmup_iters=temp, start_warmup_value=cfg.teacher.warmup_teacher_temp)
    return result


def contract(cfg, data_identity, initialization, device):
    import torch
    import torchvision
    scientific = OmegaConf.to_container(cfg, resolve=True)
    # Stage extension changes the stopping boundary only; schedules stay fixed.
    scientific['experiment'].pop('run_until_step')
    files = [*sorted((ROOT / 'src/sem_cellstyle').glob('*.py')),
             *sorted((ROOT / 'src/ssl').glob('*.py')), ROOT / 'src/preprocessing_v2.py',
             ROOT / 'scripts/train_sem_cellstyle.py']
    if cfg.dataset.provider == 'verified_nffa_inventory_v1':
        files.append(ROOT / 'src/ssl_benchmark.py')
    return dict(schema='sem-cellstyle-training-v1', configuration=scientific,
                data=data_identity, initialization_sha256=sha256(initialization), upstream_commit=PIN['commit'],
                device_type=str(device).split(':')[0], torch=str(torch.__version__), torchvision=str(torchvision.__version__),
                sources={str(p.relative_to(ROOT)): sha256(p) for p in files})


def require_initialization(cfg):
    value, expected = cfg.initialization.checkpoint, cfg.initialization.sha256
    if not value or not expected or len(expected) != 64:
        raise ValueError('Set the original dinov2_vitl14 checkpoint and its trusted SHA256; no automatic download')
    path = Path(value).expanduser().resolve()
    if sha256(path) != expected:
        raise ValueError('Original Large checkpoint SHA256 differs from the declared initialization')
    return path
