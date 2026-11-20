"""Exact, seekable V0 epochs and milestone execution using the existing SSL step."""

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import signal
import time

from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader

from src.sem_cellstyle.recipe import schedules
from src.sem_cellstyle.runner import initialize
from src.sem_cellstyle.training import accumulated_step
from src.ssl.checkpoints import (atomic_json, find_latest, reconcile_metrics, restore_rng,
                                 save_snapshot, trim_recovery, validate_snapshot)
from src.ssl.data import seed_worker
from src.ssl.experiment import digest
from src.ssl.loading import loader_kwargs
from src.ssl.stream import DrawCollate, DrawDataset, draw_seed


MILESTONE_EPOCHS = (0, 1, 5, 10, 25, 50, 100, 200, 300)


@dataclass(frozen=True)
class EpochLayout:
    originals: int
    physical_batch: int

    def __post_init__(self):
        if self.physical_batch < 2 or self.originals < 4 or self.originals < 4*self.updates:
            raise ValueError('Two KoLeo microbatches require at least four images per epoch')

    @property
    def effective_batch(self):
        return 2 * self.physical_batch

    @property
    def updates(self):
        return math.ceil(self.originals/self.effective_batch)

    @property
    def tail(self):
        return self.originals % self.effective_batch or self.effective_batch

    def offset(self, update):
        if update == self.updates:
            return self.originals
        # For size%effective_batch in 1..3, borrow originals from the
        # preceding update so the tail has two nonempty KoLeo batches of two.
        if update == self.updates-1 and self.tail < 4:
            return self.originals-4
        return update*self.effective_batch

    def cursor(self, step):
        epoch, update = divmod(step, self.updates)
        position = self.offset(update)
        return dict(policy='exact_epoch_randperm_balanced_tail_v1', completed_steps=step,
                    completed_microbatches=2*step, completed_epochs=epoch, image_position=position,
                    image_presentations=epoch*self.originals+position)

    def batches(self, seed, start, stop):
        previous, permutation = None, []
        for step in range(start, stop):
            epoch, unit = divmod(step, self.updates)
            if epoch != previous:
                generator = torch.Generator().manual_seed(draw_seed(seed, 'corpus-permutation-v1', epoch))
                permutation = torch.randperm(self.originals, generator=generator).tolist()
                previous = epoch
            left, right = self.offset(unit), self.offset(unit+1)
            middle = (left+right+1)//2
            for first, last in ((left, middle), (middle, right)):
                yield [(permutation[i], epoch*self.originals+i) for i in range(first, last)]


class EpochBatches:
    def __init__(self, layout, seed, start, stop):
        if not 0 <= start <= stop:
            raise ValueError('Invalid epoch update window')
        self.layout, self.seed, self.start, self.stop = layout, seed, start, stop

    def __len__(self):
        return 2*(self.stop-self.start)

    def __iter__(self):
        return self.layout.batches(self.seed, self.start, self.stop)


def full_schedule(cfg, originals):
    if cfg.runtime.accumulation_steps != 2 or cfg.runtime.effective_batch != 2*cfg.runtime.physical_batch:
        raise ValueError('V0 epochs require two equal physical microbatches for each full update')
    layout = EpochLayout(originals, cfg.runtime.physical_batch)
    # Preserve the Aug-A pilot's 5%, 1% and 20% schedule proportions.
    horizon, warmup, freeze, temperature = (layout.updates*n for n in (300,15,3,60))
    cfg.experiment = dict(name='sem_v0_vits14_aug_a_300epochs',
                          schedule_total_steps=horizon, run_until_step=horizon, warmup_steps=warmup,
                          freeze_last_layer_steps=freeze, teacher_temp_warmup_steps=temperature,
                          milestones=[layout.updates*epoch for epoch in MILESTONE_EPOCHS],
                          epoch_to_step={str(epoch):layout.updates*epoch for epoch in MILESTONE_EPOCHS})
    return layout


def epoch_loader(cfg, dataset, collate, layout, device, start, stop):
    seed = cfg.runtime.seed
    return DataLoader(DrawDataset(dataset, seed), batch_sampler=EpochBatches(layout, seed, start, stop),
                      collate_fn=DrawCollate(collate, seed), worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(seed), **loader_kwargs(cfg, device))


def verified_milestone(folder, expected, layout, original_sha):
    metadata = validate_snapshot(folder)
    step = metadata['step']
    if metadata['contract_sha256'] != digest(expected) or not metadata['milestone']:
        raise ValueError(f'Milestone contract mismatch: {folder}')
    payload = torch.load(folder/'training.pth', map_location='cpu', weights_only=False, mmap=True)
    if (payload['step'] != step or digest(payload['contract']) != metadata['contract_sha256']
            or payload['data_cursor'] != layout.cursor(step)):
        raise ValueError(f'Milestone data cursor mismatch: {folder}')
    export = torch.load(folder/'teacher_backbone.pth', map_location='cpu', weights_only=True, mmap=True)
    prefix = 'teacher.backbone.'
    teacher = {k[len(prefix):]:v for k,v in payload['model'].items() if k.startswith(prefix)}
    if export.keys() != teacher.keys() or any(not torch.equal(value, teacher[key]) for key,value in export.items()):
        raise ValueError(f'Teacher export tensor mismatch: {folder}')
    if step == 0 and metadata['files']['teacher_backbone.pth'] != original_sha:
        raise ValueError('Epoch-zero teacher differs from the original Meta release')
    return metadata


def train_to(cfg, data, collate, layout, initialization, expected, root, target, *,
             device='cuda', interrupt_after=None):
    """Reach a milestone using the fixed total schedule and full restored state."""
    root.mkdir(parents=True, exist_ok=True)
    sched = schedules(cfg)
    if target not in cfg.experiment.milestones:
        raise ValueError('Training target must be a declared milestone')
    saved = None
    start = 0
    run_file = root/'run.json'
    if run_file.exists():
        _, saved, rejected = find_latest(root, expected)
        if rejected:
            print('Rejected checkpoints: '+str(rejected), flush=True)
        start = saved['step']
        if start > target or saved['data_cursor'] != layout.cursor(start):
            raise ValueError('Saved checkpoint does not match the requested epoch cursor')
        if any(saved['schedule'][key] != s.schedule.tolist() for key,s in sched.items()):
            raise ValueError('The full 300-epoch schedules changed')
        reconcile_metrics(root, start)
    elif any(root.iterdir()):
        raise FileExistsError(f'Unexpected files in fresh training output {root}')
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    model, optimizer = initialize(cfg, initialization, device)
    if saved is not None:
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        restore_rng(saved['rng'])
        del saved
    else:
        atomic_json(run_file, dict(config=cfg_dict, contract=expected))
        save_snapshot(root, model, optimizer, 0, cfg_dict, expected, sched,
                      initialization, milestone=True, data_cursor=layout.cursor(0))
    if start == target:
        return

    stopping = False

    def request_stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signals = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    step = start
    recovery = cfg.checkpointing.recovery_every_optimizer_updates
    try:
        batches = iter(epoch_loader(cfg, data, collate, layout, device, start, target))
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        with (root/'metrics.jsonl').open('a') as stream:
            for index in range(start, target):
                before = time.perf_counter()
                row = accumulated_step(model, optimizer, batches, sched, index,
                                       precision='bf16' if device == 'cuda' else 'fp32')
                step = index+1
                cursor = layout.cursor(step)
                begin = layout.cursor(index)['image_presentations']
                if row.pop('draw_ids') != list(range(begin, cursor['image_presentations'])):
                    raise ValueError('Consumed SSL draws differ from the epoch cursor')
                row.pop('record_indices')
                row.pop('group_lrs')
                elapsed = time.perf_counter()-before
                row.update(wall_seconds=elapsed, original_presentations=cursor['image_presentations'],
                           epoch=cursor['completed_epochs'], sample_position=cursor['image_position'],
                           original_images_per_second=(cursor['image_presentations']-begin)/elapsed,
                           peak_vram_bytes=torch.cuda.max_memory_allocated() if device == 'cuda' else 0)
                stream.write(json.dumps(row, allow_nan=False)+'\n')
                if step == target or step % 50 == 0:
                    stream.flush()
                    print(json.dumps({k:row[k] for k in ('step','epoch','total_loss','original_images_per_second')}),
                          flush=True)
                    atomic_json(root/'status.json', cursor | {'status':'running'})
                interrupted = stopping or step == interrupt_after
                if step == target or step % recovery == 0 or interrupted:
                    stream.flush()
                    os.fsync(stream.fileno())
                    save_snapshot(root, model, optimizer, step, cfg_dict, expected, sched, initialization,
                                  milestone=step == target, data_cursor=cursor)
                    trim_recovery(root, cfg.checkpointing.keep_recent_recovery_snapshots)
                if interrupted:
                    atomic_json(root/'status.json', cursor | {'status':'interrupted'})
                    raise InterruptedError('Training checkpoint saved. Repeat the same command to resume.')
        atomic_json(root/'status.json', layout.cursor(step) | {'status':'stage_complete'})
    except Exception as error:
        if not isinstance(error, InterruptedError):
            atomic_json(root/'status.json', layout.cursor(step) | {'status':'failed', 'error':str(error)})
        raise
    finally:
        for sig, handler in signals.items():
            signal.signal(sig, handler)
