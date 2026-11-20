"""Bounded single-device loop using the existing Meta component adapter."""
import json
import math
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader

from src.ssl.adapter import SingleGPUAdapter, train_step
from src.ssl.checkpoints import (atomic_json, find_latest, reconcile_metrics, restore_rng,
                                run_lock, save_snapshot, trim_recovery)
from src.ssl.data import seed_worker
from src.ssl.experiment import contract, digest, step_schedules
from src.ssl.stream import CursorBatches, DrawCollate, DrawDataset
from src.ssl.loading import loader_kwargs, options


def execute(cfg, dataset, collate, data_identity, initialization, root, *, device='cpu', precision='fp32',
            resume=False, model_factory=SingleGPUAdapter, interrupt_after=None):
    """interrupt_after is a CPU-test seam for a safe post-update interruption."""
    root = Path(root).resolve()
    seed = cfg.train.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith('cuda'):
        torch.cuda.manual_seed_all(seed)
    policy = cfg.experiment.deterministic
    torch.use_deterministic_algorithms(policy != 'off', warn_only=policy == 'warn')
    if device.startswith('cuda'):
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    run_contract = contract(cfg, data_identity, initialization, precision)
    run_contract['device_type'] = torch.device(device).type
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    schedules = step_schedules(cfg)
    stop = cfg.experiment.run_until_step
    with run_lock(root):
        metadata_path = root / 'run.json'
        if metadata_path.exists() and not resume:
            raise FileExistsError('Run already exists; use --resume with compatible configuration')
        if resume and not metadata_path.exists():
            raise ValueError('Cannot resume an uninitialized run')
        saved, start = None, 0
        if resume:
            snapshot, saved, rejected = find_latest(root, run_contract)
            start = saved['step']
            if any(saved['schedule'][key] != value.schedule.tolist() for key, value in schedules.items()):
                raise ValueError('Original schedule changed')
            print(f'Resume {snapshot}; ignored invalid snapshots: {rejected}', flush=True)
            if stop < start:
                raise ValueError('run_until_step precedes saved optimizer step')
        else:
            if any(p.name != '.writer.lock' for p in root.iterdir()):
                raise FileExistsError('New training output must be empty')
        model = model_factory(cfg, initialization).to(device).train()
        if not model.pretrained_verified:
            raise ValueError('Pretrained initialization was not strictly verified')
        if any(p.requires_grad for p in model.teacher.parameters()) or model.teacher.training:
            raise RuntimeError('Teacher must be frozen and in evaluation mode')
        if not saved and any(not torch.equal(v, model.teacher.state_dict()[key])
                             for key, v in model.student.state_dict().items()):
            raise RuntimeError('Student/teacher initialization differs')
        optimizer = torch.optim.AdamW(model.parameter_groups(), betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2))
        if saved:
            model.load_state_dict(saved['model'], strict=True)
            optimizer.load_state_dict(saved['optimizer'])
            restore_rng(saved['rng'])
            reconcile_metrics(root, start)
            del saved
        else:
            atomic_json(metadata_path, dict(schema=1, contract=run_contract, config=config_dict,
                        dataset=data_identity, schedule_sha256=digest({k: v.schedule.tolist() for k, v in schedules.items()}),
                        hardware=dict(device=device, gpu=torch.cuda.get_device_name(0) if device.startswith('cuda') else None)))
            (root / 'selected_image_ids.txt').write_text(''.join(r['image_id'] + '\n' for r in data_identity['records']))

        def checkpoint(step, milestone):
            path = save_snapshot(root, model, optimizer, step, config_dict, run_contract, schedules,
                                 initialization, milestone=milestone)
            trim_recovery(root, cfg.experiment.keep_recent)
            print(f'Checkpoint step={step}: {path}', flush=True)
            return path

        if not resume:
            checkpoint(0, True)
        # A dedicated generator avoids consuming DropPath's CPU/CUDA RNG when
        # reconstructing the DataLoader iterator after resume.
        loader_options = options(cfg)
        loader = DataLoader(DrawDataset(dataset, seed),
            batch_sampler=CursorBatches(len(dataset), cfg.train.batch_size_per_gpu, seed, start, stop),
            collate_fn=DrawCollate(collate, seed),
            worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(seed),
            **loader_kwargs(cfg, device))
        interrupted = [False]
        previous_handlers = {}

        def request_stop(_signum, _frame):
            interrupted[0] = True

        for sig in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[sig] = signal.signal(sig, request_stop)
        step = start
        last_saved = start
        status = 'completed'
        try:
            with (root / 'metrics.jsonl').open('a', buffering=1) as metrics:
                iterator = iter(loader)
                while step < stop:
                    if interrupted[0]:
                        status = 'interrupted'
                        break
                    begun = time.perf_counter()
                    batch = next(iterator)
                    if device.startswith('cuda'):
                        torch.cuda.reset_peak_memory_stats()
                    row = train_step(model, optimizer, batch, schedules, step, precision=precision,
                                     diagnostics=True, non_blocking=loader_options.non_blocking)
                    # Scalar diagnostics already synchronize once at train_step end.
                    step += 1
                    elapsed = time.perf_counter() - begun
                    row.update(step=step, image_presentations=step * cfg.train.batch_size_per_gpu,
                               step_seconds=elapsed, originals_per_second=cfg.train.batch_size_per_gpu / elapsed,
                               peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device.startswith('cuda') else 0,
                               record_indices=batch['record_indices'], draw_ids=batch['draw_ids'])
                    if any(not math.isfinite(v) for v in row.values() if isinstance(v, float)):
                        raise FloatingPointError('Nonfinite training diagnostics; retaining last valid checkpoint')
                    row['collapse_signal'] = bool(row['cls_feature_std'] < 1e-3 or row['cls_pair_cosine'] > 0.99)
                    metrics.write(json.dumps(row, allow_nan=False) + '\n')
                    if step % cfg.experiment.log_interval == 0:
                        print(f"step={step}/{cfg.experiment.schedule_total_steps} loss={row['total_loss']:.5f} "
                              f"lr={row['lr']:.3g} grad={row['gradient_norm']:.4f} "
                              f"images/s={row['originals_per_second']:.2f} collapse_signal={row['collapse_signal']}", flush=True)
                    if step in cfg.experiment.milestones or step % cfg.experiment.recovery_interval == 0 or step == stop:
                        metrics.flush()
                        os.fsync(metrics.fileno())
                        checkpoint(step, step in cfg.experiment.milestones)
                        last_saved = step
                    if interrupt_after == step:
                        interrupted[0] = True
                if interrupted[0]:
                    status = 'interrupted'
                    if step != last_saved:
                        metrics.flush()
                        os.fsync(metrics.fileno())
                        checkpoint(step, step in cfg.experiment.milestones)
        except BaseException as error:
            # Never serialize a partially completed optimizer/EMA update.
            atomic_json(root / 'status.json', dict(status='failed', last_completed_step=step, error=repr(error)))
            raise
        finally:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
        atomic_json(root / 'status.json', dict(status=status, step=step, run_until_step=stop,
                                             schedule_total_steps=cfg.experiment.schedule_total_steps))
        return model, optimizer
