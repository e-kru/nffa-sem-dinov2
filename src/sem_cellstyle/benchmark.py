"""Full Large SSL measurement and strict recipe binding; no scientific weights saved."""
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from src.ssl.experiment import digest
from .recipe import contract, require_initialization, resolve, schedules
from .runner import initialize, loader, require_a100
from .training import accumulated_step
from .resume_check import ResumeCheckError, verify_resume_update, restore_probe_state as _restore_probe_state


def benchmark_identity(cfg, data, initialization):
    value = contract(cfg, data, initialization, 'cuda')
    # Candidate batch and worker count differ, all scientific settings stay fixed.
    c = value['configuration']
    c['runtime'].pop('physical_batch'); c['runtime'].pop('accumulation_steps')
    c['train'].pop('batch_size_per_gpu'); c['train'].pop('num_workers')
    return digest(value)


def recommendation(rows, target=.90):
    stable = [row for row in rows if row['status'] == 'ok' and row['finite']
              and row['peak_reserved_bytes'] <= target*row['total_vram_bytes']]
    if not stable:
        return dict(physical_batch=None, reason='No candidate passed all training, memory, resume and export checks; inspect candidate errors before changing the recipe')
    best = max(stable, key=lambda row: row['original_images_per_second'])
    # Within 5% of best, prefer lower memory pressure over the largest batch.
    near = [row for row in stable if row['original_images_per_second'] >= .95*best['original_images_per_second']]
    choice = min(near, key=lambda row: row['peak_reserved_bytes'])
    return dict(physical_batch=choice['physical_batch'], accumulation_steps=128//choice['physical_batch'],
                reason='Finite, <=90% reserved VRAM; lower memory among candidates within 5% of best end-to-end throughput')


def validate_choice(result, cfg, data, initialization):
    if result.get('benchmark_identity') != benchmark_identity(cfg, data, initialization):
        raise ValueError('Benchmark dataset/weights/recipe/source contract differs from training')
    batch = cfg.runtime.physical_batch
    rows = [row for row in result.get('results', []) if row.get('physical_batch') == batch]
    if len(rows) != 1 or rows[0]['status'] != 'ok' or not rows[0]['finite']:
        raise ValueError('Chosen physical batch has not passed the full Large benchmark')
    if rows[0]['peak_reserved_bytes'] > .90*rows[0]['total_vram_bytes']:
        raise ValueError('Chosen physical batch exceeds the VRAM reserve target')
    if rows[0].get('resume_verified') is not True or rows[0].get('export_verified') is not True:
        raise ValueError('Benchmark must verify resume and live-teacher export before the scientific run')
    row = rows[0]
    if row.get('resume_update_bitwise_exact') is False:
        if row.get('cuda_update_drift_accepted') is not True or cfg.runtime.get('allow_cuda_resume_update_drift') is not True:
            raise ValueError('Benchmark recorded post-update drift without study approval')
    elif row.get('cuda_update_drift_accepted'):
        raise ValueError('Benchmark resume verdict is inconsistent')


def measure(cfg, dataset, collate, identity, output, batch, warmup=10, measured=30):
    from .runner import verify_export
    from src.ssl.checkpoints import atomic_json, save_snapshot
    output = Path(output)
    props = require_a100()
    initialization = require_initialization(cfg)
    cfg = resolve(cfg, len(dataset), batch)
    model, optimizer = initialize(cfg, initialization, 'cuda')
    sched = schedules(cfg)
    if warmup+measured+1 > cfg.experiment.schedule_total_steps:
        raise ValueError('Benchmark exceeds the real full schedule')
    stream = iter(loader(cfg, dataset, collate, 'cuda', 0, warmup+measured))
    for step in range(warmup):
        accumulated_step(model, optimizer, stream, sched, step)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    print(dict(stage='warmup_complete', physical_batch=batch, updates=warmup), flush=True)
    started = time.perf_counter()
    rows, intervals = [], []
    # CUDA events measure GPU stream time. Wall intervals include loader, Python,
    # native loss/gradient finite checks and the full accumulated update.
    for step in range(warmup, warmup+measured):
        start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start_event.record(); begun = time.perf_counter()
        rows.append(accumulated_step(model, optimizer, stream, sched, step))
        intervals.append(time.perf_counter()-begun); end_event.record()
        rows[-1]['events'] = (start_event, end_event)
    torch.cuda.synchronize(); elapsed = time.perf_counter()-started
    del stream
    print(dict(stage='measurements_complete', physical_batch=batch, updates=measured), flush=True)
    finite = all(np.isfinite(row[k]) for row in rows for k in ('total_loss', 'gradient_norm_backbone', 'gradient_norm_dino_head'))
    gpu_times = [row['events'][0].elapsed_time(row['events'][1])/1000 for row in rows]
    result = dict(status='ok', finite=bool(finite), physical_batch=batch, accumulation_steps=128//batch,
                  optimizer_updates=measured, wall_seconds=elapsed, median_update_seconds=float(np.median(intervals)),
                  mean_update_seconds=float(np.mean(intervals)), p95_update_seconds=float(np.percentile(intervals,95)),
                  mean_gpu_stream_seconds=float(np.mean(gpu_times)), original_images_per_second=128*measured/elapsed,
                  mean_dataloader_wait_seconds=float(np.mean([r['dataloader_wait_seconds'] for r in rows])),
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                  total_vram_bytes=props.total_memory, workers=cfg.train.num_workers,
                  gpu_name=props.name, gpu_count=1, precision='bfloat16', attention='native_pytorch_sdpa')
    regularizer=getattr(model,'regularization',None)
    result['original_reference_parameter_bytes']=sum(p.numel()*p.element_size() for p in regularizer.originals().values()) if regularizer else 0
    result['extra_reference_encoder']=bool(regularizer is not None and regularizer.reference is not None)
    # Separate IO/integrity check; these disposable weights can never be promoted.
    begun = time.perf_counter()
    temp_run = output / f'candidate_{batch}_DISPOSABLE'
    temp_run.mkdir(exist_ok=False)
    print(dict(stage='checkpoint_and_export_check', physical_batch=batch), flush=True)
    update = warmup+measured
    path = save_snapshot(temp_run, model, optimizer, update, OmegaConf.to_container(cfg),
                         contract(cfg, identity, initialization, 'cuda'), sched, initialization, milestone=False,
                         data_cursor=dict(completed_steps=update, image_presentations=update*128,
                                          completed_microbatches=update*cfg.runtime.accumulation_steps,
                                          policy='draw_seed_v1; corpus-permutation-v1; keep_epoch_tails'))
    verify_export(path, model, initialization)
    torch.cuda.synchronize()
    result['checkpoint_io_and_verification_seconds'] = time.perf_counter()-begun
    payload = torch.load(path/'training.pth', map_location='cpu', weights_only=False)
    if hasattr(model,'regularization'):
        model.regularization.verify_original(torch.load(initialization,map_location='cpu',weights_only=True,mmap=True))
    allow_drift = cfg.runtime.get('allow_cuda_resume_update_drift') is True
    print(dict(stage='resume_check', physical_batch=batch, allow_cuda_resume_update_drift=allow_drift), flush=True)
    diagnostic_path = output/f'candidate_{batch}_resume_diagnostics.json'
    result.update(resume_verified=False, export_verified=True, resume_diagnostics=str(diagnostic_path))
    try:
        report = verify_resume_update(model, optimizer, dataset, collate, sched, update, payload, diagnostic_path,
                                      allow_cuda_update_drift=allow_drift)
    except ResumeCheckError as error:
        result.update(status='resume_error', error=str(error))
        error.result = result
        atomic_json(output/f'candidate_{batch}_training_metrics.json', result)
        raise
    result.update(resume_verified=True, resume_update_bitwise_exact=report['update_bitwise_exact'],
                  cuda_update_drift_accepted=report['cuda_update_drift_accepted'])
    # Never leave large disposable snapshots in the benchmark directory.
    import shutil
    shutil.rmtree(temp_run)
    return result
