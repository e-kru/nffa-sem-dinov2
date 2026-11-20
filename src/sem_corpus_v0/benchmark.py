"""A disposable real-corpus A100 probe at the frozen Aug-A recipe."""

from collections import Counter
from pathlib import Path
import statistics
from tempfile import TemporaryDirectory
import time

from omegaconf import OmegaConf
import torch

from src.sem_cellstyle.recipe import schedules
from src.sem_cellstyle.resume_check import verify_resume_update
from src.sem_cellstyle.runner import initialize, verify_export
from src.sem_cellstyle.training import accumulated_step
from src.ssl.checkpoints import atomic_json, save_snapshot
from src.ssl.experiment import digest
from .epochs import epoch_loader


def benchmark(root, cfg, data, collate, layout, initialization, expected, gpu):
    warmup, measured = 10, 30
    end = warmup + measured
    if layout.updates <= end:
        raise ValueError('V0 probe and resume replay must lie within the first epoch')
    model, optimizer = initialize(cfg, initialization, 'cuda')
    sched = schedules(cfg)
    batches = iter(epoch_loader(cfg,data,collate,layout,'cuda',0,end))
    for update in range(warmup):
        accumulated_step(model,optimizer,batches,sched,update)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    timer = time.perf_counter()
    observed = Counter()
    step_times = []
    for update in range(warmup,end):
        start = time.perf_counter()
        row = accumulated_step(model,optimizer,batches,sched,update)
        step_times.append(time.perf_counter()-start)
        observed.update(data.records[index]['source'] for index in row['record_indices'])
    torch.cuda.synchronize()
    seconds = time.perf_counter()-timer
    allocated, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    del batches
    if len(observed) < 2:
        raise ValueError('The real-corpus benchmark did not sample several sources')
    root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='disposable_',dir=root) as tmp:
        begin = time.perf_counter()
        snapshot = save_snapshot(Path(tmp),model,optimizer,end,OmegaConf.to_container(cfg,resolve=True),
                                 expected,sched,initialization,milestone=False,data_cursor=layout.cursor(end))
        verify_export(snapshot,model,initialization)
        checkpoint_seconds = time.perf_counter()-begin
        payload = torch.load(snapshot/'training.pth', map_location='cpu', weights_only=False)
        # With no epoch tail yet, the existing regular resume probe consumes
        # exactly the same draw IDs as EpochLayout.
        resume = verify_resume_update(model,optimizer,data,collate,sched,end,payload,root/'resume.json',
                                      allow_cuda_update_drift=True)
    n = sum(observed.values())
    rate = n/seconds
    estimate = 300*layout.originals/rate
    interval = cfg.checkpointing.recovery_every_optimizer_updates
    writes = cfg.experiment.schedule_total_steps//interval + sum(
        step == 0 or step % interval != 0 for step in cfg.experiment.milestones)
    # Prior frozen ViT-S NFFA+A3 validation took roughly 18 min per
    # encoder; this is a planning allowance separate from the A100 probe.
    evaluation_allowance = (len(cfg.experiment.milestones)-1)*18*60
    result = dict(contract_sha256=digest(expected), gpu=gpu.name, gpu_total_memory=gpu.total_memory,
                  precision='bf16', warmup_updates=warmup, measured_updates=measured,
                  source_presentations=dict(observed), originals_per_second=rate, measured_seconds=seconds,
                  median_update_seconds=statistics.median(step_times), peak_allocated_bytes=allocated,
                  peak_reserved_bytes=reserved, finite_losses_and_gradients=True, export_verified=True,
                  resume_verified=resume['passed'], resume_update_bitwise_exact=resume['update_bitwise_exact'],
                  checkpoint_and_verify_seconds=checkpoint_seconds, estimated_training_only_seconds=estimate,
                  planned_checkpoint_writes=writes, assumed_validation_seconds=evaluation_allowance,
                  estimated_total_wall_seconds=estimate+writes*checkpoint_seconds+evaluation_allowance,
                  scientific_weights_promoted=False)
    atomic_json(root/'benchmark.json',result)
    return result
