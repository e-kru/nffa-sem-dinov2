"""Training lifecycle reusing the existing transactional checkpoint utilities."""
import json
from pathlib import Path
import random
import signal
import time

import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader

from src.ssl.adapter import SingleGPUAdapter
from src.ssl.checkpoints import (atomic_json, find_latest, reconcile_metrics, restore_rng,
                                run_lock, save_snapshot, trim_recovery, validate_snapshot)
from src.ssl.data import seed_worker
from src.ssl.loading import loader_kwargs
from src.ssl.stream import DrawDataset, DrawCollate
from .data import PresentationBatches
from .recipe import contract, schedules
from .training import accumulated_step, checkpoint_student, parameter_groups


def require_a100():
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('Expose exactly one physical A100 80GB using CUDA_VISIBLE_DEVICES')
    props = torch.cuda.get_device_properties(0)
    if 'A100' not in props.name or props.total_memory < 75*1024**3 or not torch.cuda.is_bf16_supported():
        raise ValueError(f'Expected one BF16-capable A100 80GB: {props}')
    from dinov2.layers import attention, block
    if attention.XFORMERS_AVAILABLE or block.XFORMERS_AVAILABLE:
        raise ValueError('Set XFORMERS_DISABLED=1 before importing Meta; native SDPA is required')
    return props


def initialize(cfg, initialization, device):
    seed = cfg.runtime.seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if str(device).startswith('cuda'):
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    model = SingleGPUAdapter(cfg, initialization).to(device).train()
    if not model.pretrained_verified:
        raise ValueError('Original pretrained tensor equality was not verified')
    teacher_parameters = dict(model.teacher.named_parameters())
    for name, param in model.student.named_parameters():
        other = teacher_parameters[name]
        if param.data_ptr() == other.data_ptr() or not torch.equal(param, other) or other.requires_grad:
            raise ValueError('Teacher must be an independent, frozen, exact Student copy')
    from .regularization import attach
    attach(model)
    if cfg.runtime.student_activation_checkpointing:
        checkpoint_student(model)
    optimizer = torch.optim.AdamW(parameter_groups(model), lr=0.,
                                  betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2))
    return model, optimizer


def loader(cfg, dataset, collate, device, start, stop):
    k, b, seed = cfg.runtime.accumulation_steps, cfg.runtime.physical_batch, cfg.runtime.seed
    return DataLoader(DrawDataset(dataset, seed),
                      batch_sampler=PresentationBatches(len(dataset), b, seed, start*k, stop*k),
                      collate_fn=DrawCollate(collate, seed), worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(seed), **loader_kwargs(cfg, device))


def verify_export(path, model, initialization=None):
    """Strict tensor AND feature equality to live Teacher (also original at step 0)."""
    from copy import deepcopy
    metadata = validate_snapshot(path)
    state = torch.load(path / 'teacher_backbone.pth', map_location='cpu', weights_only=True)
    live = model.teacher.backbone
    for name, tensor in live.state_dict().items():
        if not torch.equal(tensor.detach().cpu(), state[name]):
            raise ValueError('Teacher export tensor mismatch: ' + name)
    if metadata['step'] == 0:
        original = torch.load(initialization, map_location='cpu', weights_only=True)
        if any(not torch.equal(value, original[key]) for key, value in state.items()):
            raise ValueError('Step zero differs from the original Large initialization')
    # A second CPU backbone avoids consuming more A100 memory during checkpointing.
    exported = deepcopy(live).cpu().eval()
    exported.load_state_dict(state, strict=True)
    device = next(live.parameters()).device
    side = model.cfg.crops.global_crops_size
    x = torch.randn(1, 3, side, side, generator=torch.Generator().manual_seed(73))
    # Compare on the same device/FP32 path: CPU vs CUDA kernels need not be bitwise equal.
    exported = exported.to(device)
    with torch.inference_mode():
        expected, actual = live(x.to(device)), exported(x.to(device))
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    del exported
    return dict(step=metadata['step'], live_teacher_feature_max_abs_error=0.,
                original_step_zero_tensor_equality=metadata['step'] == 0,
                original_step_zero_feature_max_abs_error=0. if metadata['step'] == 0 else None)


def execute(cfg, dataset, collate, identity, initialization, output, *, resume=False, device='cuda', interrupt_after=None):
    device, output = torch.device(device), Path(output)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    run_contract, sched = contract(cfg, identity, initialization, device), schedules(cfg)
    stop = cfg.experiment.run_until_step
    if not 0 <= stop <= cfg.experiment.schedule_total_steps:
        raise ValueError('Stopping boundary outside the original schedule')
    with run_lock(output):
        saved, start = None, 0
        if resume:
            saved_path, saved, _ = find_latest(output, run_contract)
            if not (saved_path / 'feature_verification.json').is_file():
                raise ValueError('Resume requires a verified teacher export')
            start = saved['step']
            if saved['schedule'] != {k: v.schedule.tolist() for k, v in sched.items()}:
                raise ValueError('Original schedules changed')
            if saved['data_cursor']['image_presentations'] != start*cfg.runtime.effective_batch:
                raise ValueError('Sampler cursor differs from effective-batch update count')
            if start > stop:
                raise ValueError('Resume target is behind the saved checkpoint')
        elif any(p.name != '.writer.lock' for p in output.iterdir()):
            raise FileExistsError('Use a new output directory or explicit --resume')
        model, optimizer = initialize(cfg, initialization, device)
        if saved:
            model.load_state_dict(saved['model'], strict=True); optimizer.load_state_dict(saved['optimizer'])
            if hasattr(model, 'regularization'):
                model.regularization.verify_original(torch.load(initialization, map_location='cpu', weights_only=True, mmap=True))
            restore_rng(saved['rng']); reconcile_metrics(output, start)
            verify_export(saved_path, model, initialization)
        else:
            atomic_json(output / 'run.json', dict(config=cfg_dict, contract=run_contract,
                        status='prepared_with_verified_dataset', original_initialization_verified=True,
                        backbone_parameter_names=[name for name,_ in model.student.backbone.named_parameters()]))
        def snapshot(step, milestone):
            cursor = dict(completed_steps=step, image_presentations=step*cfg.runtime.effective_batch,
                          completed_microbatches=step*cfg.runtime.accumulation_steps,
                          policy='draw_seed_v1; corpus-permutation-v1; keep_epoch_tails')
            path = save_snapshot(output, model, optimizer, step, cfg_dict, run_contract, sched,
                                 initialization, milestone=milestone, data_cursor=cursor)
            atomic_json(path / 'feature_verification.json', verify_export(path, model, initialization))
            trim_recovery(output, cfg.checkpointing.keep_recent_recovery_snapshots)
        if not saved:
            snapshot(0, True)
        waiting_stop = [False]
        def request_stop(_signum, _frame):
            waiting_stop[0] = True
        old_handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGTERM, signal.SIGINT)}
        committed, last_saved = start, start
        try:
            batches = iter(loader(cfg, dataset, collate, device, start, stop))
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()
            for iteration in range(start, stop):
                begun = time.perf_counter()
                row = accumulated_step(model, optimizer, batches, sched, iteration,
                                       precision='bf16' if device.type == 'cuda' else 'fp32')
                committed = iteration+1
                elapsed = time.perf_counter()-begun
                row.update(wall_seconds=elapsed, original_images_per_second=cfg.runtime.effective_batch/elapsed,
                           completed_original_passes=committed*cfg.runtime.effective_batch/len(dataset),
                           peak_vram_bytes=torch.cuda.max_memory_allocated() if device.type == 'cuda' else 0)
                with (output / 'metrics.jsonl').open('a') as stream:
                    stream.write(json.dumps(row, allow_nan=False)+'\n'); stream.flush()
                print(json.dumps({k: row[k] for k in ('step', 'total_loss', 'original_images_per_second')}), flush=True)
                milestone = committed in cfg.experiment.milestones
                recovery = committed % cfg.checkpointing.recovery_every_optimizer_updates == 0
                interrupted = waiting_stop[0] or committed == interrupt_after
                if milestone or recovery or committed == stop or interrupted:
                    snapshot(committed, milestone); last_saved = committed
                atomic_json(output / 'status.json', dict(step=committed, status='running'))
                if interrupted:
                    break
            status = 'stage_complete' if committed == stop else 'interrupted'
            atomic_json(output / 'status.json', dict(status=status, step=committed,
                        schedule_total_steps=cfg.experiment.schedule_total_steps,
                        review_required=committed in [cfg.experiment.pass_to_step[str(p)] for p in (10, 25, 50, 100)]))
        except BaseException as exc:
            atomic_json(output / 'status.json', dict(status='failed', step=committed, last_saved_step=last_saved, error=str(exc)))
            raise
        finally:
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
        return model, optimizer
