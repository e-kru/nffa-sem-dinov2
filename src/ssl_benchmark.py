"""Standalone SSL performance measurements; never saves or resumes model weights."""
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import traceback

import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader

from src.ssl.adapter import SingleGPUAdapter, train_step
from src.ssl.checkpoints import atomic_json
from src.ssl.data import SEMTrainDataset, augmentation_and_collate, seed_worker, sha256, train_records
from src.ssl.experiment import configuration, digest, step_schedules
from src.ssl.stream import CursorBatches, DrawCollate, DrawDataset
from src.ssl.upstream import PIN, ROOT
from src.ssl.loading import configure_dataset, loader_kwargs, options, process_memory


def aggregate(samples):
    values = np.asarray(samples, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Timing samples must be nonempty, finite and nonnegative')
    return dict(mean=float(values.mean()), median=float(np.median(values)),
                p95=float(np.percentile(values, 95)))


def materialize(rows):
    """One device-to-host scalar transfer AFTER the window's synchronization."""
    if not rows:
        raise ValueError('Empty training window')
    tensor_keys = [k for k, value in rows[0].items() if isinstance(value, torch.Tensor)]
    packed = torch.stack([torch.stack([row[k].detach().float() for k in tensor_keys]) for row in rows])
    scalars = packed.cpu().tolist()
    result = []
    for row, values in zip(rows, scalars):
        cpu = {k: v for k, v in row.items() if k not in tensor_keys}
        cpu.update(zip(tensor_keys, values))
        for key, value in cpu.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise FloatingPointError(f'Nonfinite {key} at schedule index {cpu.get("iteration")}')
        result.append(cpu)
    return result


class CudaWindowTimer:
    """Consecutive completion events include uncovered loader/host idle time.

    The other event interval brackets H2D + full train_step on the current
    stream. All event reads follow one synchronization at the window boundary.
    """
    def begin(self):
        self.boundaries, self.starts = [], []
        self.begun = time.perf_counter()
        self.boundaries.append(self._event())

    @staticmethod
    def _event():
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def before_step(self):
        self.starts.append(self._event())

    def after_step(self):
        self.boundaries.append(self._event())

    def finish(self):
        torch.cuda.synchronize()
        wall = time.perf_counter() - self.begun
        step = [a.elapsed_time(b) / 1000 for a, b in zip(self.boundaries, self.boundaries[1:])]
        compute = [a.elapsed_time(b) / 1000 for a, b in zip(self.starts, self.boundaries[1:])]
        return dict(wall_seconds=wall, step_seconds=step, train_step_seconds=compute)


def training_window(iterator, step, count, offset, timer, *, clock=time.perf_counter):
    """Backend-independent orchestration also exercised by small CPU fixtures."""
    rows, waiting, profiles = [], [], []
    timer.begin()
    for index in range(count):
        begun = clock()
        batch = next(iterator)
        waiting.append(clock() - begun)
        if isinstance(batch, dict) and 'loader_profile' in batch:
            profiles.append(batch['loader_profile'])
        timer.before_step()
        rows.append(step(batch, offset + index))
        timer.after_step()
    timing = timer.finish()
    timing['dataloader_wait_seconds'] = waiting
    if profiles:
        timing['loader_profiles'] = profiles
    # Materialization and validation are outside the measured window.
    return dict(timing=timing, rows=materialize(rows))


def summarize_window(window, batch_size):
    timing, rows = window['timing'], window['rows']
    n = len(rows)
    if not n or any(len(timing[key]) != n for key in ('step_seconds', 'train_step_seconds', 'dataloader_wait_seconds')):
        raise ValueError('Incomplete timing window')
    if not math.isfinite(timing['wall_seconds']) or timing['wall_seconds'] <= 0:
        raise ValueError('Invalid window duration')
    step = aggregate(timing['step_seconds'])
    if step['median'] <= 0:
        raise ValueError('Nonpositive step duration')
    return dict(measured_steps=n, measured_originals=n * batch_size,
                window_wall_seconds=timing['wall_seconds'],
                originals_per_second=n * batch_size / timing['wall_seconds'],
                step_seconds=step, train_step_seconds=aggregate(timing['train_step_seconds']),
                dataloader_wait_seconds=aggregate(timing['dataloader_wait_seconds']),
                p95_to_median=step['p95'] / step['median'], finite_training=True)


def validate_gpu():
    # Only the isolated child calls this: the parent never creates a CUDA context.
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if len(visible.split(',')) != 1 or not visible.strip() or visible.strip().startswith('MIG-'):
        raise RuntimeError('Set CUDA_VISIBLE_DEVICES to exactly one physical GPU index or GPU UUID')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one CUDA GPU must be visible')
    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    uuid = str(getattr(props, 'uuid', 'unavailable'))
    if 'A100' not in props.name or 'MIG' in props.name or uuid.startswith('MIG-') or props.total_memory < 75 * 1024**3:
        raise RuntimeError(f'Expected a physical A100 80GB without MIG: {props}')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 is unavailable')
    free, total = torch.cuda.mem_get_info()
    return dict(name=props.name, uuid=uuid, total_bytes=total, free_before_model_bytes=free,
                cuda_visible_devices=visible, capability=list(torch.cuda.get_device_capability(0)))


def state_fingerprint(model):
    digestor = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digestor.update(f'{key}:{value.dtype}:{tuple(value.shape)}'.encode())
        digestor.update(value.detach().contiguous().numpy().tobytes())
    return digestor.hexdigest()


def dataset_from_inventory(cfg, inventory):
    # Parent has already hashed all selected train files once. Recheck selection
    # and manifest identity, never inspect validation/test images in the child.
    paths = [Path(cfg.dataset.manifest), Path(cfg.dataset.primary_split), *map(Path, cfg.dataset.holdout_splits)]
    current = [dict(path=str(p.resolve()), sha256=sha256(p)) for p in paths]
    if inventory['provider'] != cfg.dataset.provider or inventory['root'] != str(Path(cfg.dataset.root).resolve()) or current != inventory['manifests']:
        raise ValueError('Dataset configuration or manifests changed after audit')
    records = train_records(*paths[:2], paths[2:])
    if records != [{k: row[k] for k in ('image_id', 'group_id')} for row in inventory['records']]:
        raise ValueError('Train selection differs from the audited dataset')
    if digest(inventory['records']) != inventory['records_sha256'] or len(records) != inventory['count']:
        raise ValueError('Corrupt dataset inventory')
    transform, collate = augmentation_and_collate(cfg)
    dataset = SEMTrainDataset(Path(cfg.dataset.root), records, transform, validate_paths=False)
    return configure_dataset(cfg, dataset, inventory), collate


def memory_peaks():
    return dict(allocated_bytes=torch.cuda.max_memory_allocated(), reserved_bytes=torch.cuda.max_memory_reserved())


def measure_case(plan, batch_size, progress):
    progress['phase'] = 'hardware_check'
    progress['gpu'] = validate_gpu()
    for name, expected in {**plan['provenance']['source_sha256'], **plan['provenance']['benchmark_sources']}.items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f'Benchmark source changed after preparation: {name}')
    cfg = configuration(Path(plan['config']))
    if sha256(Path(plan['config'])) != plan['config_sha256']:
        raise ValueError('Benchmark configuration changed')
    checkpoint = Path(plan['checkpoint'])
    if sha256(checkpoint) != PIN['backbone_sha256'] or plan['checkpoint_sha256'] != PIN['backbone_sha256']:
        raise ValueError('Initialization must match the original pinned Meta release')
    inventory_path = Path(plan['inventory'])
    if sha256(inventory_path) != plan['inventory_sha256']:
        raise ValueError('Benchmark inventory changed')
    inventory = json.loads(inventory_path.read_text())
    cfg.train.batch_size_per_gpu = batch_size
    cfg.train.num_workers = plan['workers']
    cfg.optim.lr = cfg.optim.base_lr * (batch_size / 1024)**.5
    progress['effective_max_lr'] = cfg.optim.lr
    progress['loader_options'] = vars(options(cfg))
    loader_options = options(cfg)
    progress['phase'] = 'dataset'
    dataset, collate = dataset_from_inventory(cfg, inventory)
    seed = cfg.train.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    policy = cfg.experiment.deterministic
    torch.use_deterministic_algorithms(policy != 'off', warn_only=policy == 'warn')
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    progress['phase'] = 'initialization'
    model = SingleGPUAdapter(cfg, checkpoint).train()
    if (not model.pretrained_verified or model.teacher.training
            or any(p.requires_grad for p in model.teacher.parameters())
            or any(not torch.equal(value, model.teacher.state_dict()[key]) for key, value in model.student.state_dict().items())):
        raise ValueError('Pretrained student/teacher initialization or freeze invariant failed')
    progress['initial_state_sha256'] = state_fingerprint(model)
    torch.cuda.reset_peak_memory_stats()
    model.to('cuda')
    optimizer = torch.optim.AdamW(model.parameter_groups(), betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2))
    if optimizer.state:
        raise ValueError('AdamW must start with empty state')
    schedules = step_schedules(cfg)
    total_steps = plan['warmup_steps'] + plan['measured_steps']
    loader = DataLoader(DrawDataset(dataset, seed),
        batch_sampler=CursorBatches(len(dataset), batch_size, seed, 0, total_steps),
        collate_fn=DrawCollate(collate, seed),
        worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(seed), **loader_kwargs(cfg, 'cuda'))
    iterator = None
    try:
        iterator = iter(loader)

        transfers = []
        def step(batch, index):
            events = None
            if loader_options.profile:
                events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                transfers.append(events)
            return train_step(model, optimizer, batch, schedules, index,
                              precision='bf16', diagnostics=True, defer_metrics=True,
                              non_blocking=loader_options.non_blocking, transfer_events=events)

        progress['phase'] = 'warmup'
        warmup = training_window(iterator, step, plan['warmup_steps'], 0, CudaWindowTimer())
        warmup_peaks = memory_peaks()
        torch.cuda.reset_peak_memory_stats()
        progress['phase'] = 'measurement'
        measured = training_window(iterator, step, plan['measured_steps'], plan['warmup_steps'], CudaWindowTimer())
        if transfers:
            measured['timing']['h2d_seconds'] = [a.elapsed_time(b)/1000 for a,b in transfers[-plan['measured_steps']:]]
        measured_peaks = memory_peaks()
        metrics = summarize_window(measured, batch_size)
        metrics['host_memory'] = process_memory([os.getpid(), *[w.pid for w in getattr(iterator, '_workers', [])]])
        if loader_options.decoded_cache:
            metrics['decoded_cache_bytes'] = dataset.decoded_cache.metadata['bytes']
        if transfers:
            metrics['h2d_seconds'] = aggregate(measured['timing']['h2d_seconds'])
        if any(p.grad is not None for p in model.teacher.parameters()):
            raise ValueError('Teacher received gradients')
        return dict(**metrics, peak_allocated_bytes=max(warmup_peaks['allocated_bytes'], measured_peaks['allocated_bytes']),
                    peak_reserved_bytes=max(warmup_peaks['reserved_bytes'], measured_peaks['reserved_bytes']),
                    warmup_peaks=warmup_peaks, measured_peaks=measured_peaks,
                    warmup=warmup, measured=measured)
    finally:
        # Also close workers after an OOM/nonfinite failure before process exit.
        if iterator is not None and hasattr(iterator, '_shutdown_workers'):
            iterator._shutdown_workers()


def failure_status(error):
    if isinstance(error, torch.cuda.OutOfMemoryError) or (isinstance(error, RuntimeError) and 'CUDA out of memory' in str(error)):
        return 'oom'
    if isinstance(error, FloatingPointError):
        return 'nonfinite'
    return 'error'


def run_case(action, batch_size, destination):
    """Exception classification is unit tested without allocating a GPU."""
    result = dict(batch_size=batch_size, status='error')
    try:
        result.update(action(result))
        result['status'] = 'ok'
        result['phase'] = 'complete'
    except Exception as error:
        result.update(status=failure_status(error), error=str(error), traceback=traceback.format_exc())
        print(result['traceback'], file=sys.stderr, flush=True)
        # Release references held by the exception's training frame before GC.
        error.__traceback__ = None
        if torch.cuda.is_initialized():
            try:
                peaks = memory_peaks()
                result.update(peak_allocated_bytes=peaks['allocated_bytes'], peak_reserved_bytes=peaks['reserved_bytes'])
            except Exception:
                pass
    finally:
        gc.collect()
        if torch.cuda.is_initialized():
            try:
                torch.cuda.empty_cache()
            except Exception as error:
                result['cleanup_error'] = str(error)
    atomic_json(destination, result)
    return result


def subprocess_case(command, log_path):
    """Wait for process exit (and context destruction) before the next case."""
    with log_path.open('x') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return process.wait()
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            except ProcessLookupError:
                process.wait()
            raise


def memory_budget(row, fraction):
    gpu = row['gpu']
    return fraction * min(gpu['total_bytes'], gpu['free_before_model_bytes'])


def recommendation(results, fraction=.85, near_best=.03, max_p95_ratio=2.0):
    eligible, excluded = [], {}
    for row in results:
        reasons = []
        if row['status'] != 'ok' or not row.get('finite_training'):
            reasons.append(row['status'])
        else:
            if row['peak_reserved_bytes'] > memory_budget(row, fraction):
                reasons.append('insufficient VRAM headroom')
            if row['p95_to_median'] > max_p95_ratio:
                reasons.append('p95/median exceeds stability limit')
        if reasons:
            excluded[str(row['batch_size'])] = reasons
        else:
            eligible.append(row)
    if not eligible:
        return dict(batch_size=None, reason='No measured configuration meets stability and VRAM criteria', excluded=excluded)
    best = max(r['originals_per_second'] for r in eligible)
    selected = min((r for r in eligible if r['originals_per_second'] >= (1-near_best)*best), key=lambda r:r['batch_size'])
    return dict(batch_size=selected['batch_size'], originals_per_second=selected['originals_per_second'],
                reason=f'Smallest stable batch within {100*near_best:g}% of best eligible throughput; reserved VRAM <= {100*fraction:g}% of available capacity',
                excluded=excluded, eligible_batches=[r['batch_size'] for r in eligible])


def may_try_128(results):
    row = next((r for r in results if r['batch_size'] == 64), None)
    smaller = [r['originals_per_second'] for r in results if r['batch_size'] < 64 and r['status'] == 'ok']
    return bool(row and row['status'] == 'ok' and row.get('finite_training') and smaller
                and row['peak_reserved_bytes'] <= memory_budget(row, .70)
                and row['p95_to_median'] <= 2
                and row['originals_per_second'] >= 1.05 * max(smaller))


def orchestrate(plan, output, *, runner=subprocess_case, progress_callback=None):
    results = []

    def one(batch):
        folder = output / f'batch_{batch:04d}'
        folder.mkdir()
        path = folder / 'result.json'
        command = [sys.executable, str(ROOT / 'scripts/benchmark_sem_ssl_a100.py'),
                   '--worker-plan', str((output / 'plan.json').resolve()), '--worker-batch', str(batch)]
        code = runner(command, folder / 'worker.log')
        if path.exists():
            row = json.loads(path.read_text())
            if row['batch_size'] != batch or row['status'] not in ('ok', 'oom', 'nonfinite', 'error'):
                raise ValueError('Invalid worker result identity/status')
            if code != 0 and row['status'] == 'ok':
                row.update(status='error', error=f'Worker exited {code} after writing success')
        else:
            row = dict(batch_size=batch, status='error', error=f'Worker exited {code} without a result; inspect worker.log')
        row['worker_returncode'] = code
        row['worker_log'] = str(folder / 'worker.log')
        # Cases must have the same initial full model, including heads/centers.
        for key in ('initial_state_sha256',):
            known = {r[key] for r in results if key in r}
            if key in row and known and row[key] not in known:
                raise ValueError('Batch-size cases have different initial model states')
        uuids = {r['gpu']['uuid'] for r in results if 'gpu' in r}
        if 'gpu' in row and uuids and row['gpu']['uuid'] not in uuids:
            raise ValueError('Physical GPU changed across subprocesses')
        results.append(row)
        atomic_json(output / 'partial_results.json', dict(results=results))
        print(f"batch={batch}: {row['status']}; originals/s={row.get('originals_per_second')}", flush=True)
        if progress_callback:
            progress_callback(row)

    for batch in plan['batch_sizes']:
        one(batch)
    optional = dict(requested=plan['try_128'], executed=False)
    if plan['try_128']:
        optional['executed'] = may_try_128(results)
        if optional['executed']:
            one(128)
        else:
            optional['reason'] = 'Requires stable batch 64, <=70% available VRAM reserved, and >=5% throughput gain over every smaller successful batch'
    return results, optional


def write_report(output, plan, results, optional, *, plots=True):
    chosen = recommendation(results, plan['max_memory_fraction'])
    report = dict(schema=1, benchmark_only=True, plan=plan, results=results, optional_128=optional, recommendation=chosen)
    atomic_json(output / 'benchmark.json', report)
    columns = ['batch_size', 'status', 'measured_steps', 'measured_originals', 'originals_per_second',
               'step_mean_seconds', 'step_median_seconds', 'step_p95_seconds', 'train_step_mean_seconds',
               'dataloader_wait_mean_seconds', 'dataloader_wait_p95_seconds', 'peak_allocated_bytes',
               'peak_reserved_bytes', 'window_wall_seconds', 'finite_training', 'effective_max_lr',
               'host_pss_bytes','decoded_cache_bytes','h2d_mean_seconds','error']
    with (output / 'benchmark.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in results:
            flat = {k: row.get(k) for k in columns}
            flat['host_pss_bytes']=row.get('host_memory',{}).get('pss_bytes')
            flat['h2d_mean_seconds']=row.get('h2d_seconds',{}).get('mean')
            for field, source, metric in [('step_mean_seconds', 'step_seconds', 'mean'), ('step_median_seconds', 'step_seconds', 'median'),
                    ('step_p95_seconds', 'step_seconds', 'p95'), ('train_step_mean_seconds', 'train_step_seconds', 'mean'),
                    ('dataloader_wait_mean_seconds', 'dataloader_wait_seconds', 'mean'), ('dataloader_wait_p95_seconds', 'dataloader_wait_seconds', 'p95')]:
                flat[field] = row.get(source, {}).get(metric)
            writer.writerow(flat)
    if plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        successful = [r for r in results if r['status'] == 'ok']
        for kind in ('throughput', 'memory'):
            fig, ax = plt.subplots(figsize=(8, 5))
            batches = [r['batch_size'] for r in successful]
            if kind == 'throughput':
                ax.plot(batches, [r['originals_per_second'] for r in successful], 'o-', label='Complete SSL pipeline')
                ax.set_ylabel('Original images / second (window wall time)')
            else:
                for field, label in [('peak_allocated_bytes', 'Allocated'), ('peak_reserved_bytes', 'Reserved')]:
                    ax.plot(batches, [r[field] / 1024**3 for r in successful], 'o-', label=label)
                ax.plot(batches, [memory_budget(r, plan['max_memory_fraction']) / 1024**3 for r in successful], '--', label='Recommendation VRAM budget')
                ax.set_ylabel('Peak GPU memory (GiB), including warmup')
            if not successful:
                ax.text(.5, .5, 'No successful GPU measurements', transform=ax.transAxes, ha='center')
            failures = ', '.join(f"B{r['batch_size']}: {r['status']}" for r in results if r['status'] != 'ok')
            ax.set_title(failures or 'A100 SEM SSL benchmark')
            ax.set_xticks([r['batch_size'] for r in results])
            ax.set_xlabel('Physical batch size (original images)')
            ax.grid(alpha=.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(output / f'{kind}_vs_batch.png', dpi=160)
            plt.close(fig)
    return report
