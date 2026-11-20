"""Short frozen-feature worker sweep; never starts downstream evaluation or hashes datasets."""
import argparse
import csv
import json
import os
from pathlib import Path
import random
import resource
import shlex
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STEPS = [0, 500, 1000, 2000, 3000, 5000]
sys.path.insert(0, str(ROOT))
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['XFORMERS_DISABLED'] = '1'


def resources():
    cores = {}
    for block in Path('/proc/cpuinfo').read_text().split('\n\n'):
        fields = dict(line.split(':', 1) for line in block.splitlines() if ':' in line)
        fields = {k.strip(): v.strip() for k, v in fields.items()}
        if 'processor' in fields:
            cores[int(fields['processor'])] = (fields.get('physical id', '0'), fields.get('core id', fields['processor']))
    affinity = sorted(os.sched_getaffinity(0))
    memory = {line.split(':')[0]: int(line.split()[1])*1024
              for line in Path('/proc/meminfo').read_text().splitlines()
              if line.startswith(('MemTotal:', 'MemAvailable:'))}
    # Respect common cgroup quotas in addition to the affinity mask.
    cpus = len(affinity)
    bases = [Path('/sys/fs/cgroup')]
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        if line.startswith('0::'):
            bases.append(Path('/sys/fs/cgroup')/line.split('::', 1)[1].lstrip('/'))
    for base in bases:
        quota = base/'cpu.max'
        if quota.is_file():
            amount, period = quota.read_text().split()
            if amount != 'max':
                cpus = min(cpus, max(1, int(amount)//int(period)))
        limit, used = base/'memory.max', base/'memory.current'
        if limit.is_file() and used.is_file() and limit.read_text().strip() != 'max':
            memory['MemAvailable'] = min(memory['MemAvailable'], max(0, int(limit.read_text())-int(used.read_text())))
    return dict(host=os.uname().nodename, logical_cpus=len(cores), physical_cores=len(set(cores.values())),
                affinity=affinity, affinity_physical_cores=len({cores[i] for i in affinity if i in cores}),
                effective_cpus=cpus, **memory)


def recommend(rows):
    result = {}
    for task in ('classification', 'segmentation'):
        rates = {}
        for row in rows:
            if row['task'] == task and row['status'] == 'ok':
                rates.setdefault(row['workers'], []).append(row['images_per_second'])
        medians = {w: statistics.median(values) for w, values in rates.items()}
        if not medians:
            raise ValueError(f'No successful {task} measurement; inspect benchmark.json and logs')
        best = max(medians.values())
        result[task] = min(w for w, rate in medians.items() if rate >= .97*best)
    return result


def evaluation_commands(python, run, evaluation_root, report_root, workers):
    common = ['--run', str(run), '--steps', *map(str, STEPS), '--evaluation-root', str(evaluation_root)]
    return [[python, str(ROOT/'scripts/evaluate_sem_ssl.py'), *common, '--task', task,
             '--stage', 'all', '--workers', str(workers[task])]
            for task in ('classification', 'segmentation')] + [
                [python, str(ROOT/'scripts/report_sem_ssl.py'), *common, '--output', str(report_root)]]


def datasets(args):
    """Select real train/val samples once. Existing preprocessing, no image hashes."""
    from PIL import Image
    from torch.utils.data import ConcatDataset
    from src.config import DATASET_ROOT
    from src.downstream.data import ClassificationImageDataset, load_rows
    from src.aachen_ablation.cache import ExtractionDataset
    from src.aachen_ablation.data import manifest_rows, resolve
    from src.aachen_ablation.geometry import VARIANTS
    rows = [r for r in load_rows(ROOT/'outputs/dataset_v3/splits/split_A_exact.csv')
            if r['split'] in ('train', 'val')]
    sampled = random.Random(42).sample(rows, min(args.nffa_images, len(rows)))
    nffa = ConcatDataset([ClassificationImageDataset(sampled, split, DATASET_ROOT)
                          for split in ('train', 'val')])
    manifest = ROOT/'outputs/aachen_heerlen/splits/group_split_seed42.csv'
    rows = manifest_rows(manifest)  # Group checks only; no inspect_data / image hashing.
    sampled = random.Random(42).sample(rows, min(args.aachen_images, len(rows)))
    for row in sampled:
        with Image.open(resolve(manifest, row['image_path'])) as image:
            row['original_size'] = list(image.size)
    aachen = ExtractionDataset(manifest, sampled, VARIANTS['A3'], list(range(len(sampled))))
    return dict(classification=nffa, segmentation=aachen)


def cpu_seconds():
    return sum(r.ru_utime+r.ru_stime for r in
               (resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)))


def measure(task, dataset, encoder, workers, device):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    started_cpu = cpu_seconds()
    torch.cuda.synchronize()
    started = time.perf_counter()
    count = 0
    first_batch = None
    # Match the actual NFFA 512-image chunk lifecycle, including loader startup.
    chunks = [(i, min(i+512, len(dataset))) for i in range(0, len(dataset), 512)] if task == 'classification' else [(0, len(dataset))]
    for begin, end in chunks:
        loader = DataLoader(Subset(dataset, range(begin, end)),
            batch_size=64 if task == 'classification' else None, num_workers=workers,
            pin_memory=True, **(dict(multiprocessing_context='spawn', timeout=120) if task == 'segmentation' and workers else {}))
        iterator = None
        try:
            iterator = iter(loader)
            with torch.inference_mode():
                if task == 'classification':
                    for images, _ in iterator:
                        features = encoder.forward_global(images.to(device, non_blocking=True))
                        if features.shape != (len(images), 384) or not torch.isfinite(features).all():
                            raise ValueError('Invalid global features')
                        features.cpu().numpy().astype(np.float32)
                        count += len(images)
                        if first_batch is None:
                            first_batch = time.perf_counter()-started
                else:
                    for _, images in iterator:
                        # Existing A3 extractor has one whole-image region and splits at 4 crops.
                        values = [encoder.forward_dense(chunk.to(device)).cpu().numpy() for chunk in images.split(4)]
                        array = np.concatenate(values).astype(np.float32)
                        if array.shape != (1, 384, 48, 64) or not np.isfinite(array).all():
                            raise ValueError('Invalid A3 dense features')
                        count += 1
                        if first_batch is None:
                            first_batch = time.perf_counter()-started
        finally:
            if iterator is not None and hasattr(iterator, '_shutdown_workers'):
                iterator._shutdown_workers()
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-started
    cpu = cpu_seconds()-started_cpu
    return dict(images=count, wall_seconds=elapsed, images_per_second=count/elapsed,
                first_batch_seconds=first_batch, cpu_seconds=cpu, cpu_percent=100*cpu/elapsed)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--run', type=Path, default=ROOT/'outputs/sem_ssl/nffa_5000step_seed42')
    p.add_argument('--workers', type=int, nargs='+', default=[8, 16, 24, 32])
    p.add_argument('--nffa-images', type=int, default=1024)
    p.add_argument('--aachen-images', type=int, default=128)
    p.add_argument('--repeats', type=int, default=1)
    args = p.parse_args()
    if min(args.workers) < 1 or len(set(args.workers)) != len(args.workers) or min(args.nffa_images, args.aachen_images, args.repeats) < 1:
        p.error('Choose unique positive workers and positive sample/repeat counts')
    if args.nffa_images < 512 or args.nffa_images % 512:
        p.error('NFFA samples must be a multiple of 512 to preserve the real chunk lifecycle')
    if args.output.exists():
        raise FileExistsError(args.output)
    args.run = args.run.resolve()
    checkpoint = args.run/'checkpoints/step_005000/teacher_backbone.pth'
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    info = resources()
    print(json.dumps(info, indent=2), flush=True)
    selected = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not selected or ',' in selected or selected.startswith('MIG-'):
        p.error('Set CUDA_VISIBLE_DEVICES to exactly one physical A100')
    import torch
    from src.aachen_ablation.cache import make_pinned_encoder
    torch.set_num_threads(1)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or 'A100' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Exactly one A100 must be visible')
    gpu = torch.cuda.get_device_properties(0)
    gpu_id = str(gpu.uuid)
    if not gpu_id.startswith('GPU-'):
        gpu_id = 'GPU-'+gpu_id
    # Check the selected physical GPU query before starting the sweep.
    subprocess.run(['nvidia-smi', '-i', gpu_id, '--query-gpu=name', '--format=csv,noheader'], check=True)
    args.output.mkdir(parents=True)
    output = args.output.resolve()
    (output/'resources.json').write_text(json.dumps(dict(**info, gpu=str(gpu), cuda_visible_devices=selected), indent=2)+'\n')
    data = datasets(args)
    device = torch.device('cuda:0')
    encoder = make_pinned_encoder(device, checkpoint)
    with torch.inference_mode():
        encoder.forward_global(data['classification'][0][0].unsqueeze(0).to(device)).cpu()
        encoder.forward_dense(data['segmentation'][0][1].to(device)).cpu()
    rows = []
    for repeat in range(args.repeats):
        order = args.workers if repeat % 2 == 0 else list(reversed(args.workers))
        for task, dataset in data.items():
            for workers in order:
                row = dict(task=task, workers=workers, repeat=repeat, status='ok')
                # Conservative interpreter + prepared-queue allowance; no decoder caches.
                needed = workers*(512*1024**2 + 2*3*896*672*4) + 2*1024**3
                if workers > info['effective_cpus'] or needed > resources()['MemAvailable']*.8:
                    row.update(status='skipped', error='Insufficient affinity/quota CPUs or available RAM')
                else:
                    trace = output/f'{task}_w{workers}_r{repeat}_gpu.csv'
                    with trace.open('w') as stream:
                        monitor = subprocess.Popen(['nvidia-smi', '-i', gpu_id, '--query-gpu=utilization.gpu',
                            '--format=csv,noheader,nounits', '-lms', '500'], stdout=stream, stderr=subprocess.DEVNULL)
                        try:
                            row.update(measure(task, dataset, encoder, workers, device))
                            row['cpu_percent_of_affinity'] = row['cpu_percent']/info['effective_cpus']
                        except (RuntimeError, ValueError, OSError) as error:
                            row.update(status='error', error=str(error))
                            torch.cuda.empty_cache()
                        finally:
                            monitor.terminate()
                            monitor.wait()
                    samples = [float(line) for line in trace.read_text().splitlines() if line.strip().isdigit()]
                    row['gpu_utilization_percent'] = statistics.mean(samples) if samples else None
                    row['gpu_utilization_samples'] = len(samples)
                rows.append(row)
                (output/'benchmark.json').write_text(json.dumps(dict(results=rows, resources=info), indent=2)+'\n')
                print(json.dumps(row), flush=True)
    chosen = recommend(rows)
    report = dict(results=rows, resources=info, recommended_workers=chosen,
        method='Smallest worker count within 3% of best median images/s per task; includes loader startup/shutdown.',
        dataset_hashing=False, feature_files_written=False, full_evaluation_started=False)
    (output/'benchmark.json').write_text(json.dumps(report, indent=2)+'\n')
    fields = sorted(set().union(*(r.keys() for r in rows)))
    with (output/'benchmark.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    commands = evaluation_commands(sys.executable, args.run, ROOT/'outputs/encoder_evaluation/nffa_5000step_workers_v1',
                                   ROOT/'outputs/sem_ssl_reports/nffa_5000step_workers_v1', chosen)
    lines = ['#!/usr/bin/env bash', 'set -euo pipefail', 'cd '+shlex.quote(str(ROOT)),
             'export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1',
             'export CUDA_VISIBLE_DEVICES='+shlex.quote(selected), *map(shlex.join, commands)]
    (output/'evaluate_all.sh').write_text('\n'.join(lines)+'\n')
    print('Recommended workers: '+json.dumps(chosen), flush=True)
    print('Evaluation was NOT started. Review, then run: bash '+shlex.quote(str(output/'evaluate_all.sh')), flush=True)


if __name__ == '__main__':
    main()
