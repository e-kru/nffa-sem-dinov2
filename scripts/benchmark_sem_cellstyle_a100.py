"""Isolated full-update ViT-L/14 benchmark; never starts scientific training."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(name, '1')
os.environ['XFORMERS_DISABLED'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    from src.sem_cellstyle.recipe import DEFAULT_CONFIG, configuration, require_initialization, resolve
    from src.sem_cellstyle.data import provider
    from src.sem_cellstyle.benchmark import benchmark_identity, measure, recommendation
    from src.sem_cellstyle.resume_check import ResumeCheckError
    from src.ssl.checkpoints import atomic_json
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batch-sizes', type=int, nargs='+', choices=(16,32), default=[16,32])
    p.add_argument('--warmup-updates', type=int, default=10)
    p.add_argument('--measured-updates', type=int, default=30)
    p.add_argument('--num-workers', type=int, default=16)
    p.add_argument('--resume', action='store_true', help='Resume matching incomplete benchmark; reuse completed candidate results')
    p.add_argument('--candidate', type=int, choices=(16,32), help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.warmup_updates < 1 or args.measured_updates < 2 or args.num_workers < 0:
        p.error('Invalid timing budget/worker count')
    if len(set(args.batch_sizes)) != len(args.batch_sizes):
        p.error('Candidate batches must be unique')
    cfg = configuration(args.config); cfg.train.num_workers = args.num_workers
    dataset, collate, identity = provider(cfg)
    initialization = require_initialization(cfg)
    if args.candidate:
        import torch
        destination = args.output / f'candidate_{args.candidate}.json'
        if destination.exists():
            raise FileExistsError(destination)
        try:
            row = measure(cfg, dataset, collate, identity, args.output, args.candidate,
                          args.warmup_updates, args.measured_updates)
        except torch.cuda.OutOfMemoryError as error:
            row = dict(status='oom', physical_batch=args.candidate, finite=False, error=str(error))
        except ResumeCheckError as error:
            row = error.result or dict(status='resume_error', physical_batch=args.candidate, finite=True,
                                       resume_verified=False, error=str(error))
        except Exception as error:
            row = dict(status='error', physical_batch=args.candidate, finite=False,
                       error=f'{type(error).__name__}: {error}')
        import shutil
        disposable = args.output / f'candidate_{args.candidate}_DISPOSABLE'
        if disposable.is_dir():
            shutil.rmtree(disposable)
        atomic_json(destination, row)
        print(json.dumps(row)); return
    planned_cfg = resolve(cfg,len(dataset),args.batch_sizes[0])
    plan = dict(benchmark_identity=benchmark_identity(planned_cfg,identity,initialization),
                batch_sizes=args.batch_sizes,warmup_updates=args.warmup_updates,
                measured_updates=args.measured_updates,num_workers=args.num_workers)
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(args.output)
        stored=args.output/'benchmark_plan.json'
        if not stored.exists() or json.loads(stored.read_text())!=plan:
            raise ValueError('Benchmark resume identity/protocol differs')
    else:
        args.output.mkdir(parents=True,exist_ok=False)
        atomic_json(args.output/'benchmark_plan.json',plan)
    rows = []
    for batch in args.batch_sizes:
        command = [sys.executable, str(Path(__file__).resolve()), '--config', str(args.config.resolve()),
                   '--output', str(args.output.resolve()), '--candidate', str(batch),
                   '--warmup-updates', str(args.warmup_updates), '--measured-updates', str(args.measured_updates),
                   '--num-workers', str(args.num_workers)]
        path = args.output / f'candidate_{batch}.json'
        if path.exists():
            result = None
        else:
            # Only disposable files owned by this validated benchmark plan.
            import shutil
            disposable=args.output/f'candidate_{batch}_DISPOSABLE'
            if disposable.is_dir(): shutil.rmtree(disposable)
            result = subprocess.run(command, check=False)
        row = json.loads(path.read_text()) if path.exists() else dict(status='error', physical_batch=batch,
                    finite=False, error=f'Child exited {result.returncode if result else None} without result')
        rows.append(row)
    resolved = resolve(cfg, len(dataset), args.batch_sizes[0])
    selected = recommendation(rows)
    good = next((r for r in rows if r['physical_batch'] == selected['physical_batch']), None)
    data = dict(schema='sem-cellstyle-full-update-benchmark-v1', results=rows, recommendation=selected,
                benchmark_identity=benchmark_identity(resolved, identity, initialization),
                warmup_optimizer_updates=args.warmup_updates, measured_optimizer_updates=args.measured_updates,
                allow_cuda_resume_update_drift=cfg.runtime.get('allow_cuda_resume_update_drift', False),
                scientific_training_started=False, weights_promoted=False,
                checkpoint_io_excluded_from_timing=True)
    if good:
        rate = good['original_images_per_second']
        data['runtime_estimate_compute_seconds'] = {
            'initial_stage': resolved.experiment.run_until_step*128/rate,
            'full_horizon': resolved.experiment.schedule_total_steps*128/rate}
    atomic_json(args.output / 'benchmark.json', data)
    fields = sorted(set().union(*(r.keys() for r in rows)))
    with (args.output / 'benchmark.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ok = [row for row in rows if row['status'] == 'ok']
    for filename, key, label, factor in (
            ('throughput_vs_batch.png', 'original_images_per_second', 'Original images / second', 1.),
            ('memory_vs_batch.png', 'peak_reserved_bytes', 'Peak reserved GPU memory (GiB)', 1./2**30)):
        fig, ax = plt.subplots()
        ax.plot([r['physical_batch'] for r in ok], [r[key]*factor for r in ok], marker='o')
        ax.set(xlabel='Physical batch (effective batch 128)', ylabel=label, xticks=args.batch_sizes)
        ax.grid(True, alpha=.3); fig.tight_layout(); fig.savefig(args.output / filename); plt.close(fig)
    print(json.dumps(data, indent=2))


if __name__ == '__main__':
    main()
