"""Benchmark full SEM SSL updates on one A100 80GB; no experiment/checkpoint launch."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(name, '1')
os.environ['XFORMERS_DISABLED'] = '1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omegaconf import OmegaConf
from src.ssl.checkpoints import atomic_json
from src.ssl.data import sha256
from src.ssl.experiment import REFERENCE, configuration, contract, dataset_plan
from src.ssl.upstream import PIN, ROOT
from src.ssl_benchmark import measure_case, orchestrate, run_case, write_report
from src.ssl.loading import add_arguments, apply_arguments


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT / 'configs/sem_ssl/nffa_default.yaml')
    p.add_argument('--checkpoint', type=Path, default=REFERENCE)
    p.add_argument('--inventory', type=Path, help='Reuse an audited immutable corpus inventory; manifest/selection rechecked')
    p.add_argument('--output', type=Path, help='Required new directory, separate from SSL experiment outputs')
    p.add_argument('--batch-sizes', type=int, nargs='+', default=[8, 16, 32, 64])
    p.add_argument('--try-128', action='store_true', help='Only run if batch 64 is stable, has VRAM headroom and throughput still improves')
    p.add_argument('--warmup-steps', type=int, default=10)
    p.add_argument('--measured-steps', type=int, default=20)
    p.add_argument('--num-workers', '--workers', dest='workers', type=int, default=8)
    p.add_argument('--max-memory-fraction', type=float, default=.85,
                   help='Recommendation reserved-memory limit relative to free/total VRAM')
    add_arguments(p)
    # Internal subprocess entry; never offers a CPU/fake encoder production mode.
    p.add_argument('--worker-plan', type=Path, help=argparse.SUPPRESS)
    p.add_argument('--worker-batch', type=int, help=argparse.SUPPRESS)
    return p


def validate_options(args):
    if not args.output:
        raise ValueError('--output is required')
    if not 5 <= args.warmup_steps <= 10 or args.measured_steps < 1 or args.workers < 0:
        raise ValueError('Use 5–10 warmup steps, >=1 measured step and >=0 workers')
    if not 0 < args.max_memory_fraction < 1:
        raise ValueError('--max-memory-fraction must be between 0 and 1')
    if (not args.batch_sizes or len(args.batch_sizes) != len(set(args.batch_sizes))
            or any(b < 2 or b > 128 for b in args.batch_sizes)):
        raise ValueError('Choose unique physical batch sizes between 2 and 128')
    # Listing 128 explicitly is equivalent to requesting the gated optional case.
    args.try_128 = args.try_128 or 128 in args.batch_sizes
    args.batch_sizes = sorted(b for b in args.batch_sizes if b != 128)
    if not args.batch_sizes or (args.try_128 and 64 not in args.batch_sizes):
        raise ValueError('Optional batch 128 requires batch 64 in --batch-sizes')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if len(visible.split(',')) != 1 or not visible.strip() or visible.startswith('MIG-'):
        raise ValueError('Set CUDA_VISIBLE_DEVICES to one physical GPU index or GPU UUID')
    if os.environ.get('CUDA_LAUNCH_BLOCKING', '0') != '0':
        raise ValueError('Unset CUDA_LAUNCH_BLOCKING for asynchronous benchmark measurements')
    args.config = args.config.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError(f'Benchmark output already exists: {args.output}')
    for parent in args.output.parents:
        if (parent / 'run.json').exists():
            raise ValueError('Benchmark output must be outside an existing SSL experiment')


def worker(args):
    if args.worker_batch is None:
        raise ValueError('--worker-batch is required with --worker-plan')
    plan = json.loads(args.worker_plan.read_text())
    batch = args.worker_batch
    if batch not in [*plan['batch_sizes'], *([128] if plan['try_128'] else [])]:
        raise ValueError('Worker batch was not requested')
    folder = args.worker_plan.parent / f'batch_{batch:04d}'
    if not folder.is_dir() or (folder / 'result.json').exists():
        raise FileExistsError('Worker requires its new parent-owned output directory')
    result = run_case(lambda progress: measure_case(plan, batch, progress), batch, folder / 'result.json')
    if result['status'] == 'ok':
        # Per-step values remain in result.json; convenient JSONL has measured rows only.
        with (folder / 'steps.jsonl').open('x') as stream:
            for row in result['measured']['rows']:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
    return 0


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        if args.worker_plan:
            return worker(args)
        if args.worker_batch is not None:
            raise ValueError('--worker-batch is internal and requires --worker-plan')
        validate_options(args)
        os.chdir(ROOT)
        cfg = configuration(args.config)
        apply_arguments(cfg, args)
        if args.warmup_steps + args.measured_steps > cfg.experiment.schedule_total_steps:
            raise ValueError('Warmup + measurement exceeds the original experiment schedule horizon')
        if sha256(args.checkpoint) != PIN['backbone_sha256']:
            raise ValueError('Use the original SHA256-pinned Meta release')
        cfg.train.num_workers = args.workers
        args.output.mkdir(parents=True, exist_ok=False)
        print('Auditing and hashing only eligible real NFFA training images; no GPU context in parent.', flush=True)
        if args.inventory:
            from src.ssl_benchmark import dataset_from_inventory
            inventory = json.loads(args.inventory.read_text())
            dataset_from_inventory(cfg, inventory)
        else:
            _, _, inventory = dataset_plan(cfg, args.workers)
        if inventory['count'] < max([*args.batch_sizes, *([128] if args.try_128 else [])]):
            raise ValueError('Not enough eligible original images for requested physical batch')
        inventory_path = args.output / 'dataset_inventory.json'
        atomic_json(inventory_path, inventory)
        (args.output / 'selected_image_ids.txt').write_text(''.join(r['image_id'] + '\n' for r in inventory['records']))
        config_path = args.output / 'config.yaml'
        OmegaConf.save(cfg, config_path)
        provenance = contract(cfg, inventory, args.checkpoint, 'bf16')
        provenance['benchmark_sources'] = {str(path.relative_to(ROOT)): sha256(path) for path in (
            ROOT / 'src/ssl_benchmark.py', ROOT / 'scripts/benchmark_sem_ssl_a100.py')}
        provenance['git_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        plan = dict(schema=1, benchmark_only=True, created_utc=datetime.now(timezone.utc).isoformat(),
                    config=str(config_path), config_sha256=sha256(config_path),
                    checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=PIN['backbone_sha256'],
                    inventory=str(inventory_path), inventory_sha256=sha256(inventory_path),
                    batch_sizes=args.batch_sizes, try_128=args.try_128,
                    warmup_steps=args.warmup_steps, measured_steps=args.measured_steps, workers=args.workers,
                    max_memory_fraction=args.max_memory_fraction, precision='bf16',
                    lr_policy='Original trainer sqrt(batch/1024) scaling; all cases reset schedule index and state',
                    timing='CUDA completion-event step intervals; window wall throughput; host loader blocking reported separately',
                    finite_checks='Deferred until each window boundary; invalid states discarded and never exported',
                    environment={key:os.environ.get(key) for key in ('CUDA_VISIBLE_DEVICES', 'CUDA_LAUNCH_BLOCKING',
                        'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'CUBLAS_WORKSPACE_CONFIG', 'XFORMERS_DISABLED')},
                    provenance=provenance)
        atomic_json(args.output / 'plan.json', plan)

        def stop(_sig, _frame):
            raise KeyboardInterrupt('Benchmark interrupted')

        previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            results, optional = orchestrate(plan, args.output)
        except KeyboardInterrupt:
            atomic_json(args.output / 'interrupted.json', dict(interrupted=True,
                        note='Active child process group terminated; completed cases are in partial_results.json'))
            return 130
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
        report = write_report(args.output, plan, results, optional)
        print(json.dumps(report['recommendation'], indent=2), flush=True)
        print(f'Benchmark outputs: {args.output}', flush=True)
        return 0 if report['recommendation']['batch_size'] is not None else 1
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        p.error(str(error))


if __name__ == '__main__':
    raise SystemExit(main())
