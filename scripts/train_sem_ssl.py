"""Prepare or run bounded SEM continued pretraining; never invokes downstream jobs."""
import argparse
import json
import os
from pathlib import Path
import sys

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(name, '1')
os.environ['XFORMERS_DISABLED'] = '1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omegaconf import OmegaConf
import torch

from src.ssl.checkpoints import atomic_json
from src.ssl.data import sha256
from src.ssl.experiment import REFERENCE, configuration, contract, dataset_plan, step_schedules
from src.ssl.trainer import execute
from src.ssl.upstream import PIN
from src.ssl.loading import add_arguments, apply_arguments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/sem_ssl/nffa_default.yaml'))
    parser.add_argument('--stage', choices=('preflight', 'train'), default='preflight')
    parser.add_argument('--output', type=Path, required=True, help='New run directory; preflight writes a separate JSON path')
    parser.add_argument('--checkpoint', type=Path, default=REFERENCE)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--precision', choices=('fp32', 'bf16'), default='bf16')
    parser.add_argument('--num-workers', '--workers', dest='workers', type=int)
    parser.add_argument('--run-until-step', type=int)
    parser.add_argument('--batch-size', type=int, help='Explicit bounded memory-check override; new run only')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--inventory', type=Path, help='Reuse a verified dataset inventory; no image-content rehashing')
    add_arguments(parser)
    args = parser.parse_args()
    cfg = configuration(args.config)
    apply_arguments(cfg, args)
    if args.workers is not None:
        if args.workers < 0:
            parser.error('workers must be nonnegative')
        cfg.train.num_workers = args.workers
    if args.run_until_step is not None:
        if not 0 <= args.run_until_step <= cfg.experiment.schedule_total_steps:
            parser.error('run-until-step must be within original schedule horizon')
        cfg.experiment.run_until_step = args.run_until_step
    if args.batch_size is not None:
        if args.batch_size < 2:
            parser.error('KoLeo needs at least two originals')
        cfg.train.batch_size_per_gpu = args.batch_size
        cfg.optim.lr = cfg.optim.base_lr * (args.batch_size / 1024) ** 0.5
    if sha256(args.checkpoint) != PIN['backbone_sha256']:
        raise ValueError('Initialization must be the SHA256-pinned original Meta release')
    if args.stage == 'preflight' and args.output.exists():
        raise FileExistsError(args.output)
    if args.stage == 'train':
        if (args.output / 'run.json').exists() and not args.resume:
            raise FileExistsError('Run already exists; use --resume or a new output directory')
        if args.resume and not (args.output / 'run.json').exists():
            raise ValueError('Cannot resume a run without run.json')
        if args.device == 'cpu' and args.precision != 'fp32':
            parser.error('CPU requires --precision fp32')
        if args.device == 'cuda':
            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise RuntimeError('Expose exactly one A100 with CUDA_VISIBLE_DEVICES')
            props = torch.cuda.get_device_properties(0)
            if 'A100' not in props.name or props.total_memory < 75 * 1024**3:
                raise RuntimeError(f'Expected A100 80GB, found {props}')
            if args.precision == 'bf16' and not torch.cuda.is_bf16_supported():
                raise RuntimeError('BF16 is unavailable')
            print(f'GPU={props.name}; torch={torch.__version__}; CUDA={torch.version.cuda}', flush=True)
    if args.inventory:
        from src.ssl_benchmark import dataset_from_inventory
        identity = json.loads(args.inventory.read_text())
        dataset, collate = dataset_from_inventory(cfg, identity)
        print('Reusing verified train inventory; checking manifests/selection, no image-content rehashing.', flush=True)
    else:
        print('Selecting and hashing only eligible NFFA training files...', flush=True)
        dataset, collate, identity = dataset_plan(cfg, cfg.train.num_workers)
    print(f"Eligible originals: {identity['count']}; records SHA256={identity['records_sha256']}", flush=True)
    if args.stage == 'preflight':
        schedules = step_schedules(cfg)
        atomic_json(args.output, dict(passed=True, selected_train_images=identity['count'], dataset=identity,
            config=OmegaConf.to_container(cfg, resolve=True), contract=contract(cfg, identity, args.checkpoint, args.precision),
            schedules={key: dict(length=len(value.schedule), first=float(value[0]),
                last=float(value[cfg.experiment.schedule_total_steps - 1])) for key, value in schedules.items()},
            gpu_checked=False, augmentation='Original Meta DataAugmentationDINO; crop_metadata_strip + RGB only'))
    else:
        execute(cfg, dataset, collate, identity, args.checkpoint, args.output,
                device=args.device, precision=args.precision, resume=args.resume)


if __name__ == '__main__':
    main()
