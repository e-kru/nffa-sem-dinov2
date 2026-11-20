"""Bounded A100 resume diagnostic: 2 warmups + 2 comparisons, never a study run."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import time
import traceback

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '1')
os.environ['XFORMERS_DISABLED'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@contextmanager
def deterministic_probe(enabled):
    """Diagnostic override only; restore process settings even if an op rejects it."""
    if not enabled:
        yield
        return
    import torch
    previous = (torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled(),
                torch.backends.cudnn.deterministic)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])
        torch.backends.cudnn.deterministic = previous[2]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--physical-batch', type=int, choices=(16, 32), default=16)
    parser.add_argument('--warmup-updates', type=int, choices=range(1, 11), default=2)
    parser.add_argument('--deterministic', action='store_true',
                        help='Diagnostic only: require deterministic operations; retain unsupported-op errors')
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError('Use a new diagnostic output; historical results are preserved')
    if args.deterministic:
        # Set before importing Torch or initializing CUDA. Do not override an
        # explicit caller choice; PyTorch reports incompatible settings itself.
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    from omegaconf import OmegaConf
    import torch
    from src.sem_cellstyle.recipe import configuration, require_initialization, resolve, schedules, contract
    from src.sem_cellstyle.data import provider
    from src.sem_cellstyle.runner import require_a100, initialize, loader, verify_export
    from src.sem_cellstyle.training import accumulated_step
    from src.sem_cellstyle.resume_check import verify_resume_update, ResumeCheckError
    from src.ssl.checkpoints import atomic_json, save_snapshot
    cfg = configuration(args.config)
    dataset, collate, identity = provider(cfg)
    require_a100()
    initialization = require_initialization(cfg)
    cfg = resolve(cfg, len(dataset), args.physical_batch)
    cfg.runtime.output = str(args.output.absolute())
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output/'plan.json', dict(physical_batch=args.physical_batch, warmup_updates=args.warmup_updates,
        comparison_updates=2, total_optimizer_updates=args.warmup_updates+2,
        scientific_training_started=False, weights_promoted=False,
        diagnostic_deterministic_override=args.deterministic,
        cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        contract=contract(cfg, identity, initialization, 'cuda')))
    stage = 'initialization'
    report = None
    try:
        model, optimizer = initialize(cfg, initialization, 'cuda')
        # initialize deliberately installs the historical nondeterministic
        # recipe. Apply the opt-in diagnostic override afterwards, covering
        # warmup, backward, checkpoint verification and BOTH comparison paths.
        with deterministic_probe(args.deterministic):
            print(json.dumps(dict(stage='numerics', deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                                  cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))), flush=True)
            sched = schedules(cfg)
            stage = 'warmup'
            batches = iter(loader(cfg, dataset, collate, 'cuda', 0, args.warmup_updates))
            for step in range(args.warmup_updates):
                start = time.perf_counter()
                row = accumulated_step(model, optimizer, batches, sched, step)
                print(json.dumps(dict(stage='warmup', step=step+1, loss=row['total_loss'],
                                      wall_seconds=time.perf_counter()-start)), flush=True)
            del batches
            stage = 'checkpoint_and_export'
            print('Saving DISPOSABLE diagnostic state and verifying teacher export', flush=True)
            path = save_snapshot(args.output, model, optimizer, args.warmup_updates, OmegaConf.to_container(cfg),
                                 contract(cfg, identity, initialization, 'cuda'), sched, initialization, milestone=False)
            verify_export(path, model, initialization)
            payload = torch.load(path/'training.pth', map_location='cpu', weights_only=False)
            stage = 'resume_comparison'
            print('Comparing uninterrupted/restored update with exact inputs, RNG and tensor diagnostics', flush=True)
            report = verify_resume_update(model, optimizer, dataset, collate, sched, args.warmup_updates,
                                          payload, args.output/'resume_diagnostics.json')
    except ResumeCheckError as error:
        report = error.report
    except RuntimeError as error:
        # Unsupported deterministic CUDA backwards must name the operator in a
        # durable report, rather than disappearing before result.json exists.
        details = dict(status='runtime_error', phase=stage, error_type=type(error).__name__,
                       error=str(error), traceback=traceback.format_exc(),
                       diagnostic_deterministic_override=args.deterministic)
        atomic_json(args.output/'runtime_error.json', details)
        value = {key: details[key] for key in ('status', 'phase', 'error_type', 'error')}
        value['report'] = str(args.output/'runtime_error.json')
    if report is not None:
        value = dict(status='passed' if report['passed'] else 'resume_error', checks=report['checks'],
                     model_difference=report['model_difference'], optimizer_difference=report['optimizer_difference'],
                     report=str(args.output/'resume_diagnostics.json'))
    value.update(scientific_training_started=False, weights_promoted=False,
                 diagnostic_deterministic_override=args.deterministic,
                 cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                 checkpoint_role='DISPOSABLE_DIAGNOSTIC_ONLY')
    atomic_json(args.output/'result.json', value)
    print(json.dumps(value, indent=2), flush=True)
    return 0 if value['status'] == 'passed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
