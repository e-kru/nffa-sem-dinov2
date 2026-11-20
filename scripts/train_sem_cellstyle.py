"""Prepare the opt-in Large SEM recipe. Default action is a dataset-free plan."""
import argparse
import json
import os
from pathlib import Path
import sys

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(name, '1')
os.environ['XFORMERS_DISABLED'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omegaconf import OmegaConf
import torch

from src.sem_cellstyle.recipe import DEFAULT_CONFIG, budgets, configuration, require_initialization, resolve
from src.sem_cellstyle.data import provider
from src.ssl.checkpoints import atomic_json


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    p.add_argument('--stage', choices=('plan', 'preflight', 'preview', 'train', 'verify'), default='plan')
    p.add_argument('--example-originals', type=int, help='Planning only, never substitutes for a verified corpus')
    p.add_argument('--physical-batch', type=int, choices=(16, 32))
    p.add_argument('--benchmark-result', type=Path, help='Required for scientific training; Large full-step benchmark.json')
    p.add_argument('--num-workers', type=int)
    p.add_argument('--run-until-step', type=int, help='Only the stopping boundary changes, preserving the full schedule')
    p.add_argument('--output', type=Path)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--preview-count', type=int, default=3)
    return p


def main(argv=None):
    p = arguments(); args = p.parse_args(argv)
    cfg = configuration(args.config)
    if args.example_originals is not None and args.stage != 'plan':
        p.error('--example-originals is planning only')
    if args.stage == 'plan':
        plan = dict(recipe=cfg.recipe, scientific_run_ready=False, dataset_ready=False,
                    required=['verified training manifest + receipt', 'original Meta ViT-L/14 weights + trusted SHA256',
                              'real-data Large A100 benchmark; fixed physical batch', 'bounded resume/export check'],
                    full_horizon_passes=cfg.budget.full_original_presentation_passes,
                    initial_stop_passes=cfg.budget.initial_run_until_passes,
                    physical_batch=cfg.runtime.physical_batch, target_train_originals=cfg.dataset.target_train_originals,
                    feasible_first_corpus_originals=cfg.dataset.feasible_first_corpus_originals, feature_dim=1024,
                    global_tokens=1369, local_tokens=256, training_started=False)
        if args.example_originals:
            plan['planning_example_only'] = budgets(cfg, args.example_originals)
        print(json.dumps(plan, indent=2)); return
    if args.num_workers is not None:
        if args.num_workers < 0:
            p.error('num-workers must be nonnegative')
        cfg.train.num_workers = args.num_workers
    dataset, collate, identity = provider(cfg)
    physical = args.physical_batch or cfg.runtime.physical_batch
    cfg = resolve(cfg, len(dataset), physical)
    if args.run_until_step is not None:
        if not 0 <= args.run_until_step <= cfg.experiment.schedule_total_steps:
            p.error('run-until-step outside the full schedule')
        cfg.experiment.run_until_step = args.run_until_step
    if args.stage == 'preview':
        from torchvision.utils import save_image
        from src.ssl.stream import data_rng
        output = args.output
        if output is None:
            p.error('preview requires a new --output directory')
        output.mkdir(parents=True, exist_ok=False)
        metadata = []
        for i in range(min(args.preview_count, len(dataset))):
            with data_rng(42+i):
                sample, _ = dataset[i]
            for kind in ('global_crops', 'local_crops'):
                for j, value in enumerate(sample[kind]):
                    value = value*torch.tensor([.229,.224,.225])[:,None,None]+torch.tensor([.485,.456,.406])[:,None,None]
                    save_image(value.clamp(0,1), output / f'{i:03d}_{kind}_{j}.png')
            metadata.append(dataset.records[i])
        atomic_json(output / 'preview.json', dict(records=metadata, data=identity))
        return
    initialization = require_initialization(cfg)
    output = args.output or Path(cfg.runtime.output)
    if args.stage == 'preflight':
        if args.output is None or args.output.exists():
            p.error('preflight requires a new --output JSON path')
        atomic_json(output, dict(config=OmegaConf.to_container(cfg, resolve=True), dataset=identity,
                    initialization=str(initialization), gpu_checked=False, physical_batch_benchmarked=False))
        return
    from src.sem_cellstyle.runner import execute, initialize, require_a100, verify_export
    require_a100()
    if args.stage == 'verify':
        from src.ssl.checkpoints import find_latest
        _, payload, _ = find_latest(output)
        model, _ = initialize(cfg, initialization, 'cuda')
        model.load_state_dict(payload['model'], strict=True)
        results = []
        for folder in sorted((output / 'checkpoints').glob('step_*')):
            state = torch.load(folder/'training.pth', map_location='cpu', weights_only=False)
            model.load_state_dict(state['model'], strict=True)
            if hasattr(model,'regularization'):
                model.regularization.verify_original(torch.load(initialization,map_location='cpu',weights_only=True,mmap=True))
            results.append(verify_export(folder, model, initialization))
        print(json.dumps(results, indent=2)); return
    if args.benchmark_result is None:
        p.error('Scientific training requires --benchmark-result from the real Large full-step benchmark')
    from src.sem_cellstyle.regularization_study import require_calibration
    require_calibration(cfg, identity, initialization)
    from src.sem_cellstyle.benchmark import validate_choice
    validate_choice(json.loads(args.benchmark_result.read_text()), cfg, identity, initialization)
    execute(cfg, dataset, collate, identity, initialization, output, resume=args.resume)


if __name__ == '__main__':
    main()
