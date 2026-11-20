"""Short, explicitly launched ViT-S regularization study on the approved NFFA inventory."""

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

from scripts.run_sem_cellstyle_regularization_grid import read, write_config
from scripts.run_sem_ssl_ablations import run_command
from src.sem_cellstyle.data import provider
from src.sem_cellstyle.recipe import contract
from src.sem_cellstyle.regularization import drift_by_block
from src.sem_cellstyle.regularization_report import differences, scalar_metrics
from src.sem_cellstyle.regularization_study import (
    ARMS, arm_config, calibrate, common_identity, require_calibration,
)
from src.sem_cellstyle.runner import execute, loader, require_a100
from src.ssl.adapter import load_config
from src.ssl.checkpoints import atomic_json, run_lock, validate_snapshot
from src.ssl.data import sha256
from src.ssl.experiment import digest
from src.ssl.reporting import result_bundle
from src.ssl.upstream import PIN, ROOT

CONFIG = ROOT / 'configs/sem_ssl/sem_regularization_vits14_500step.yaml'
ENCODER = 'dinov2_vits14'
DRIFT_UPDATES = 20
PROBES = 5


def configuration(output):
    cfg = load_config(CONFIG)
    cfg.optim.lr = cfg.optim.backbone_lr
    dataset, collate, identity = provider(cfg)
    if len(dataset) != cfg.dataset.target_train_originals:
        raise ValueError('Approved NFFA train count differs from the pilot recipe')
    initialization = Path(cfg.initialization.checkpoint).resolve()
    if sha256(initialization) != cfg.initialization.sha256 or cfg.initialization.sha256 != PIN['backbone_sha256']:
        raise ValueError('Pilot requires the pinned original Meta ViT-S/14 checkpoint')
    # The common runner reports review flags against whole-corpus passes.
    # These Large review points are beyond this pilot's 500-update horizon.
    cfg.experiment.pass_to_step = {
        str(p): math.ceil(p * len(dataset) / cfg.runtime.effective_batch)
        for p in (10, 25, 50, 100)
    }
    if output is not None:
        cfg.runtime.output = str(output.resolve())
    return cfg, dataset, collate, identity, initialization


def verify_arm(run, study_contract, initialization_sha, milestones):
    if read(run / 'run.json')['contract'] != study_contract:
        raise ValueError(f'Training contract differs in {run}')
    for step in milestones:
        path = run / 'checkpoints' / f'step_{step:06d}'
        meta = validate_snapshot(path)
        verification = read(path / 'feature_verification.json')
        if (meta['contract_sha256'] != digest(study_contract)
                or verification['step'] != step or verification['live_teacher_feature_max_abs_error'] != 0):
            raise ValueError(f'Checkpoint identity/feature verification differs in {path}')
        if step == 0 and (meta['files']['teacher_backbone.pth'] != initialization_sha
                          or not verification['original_step_zero_tensor_equality']):
            raise ValueError(f'{run} did not start from the original Meta weights')
    return path


def draws(run, steps, effective_batch):
    with (run / 'metrics.jsonl').open() as stream:
        history = [json.loads(line) for line in stream if line.strip()]
    if len(history) != steps:
        raise ValueError(f'Incomplete update history in {run}')
    for step, row in enumerate(history, 1):
        expected = list(range((step - 1) * effective_batch, step * effective_batch))
        if row['step'] != step or row['draw_ids'] != expected or len(row['record_indices']) != effective_batch:
            raise ValueError(f'Draw cursor differs in {run} at step {step}')
    return [row['record_indices'] for row in history]


def train_arms(root, base, dataset, collate, identity, initialization):
    calibration_file = root / 'calibration.json'
    if not calibration_file.exists():
        print(f'Calibration: two disposable trajectories with {DRIFT_UPDATES} drift updates '
              f'and {PROBES} measured updates each', flush=True)
        calibrate(base, dataset, collate, identity, initialization, calibration_file,
                  warmup=DRIFT_UPDATES, probes=PROBES)
    coefficients = read(calibration_file)['coefficients']
    print(f'Frozen regularization coefficients: {coefficients}', flush=True)
    steps = base.experiment.run_until_step
    checkpoints, common_draws = {}, None
    for arm in ARMS:
        run = root / 'runs' / arm
        cfg = arm_config(base, arm, run, coefficients, calibration_file)
        require_calibration(cfg, identity, initialization)
        write_config(root / 'configs' / f'{arm}.yaml', cfg)
        expected = contract(cfg, identity, initialization, 'cuda')
        resume = (run / 'run.json').exists()
        status_file = run / 'status.json'
        complete = {'status': 'stage_complete', 'step': steps}
        status = read(status_file) if status_file.exists() else {}
        if any(status.get(key) != value for key, value in complete.items()):
            print(f'Arm {arm}, {steps} optimizer updates, resume={resume}', flush=True)
            execute(cfg, dataset, collate, identity, initialization, run, resume=resume)
        status = read(status_file)
        if any(status.get(key) != value for key, value in complete.items()):
            raise InterruptedError(f'Arm {arm} stopped early. Repeat the command to resume.')
        checkpoints[arm] = verify_arm(run, expected, base.initialization.sha256, cfg.experiment.milestones)
        indices = draws(run, steps, cfg.runtime.effective_batch)
        if common_draws is None:
            common_draws = indices
        elif indices != common_draws:
            raise ValueError(f'Shared training draw order differs in arm {arm}')
    return checkpoints, coefficients


def evaluation_command(cfg, checkpoint, output, task):
    classification = task == 'classification'
    heads = ['logreg', 'mlp'] if classification else ['linear', 'cnn8']
    workers = cfg.evaluation[f'{task}_workers']
    # Full-corpus V0 checkpoints still use the pilot's verified NFFA evaluation data.
    data = load_config(CONFIG).dataset if cfg.dataset.get('provider') == 'frozen_sem_posttraining_v0' else cfg.dataset
    command = [sys.executable, '-m', 'scripts.evaluate_sem_encoder', '--encoder-name', ENCODER,
               '--encoder-checkpoint', str(checkpoint), '--output-root', str(output),
               '--head-profile', cfg.evaluation.head_profile, '--task', task, '--heads', *heads,
               '--splits', 'A', '--seed', str(cfg.runtime.seed), '--workers', str(workers), '--device', 'cuda:0',
               '--verified-data-root', str(Path(cfg.evaluation.verified_data_root).resolve()),
               '--classification-manifest', data.manifest, '--split-a', data.primary_split,
               '--image-root', data.root, '--stage', 'all', '--resume']
    return command


def prepare_original_caches(cfg, initialization, output):
    """Link complete original-S features when the existing cache reader accepts their exact contract."""
    from scripts.evaluate_sem_encoder import identity_and_root, parser
    from src.aachen_ablation.cache import cache_contract, source_hashes
    from src.aachen_ablation.data import manifest_rows
    from src.aachen_ablation.geometry import VARIANTS
    from src.downstream.benchmark import feature_contract
    from src.ssl.ablation_data import classification, segmentation, segmentation_reference

    commands = evaluation_command(cfg, initialization, output, 'classification')
    args = parser().parse_args(commands[3:])
    _, identity, base = identity_and_root(args)
    prior = args.verified_data_root
    split = Path('classification/split_A')
    cls_contract = feature_contract(
        classification(prior / split / 'preflight.json', args.classification_manifest, args.split_a, args.image_root),
        args.classification_manifest, args.split_a, args.image_root, identity, False)
    variant = VARIANTS['A3']
    manifest = args.segmentation_manifest.resolve()
    records, _ = segmentation(segmentation_reference(prior), manifest, manifest_rows(manifest), [variant])
    seg_contract = cache_contract(manifest, records, variant, 'float32', initialization, False, ENCODER)
    seg_contract.update(encoder_source='pinned Meta checkout', upstream_commit=PIN['commit'],
                        pinned_encoder_sources={k: v for k, v in source_hashes().items()
                                                if k in ('src/aachen_ablation/cache.py', 'src/downstream/pinned_encoder.py')})
    candidates = [(prior / split / 'features', cls_contract)]
    candidates.extend((p.parent, seg_contract) for p in sorted(prior.glob('segmentation/caches/A3_*/cache_metadata.json')))
    linked = []
    for cache, expected in candidates:
        meta = cache / 'cache_metadata.json'
        if meta.is_file():
            saved = read(meta)
            if saved.get('complete') and saved.get('contract') == expected:
                target = base / cache.relative_to(prior)
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists() and not target.is_symlink():
                    target.symlink_to(cache, target_is_directory=True)
                if target.resolve() == cache.resolve():
                    linked.append(str(cache))
    return linked


def evaluate(root, cfg, initialization, checkpoints):
    results, protocols = {}, {}
    encoders = {'original': initialization} | {arm: folder / 'teacher_backbone.pth' for arm, folder in checkpoints.items()}
    for name, checkpoint in encoders.items():
        output = root / 'evaluation' / name
        checksum = sha256(checkpoint)
        base = output / f'{ENCODER}_{checksum[:16]}'
        bundle, states = {}, set()
        for task in ('classification', 'segmentation'):
            options = dict(classification_splits=('A',), head_profile=cfg.evaluation.head_profile)
            metrics, protocol, missing, fingerprints = result_bundle(base, checksum, task, **options)
            if missing:
                run_command(evaluation_command(cfg, checkpoint, output, task), root / 'logs' / f'{name}_{task}.log')
                metrics, protocol, missing, fingerprints = result_bundle(base, checksum, task, **options)
            if missing:
                raise ValueError(f'Missing {name} validation artifacts: {missing}')
            for key, value in protocol.items():
                if protocols.setdefault(key, value) != value:
                    raise ValueError(f'Validation protocol differs for {name}: {key}')
            bundle.update(metrics)
            states.update(fingerprints)
        if len(states) != 1:
            raise ValueError(f'Downstream backbones differ for {name}')
        results[name] = scalar_metrics(bundle)
    return results


def diagnostics(cfg, dataset, collate, initialization, checkpoints):
    from src.downstream.pinned_encoder import get_pinned_dinov2

    original = torch.load(initialization, map_location='cpu', weights_only=True, mmap=True)
    batch = next(iter(loader(cfg, dataset, collate, 'cuda', 0, 1)))
    views = batch['collated_global_crops'].to('cuda')
    keys = {'cls_cosine': 'x_norm_clstoken', 'patch_cosine': 'x_norm_patchtokens'}

    @torch.inference_mode()
    def features(weights):
        return get_pinned_dinov2(weights, ENCODER).to('cuda')(views, is_training=True)

    reference = features(initialization)
    result = {}
    for arm, folder in checkpoints.items():
        state = torch.load(folder / 'training.pth', map_location='cpu', weights_only=False, mmap=True)['model']
        teacher = features(folder / 'teacher_backbone.pth')
        result[arm] = {name: float(F.cosine_similarity(teacher[key], reference[key], dim=-1).clamp(-1, 1).mean())
                       for name, key in keys}
        names = read(folder.parents[1] / 'run.json')['backbone_parameter_names']
        for model in ('student', 'teacher'):
            pairs = ((name, state[f'{model}.backbone.{name}'], original[name]) for name in names)
            result[arm][f'{model}_relative_l2_by_block'] = {key: float(value) for key, value in drift_by_block(pairs).items()}
    return dict(training_draw_ids=batch['draw_ids'], record_indices=batch['record_indices'],
                views='Fixed training globals, CLS and spatially aligned patch tokens, FP32', arms=result)


def write_report(root, metrics, diagnostic, coefficients):
    comparisons = [
        dict(arm=arm, metric=name, **differences(value, metrics['original'][name], metrics['A'][name]))
        for arm, scores in metrics.items() for name, value in scores.items()
    ]
    destination = root / 'report'
    destination.mkdir(exist_ok=True)
    temp = destination / 'comparison.partial.csv'
    with temp.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    temp.replace(destination / 'comparison.csv')
    atomic_json(destination / 'comparison.json', dict(
        reference='Original Meta DINOv2 ViT-S/14', validation_only=True, seed=42, steps=500,
        coefficients=coefficients, metrics=metrics, differences=comparisons, diagnostics=diagnostic))
    print(f'Comparison: {destination / "comparison.csv"}\nFeature similarity and drift: {destination / "comparison.json"}',
          flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', action='store_true', help='Verify pretrained weights and approved inventory on CPU')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    base, dataset, collate, identity, initialization = configuration(args.output)
    study = dict(recipe=base.recipe, shared_training_identity=common_identity(base, identity, initialization),
                 originals=len(dataset),
                 effective_batch=base.runtime.effective_batch, physical_batch=base.runtime.physical_batch,
                 schedule_total_steps=base.experiment.schedule_total_steps, drift_updates=DRIFT_UPDATES,
                 probes=PROBES, runner_sha256=sha256(Path(__file__)),
                 arms={name: dict(l2_sp=sp, feature_distillation=feat) for name, (sp, feat) in ARMS.items()})
    if args.plan:
        print(json.dumps(study, indent=2))
        return
    require_a100()
    root = Path(base.runtime.output).resolve()
    with run_lock(root):
        path = root / 'study.json'
        if path.exists():
            if read(path) != study:
                raise ValueError('Study recipe changed. Use a new output directory.')
        else:
            if any(p.name != '.writer.lock' for p in root.iterdir()):
                raise FileExistsError('Use an empty study directory')
            atomic_json(path, study)
        write_config(root / 'configs' / 'base.yaml', base)
        cached = prepare_original_caches(base, initialization, root / 'evaluation' / 'original')
        print(f'Compatible original feature caches: {cached}', flush=True)
        checkpoints, coefficients = train_arms(root, base, dataset, collate, identity, initialization)
        metrics = evaluate(root, base, initialization, checkpoints)
        write_report(root, metrics, diagnostics(base, dataset, collate, initialization, checkpoints), coefficients)


if __name__ == '__main__':
    main()
