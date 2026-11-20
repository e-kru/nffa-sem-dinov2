"""Run the ViT-S augmentation ablation through the existing 500-update pilot."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import random
import statistics

from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision.transforms import functional as TF

from scripts import run_sem_regularization_vits14 as pilot
from src import sem_augmentation
from src.sem_augmentation import FactorialViews, GEOMETRY, NOISE, training_dataset
from src.sem_cellstyle.recipe import contract
from src.sem_cellstyle.regularization import options
from src.sem_cellstyle.regularization_report import scalar_metrics
from src.sem_cellstyle.runner import execute, require_a100
from src.ssl.ablation_data import segmentation_reference
from src.ssl.checkpoints import atomic_json, run_lock
from src.ssl.experiment import digest
from src.ssl.loading import load_image
from src.ssl.reporting import result_bundle
from src.ssl.stream import data_rng, draw_seed
from src.ssl.upstream import ROOT


OUTPUT = ROOT / 'outputs/sem_ssl_augmentation/nffa_split_a_vits14_500step_rotation_trial_v2_seed42'
BASELINE_OUTPUT = ROOT / 'outputs/sem_ssl_augmentation/nffa_split_a_vits14_500step_aug_a_v1_seed42'
POLICY = ROOT / 'configs/sem_ssl/nffa_augmentation_rotation_trial.json'
HISTORICAL = ROOT / 'outputs/sem_ssl_regularization/nffa_split_a_vits14_500step_v1_seed42'
ARMS = {'Aug-A': (False, False), 'Aug-B': (True, False), 'Aug-C': (False, True), 'Aug-D': (True, True)}
# The prior runner used the same recipe for explicit ablation stages.
PREVIOUS_RUNNER_SHA256 = 'ec967f99f38455a2fd2caaa87ea9b19dbea338a46eb53c37ed32bc496dd57c41'


def geometry_policy(path, train_ids):
    policy = pilot.read(path)
    if policy['source'] != 'nffa_sem_100' or not policy['rationale']:
        raise ValueError('Expected a documented NFFA geometry policy')
    if policy['schema'] == 'nffa_split_a_rotation_trial_v1':
        if policy['scope'] != 'all_verified_split_a_train':
            raise ValueError('Rotation trial must use only the verified Split A training inventory')
        return policy, frozenset(train_ids)
    if policy['schema'] != 'nffa_split_a_geometry_allowlist_v1' or not policy['evidence']:
        raise ValueError('Geometry requires an evidence-backed NFFA source policy')
    allowed = set()
    for group in policy['allow']:
        ids = set(group['image_ids'])
        if (not group['source_id'] or not group['evidence'] or not group['rationale'] or not ids
                or len(ids) != len(group['image_ids']) or allowed & ids or not ids <= train_ids):
            raise ValueError('Geometry decisions must cite a source and refer to unique approved Split A train IDs')
        allowed.update(ids)
    return policy, frozenset(allowed)


def audit_training(cfg, identity, records):
    ids = {row['image_id'] for row in records}
    groups = {row['group_id'] for row in records}
    if len(ids) != cfg.dataset.target_train_originals or cfg.dataset.selection_protocol != 'split_A_train_v1':
        raise ValueError('Expected the approved Split A SSL inventory')
    with Path(cfg.dataset.primary_split).open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['split'] != 'train' and (row['image_id'] in ids or row['group_id'] in groups):
                raise ValueError('Split A held-out group/image overlaps the training inventory')
    # Only split assignment metadata is read. The runner always evaluates A.
    with (Path(cfg.dataset.primary_split).parent / 'split_B_full.csv').open(newline='') as stream:
        split_b_val_overlap = sum(row['split'] == 'val' and row['image_id'] in ids for row in csv.DictReader(stream))
    aachen = pilot.read(segmentation_reference(cfg.evaluation.verified_data_root))['cache_metadata']['contract']
    ssl_hashes = {row['sha256'] for row in identity['audited_inventory']['records']}
    if any(row['image_sha256'] in ssl_hashes for row in aachen['records']):
        raise ValueError('Exact image overlap between NFFA SSL and Aachen A3 inventory')
    return ids, dict(nffa_train=len(ids), nffa_train_groups=len(groups), split_a_heldout_overlap=0,
                     split_b_validation_overlap=split_b_val_overlap, aachen_a3_exact_overlap=0)


def scientific_identity(value):
    normalized = json.loads(json.dumps(value))
    cfg = normalized['configuration']
    regularization = options(cfg)
    if regularization['l2_sp'] or regularization['feature_distillation']:
        raise ValueError('Augmentation experiments require unregularized SSL')
    cfg.pop('regularization', None)
    cfg['runtime'].pop('output')
    return normalized


def historical_control(base, identity, initialization):
    run = HISTORICAL / 'runs/A'
    if not (run / 'status.json').is_file():
        return None, 'Previous unregularized control unavailable'
    try:
        expected = contract(base, identity, initialization, 'cuda')
        saved = pilot.read(run / 'run.json')['contract']
        if scientific_identity(saved) != scientific_identity(expected):
            return None, 'Previous control differs in its scientific training contract'
        status = pilot.read(run / 'status.json')
        if status['status'] != 'stage_complete' or status['step'] != base.experiment.run_until_step:
            return None, 'Previous control is incomplete'
        pilot.draws(run, base.experiment.run_until_step, base.runtime.effective_batch)
        pilot.verify_arm(run, saved, base.initialization.sha256, base.experiment.milestones)
    except (ValueError, RuntimeError, KeyError, OSError) as error:
        return None, str(error)
    return run, 'Exact unregularized training contract, draw cursor and checkpoints verified'


def arm_config(base, run, noise_enabled, geometry_enabled, allowed):
    cfg = OmegaConf.create(OmegaConf.to_container(base, resolve=True))
    cfg.runtime.output = str(run)
    cfg.augmentation_ablation = dict(
        noise=NOISE if noise_enabled else None,
        geometry=GEOMETRY if geometry_enabled else None,
        geometry_image_count=len(allowed),
        geometry_ids_sha256=digest(sorted(allowed)),
        implementation_sha256=pilot.sha256(Path(sem_augmentation.__file__)),
        rng='per_draw_torch_initial_seed_then_per_crop_domain_sha256_v1',
    )
    return cfg


def visualize(cfg, dataset, allowed, destination):
    """Paired baseline/extras on approved train images, with explicit preview-only geometry."""
    destination.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    if allowed and len(allowed) < len(dataset):
        candidates = sorted(allowed)
        selected = rng.sample(candidates, min(12, len(candidates)))
    else:
        selected = []
        for group in ('Films_Coated_Surface', 'Particles', 'Porous_Sponge', 'Powder',
                      'Nanowires', 'Patterned_surface'):
            candidates = [r['image_id'] for r in dataset.records if r['image_id'].split('/')[0] == group]
            selected.extend(rng.sample(candidates, 2))
    regular = FactorialViews(cfg, False)
    noised = FactorialViews(cfg, True)
    font = ImageFont.truetype('DejaVuSans.ttf', 15)
    title = ImageFont.truetype('DejaVuSans.ttf', 22)
    layout = [(regular.global_size, regular.global_scale)] * 2 + [(regular.local_size, regular.local_scale)]
    receipts = []
    for crop_index, name in ((0, 'global_224'), (2, 'local_98')):
        for start in range(0, len(selected), 6):
            group = selected[start:start+6]
            board = Image.new('RGB', (1180, 86+180*len(group)), '#f0f3f6')
            draw = ImageDraw.Draw(board)
            heading = 'NFFA Split A Train | Rotationstest' if allowed else 'NFFA Split A Train | Drehung nur Vorschau'
            draw.text((10, 8), f'{heading} | {name}', fill='#172336', font=title)
            draw.text((10, 39), 'Gepaarter Crop und gleiche Photometrie. Ausgewählt sind Beispiele mit Noise und sichtbarer Drehung.',
                      fill='#27364c', font=font)
            for column, label in enumerate(('Original (nach Strip-Crop)', 'Aug-A · Baseline',
                                            'Aug-B · Noise', 'Aug-C · Drehung', 'Aug-D · Beides')):
                draw.text((10+234*column, 62), label, fill='#172336', font=font)
            for offset, image_id in enumerate(group):
                original = load_image(dataset.root / image_id).convert('L')
                for trial in range(200):
                    seed = draw_seed(42, f'preview-{name}-{start+offset}', trial)
                    # Choose an illustrative draw, rather than estimating the
                    # activation probability from the contact sheet.
                    neutral = torch.full((1, 2, 2), .5)
                    _, ops, transform = noised.augment_gray(neutral, seed, crop_index, True)
                    if ops and transform:
                        break
                else:
                    raise ValueError('No illustrative draw for the paired preview')
                with data_rng(seed):
                    for size, scale in layout[:crop_index+1]:
                        gray = regular.gray_view(original, size, scale)
                b = noised.augment_gray(gray, seed, crop_index, False)[0]
                c = regular.augment_gray(gray, seed, crop_index, True)[0]
                d = noised.augment_gray(gray, seed, crop_index, True)[0]
                row = [original] + [TF.to_pil_image((x*255).round().to(torch.uint8)) for x in (gray,b,c,d)]
                for column, image in enumerate(row):
                    if column and name == 'local_98':
                        image = image.resize((151, 151), Image.Resampling.NEAREST)
                    image.thumbnail((220, 151), Image.Resampling.LANCZOS)
                    board.paste(image, (10+234*column+(220-image.width)//2, 86+180*offset))
                draw.text((10, 242+180*offset), f'{start+offset+1:02d} · {image_id.split("/")[0]}',
                          fill='#172336', font=font)
                draw.text((10+234*2, 242+180*offset), '+'.join(ops), fill='#172336', font=font)
                draw.text((10+234*3, 242+180*offset), f'{transform}°',
                          fill='#172336', font=font)
                receipts.append(dict(image_id=image_id, crop=name, draw_seed=seed, operations=ops,
                                     rotation_degrees=transform))
            board.save(destination / f'{name}_{start//6+1:02d}.jpg', quality=93)
    atomic_json(destination / 'samples.json', dict(geometry_preview_only=not bool(allowed), samples=receipts))


def fit_arms(root, base, baseline, collate, identity, initialization, allowed, labels, reused):
    runs, checkpoints, order = {}, {}, None
    for label in labels:
        noise, geometry = ARMS[label]
        if geometry and not allowed:
            raise ValueError('Geometry arms require a nonempty supported allowlist')
        run = reused if label == 'Aug-A' and reused else root / 'runs' / label
        if run == reused:
            recipe = pilot.read(run / 'run.json')['contract']
        else:
            cfg = arm_config(base, run, noise, geometry, allowed if geometry else ())
            pilot.write_config(root / 'configs' / f'{label}.yaml', cfg)
            recipe = contract(cfg, identity, initialization, 'cuda')
            status_path = run / 'status.json'
            resume = (run / 'run.json').is_file()
            if resume and pilot.read(run / 'run.json')['contract'] != recipe:
                raise ValueError(f'{label} checkpoint contract changed')
            status = pilot.read(status_path) if status_path.is_file() else {}
            if status.get('status') != 'stage_complete':
                data = training_dataset(cfg, baseline, add_noise=noise, geometry_ids=allowed if geometry else ())
                print(f'{label} on one A100, resume={resume}', flush=True)
                execute(cfg, data, collate, identity, initialization, run, resume=resume)
            status = pilot.read(status_path)
            if status['status'] != 'stage_complete' or status['step'] != base.experiment.run_until_step:
                raise InterruptedError(f'{label} stopped early. Repeat the command to resume.')
        folders = pilot.verify_arm(run, recipe, base.initialization.sha256, base.experiment.milestones)
        draws = pilot.draws(run, base.experiment.run_until_step, base.runtime.effective_batch)
        if order is not None and draws != order:
            raise ValueError(f'SSL data order differs in {label}')
        order = draws
        checkpoints[label] = folders / 'teacher_backbone.pth'
        runs[label] = run
    return runs, checkpoints


def evaluate(root, cfg, checkpoints, initialization, reused):
    encoders = {'Original': initialization} | checkpoints
    results, protocols = {}, {}
    for label, checkpoint in encoders.items():
        checksum = pilot.sha256(checkpoint)
        suffix = f'{pilot.ENCODER}_{checksum[:16]}'
        previous = {'Original': 'original', 'Aug-A': 'A' if reused else None}.get(label)
        output = root / 'evaluation' / label
        metrics, weights = {}, set()
        for task in ('classification', 'segmentation'):
            params = dict(classification_splits=('A',), head_profile=cfg.evaluation.head_profile)
            reference = HISTORICAL / 'evaluation' / previous / suffix if previous else output / suffix
            try:
                result, contract_, missing, hashes = result_bundle(reference, checksum, task, **params)
            except (ValueError, KeyError, OSError):
                if not previous:
                    raise
                missing = ['historical evaluation incompatible']
            if missing:
                pilot.run_command(pilot.evaluation_command(cfg, checkpoint, output, task),
                                  root / 'logs' / f'{label}_{task}.log')
                result, contract_, missing, hashes = result_bundle(output / suffix, checksum, task, **params)
            if missing:
                raise ValueError(f'{label} evaluation incomplete: {missing}')
            for key,value in contract_.items():
                if protocols.setdefault(key, value) != value:
                    raise ValueError(f'Mismatched evaluation protocol for {label}: {key}')
            metrics.update(result)
            weights.update(hashes)
        if len(weights) != 1:
            raise ValueError(f'Backbone identity differs across tasks for {label}')
        results[label] = dict(metrics=scalar_metrics(metrics),
                              artifacts={name: value['artifact'] for name, value in metrics.items()})
    return results


def report(root, results, runs, plan):
    metrics = {key: value['metrics'] for key,value in results.items()}
    full = set(ARMS) <= metrics.keys()
    effects = {}
    summaries = {}
    dest = root / 'report'
    dest.mkdir(exist_ok=True)
    temporary = dest / 'comparison.partial.csv'
    with temporary.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['arm','metric','value','delta_vs_original_pp','delta_vs_aug_a_pp'])
        for label, scores in metrics.items():
            for metric, value in sorted(scores.items()):
                writer.writerow([label, metric, value, 100*(value-metrics['Original'][metric]),
                                 100*(value-metrics['Aug-A'][metric])])
    temporary.replace(dest / 'comparison.csv')
    for label, run in runs.items():
        history = [json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines() if line.strip()]
        n = sum(len(row['draw_ids']) for row in history)
        loss_keys = ('total_loss', 'dino_global_crops_loss', 'dino_local_crops_loss', 'ibot_loss', 'koleo_loss')
        summaries[label] = dict(
            run_directory=str(run),
            updates=len(history), seconds=sum(row['wall_seconds'] for row in history),
            originals_per_second=n/sum(row['wall_seconds'] for row in history),
            median_update_seconds=statistics.median(row['wall_seconds'] for row in history),
            peak_gpu_allocated_bytes=max(row['peak_vram_bytes'] for row in history),
            final_losses={k: history[-1][k] for k in loss_keys if k in history[-1]},
            loss_log=str(run/'metrics.jsonl'))
    if full:
        for metric in metrics['Original']:
            a,b,c,d = (metrics[arm][metric] for arm in ARMS)
            effects[metric] = dict(noise_pp=50*(b-a+d-c), geometry_pp=50*(c-a+d-b),
                                   interaction_pp=100*(d-c-b+a))
    atomic_json(dest/'comparison.json', dict(plan=plan, results=results, training=summaries,
                                            factorial_effects=effects, val_only=True, seed=42))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    panels = (('nffa_A_logreg.macro_f1','NFFA LogReg Macro-F1'),
              ('nffa_A_mlp.macro_f1','NFFA residual MLP Macro-F1')) + tuple(
        (f'aachen_{head}.{point}.ma_iou',f'Aachen {head} MA-IoU · {point}')
        for head in ('linear','cnn8') for point in ('fixed','selected'))
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout='constrained')
    for axis, (metric, title) in zip(axes.flat, panels):
        deltas = [100*(scores[metric]-metrics['Original'][metric]) for scores in metrics.values()]
        axis.bar(list(metrics), deltas, color=['#446681' if key not in ('Original','Aug-A') else '#87929c' for key in metrics])
        axis.axhline(0, color='black', linewidth=.8)
        axis.set(title=title, ylabel='Δ vs Original (pp)')
        axis.tick_params(axis='x', labelrotation=25)
    fig.savefig(dest/'validation_deltas.png', dpi=150)
    plt.close(fig)
    print(f'Comparison saved under {dest}', flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('baseline','prepare','smoke','noise-only','run'),
                        default='baseline', help='baseline uses only Aug-A (default); run executes the full 2×2 study')
    parser.add_argument('--geometry-policy', type=Path, default=POLICY)
    parser.add_argument('--output', type=Path)
    return parser.parse_args(argv)


def frozen_plan(previous, plan):
    compatible_prior = (previous['mode'] in ('run', 'noise-only')
                        and previous['runner_sha256'] == PREVIOUS_RUNNER_SHA256
                        and (previous | {'runner_sha256': plan['runner_sha256']}) == plan)
    if previous != plan and not compatible_prior:
        raise ValueError('This run has a different frozen study plan. Use a separate output.')
    return previous


def main():
    args = parse_args()
    default = OUTPUT if args.stage in ('prepare','run') else OUTPUT.with_name(OUTPUT.name+'_'+args.stage.replace('-', '_'))
    if args.stage == 'baseline':
        default = BASELINE_OUTPUT
    root = (args.output or default).resolve()
    torch.set_num_threads(1)
    base, dataset, collate, identity, init = pilot.configuration(None)
    ids, leakage = audit_training(base, identity, dataset.records)
    if args.stage == 'baseline':
        labels = ['Aug-A']
        policy, allowed = None, frozenset()
    else:
        labels = list(ARMS) if args.stage in ('prepare','run') else ['Aug-A','Aug-B']
        policy, allowed = geometry_policy(args.geometry_policy, ids)
    if args.stage == 'run' and not allowed:
        raise SystemExit('The selected policy gives C/D zero eligible images. '
                         'Use the rotation trial policy or run --stage noise-only.')
    if args.stage not in ('baseline', 'prepare'):
        require_a100()
    if args.stage == 'smoke':
        cfg = arm_config(base, root, True, bool(allowed), allowed)
        cfg.experiment.run_until_step = 2
        cfg.experiment.milestones = [0,2]
        data = training_dataset(cfg, dataset, add_noise=True, geometry_ids=allowed)
        execute(cfg, data, collate, identity, init, root, resume=(root/'run.json').exists())
        pilot.verify_arm(root, contract(cfg, identity, init, 'cuda'), base.initialization.sha256, [0,2])
        print('Two finite SSL updates and teacher exports verified', flush=True)
        return
    with run_lock(root):
        reused, reason = historical_control(base, identity, init)
        if args.stage == 'baseline' and reused is None:
            require_a100()
        plan = dict(mode=args.stage, arms=labels, leakage=leakage, geometry_policy=policy,
                    geometry_transform=GEOMETRY if args.stage != 'baseline' else None,
                    geometry_arms_ready=bool(allowed),
                    geometry_eligible_training_images=len(allowed),
                    geometry_eligible_fraction=len(allowed)/len(dataset),
                    historical_aug_a=str(reused) if reused else None, historical_aug_a_decision=reason,
                    scientific_identity=digest(scientific_identity(contract(base, identity, init, 'cuda'))),
                    runner_sha256=pilot.sha256(Path(__file__)),
                    augmentation_sha256=pilot.sha256(Path(sem_augmentation.__file__)))
        if args.stage == 'prepare':
            visualize(base, dataset, allowed, root/'previews')
            atomic_json(root/'preparation.json', plan)
            print(json.dumps(plan, indent=2), flush=True)
            return
        study_path = root/'study.json'
        if study_path.exists():
            plan = frozen_plan(pilot.read(study_path), plan)
        else:
            atomic_json(study_path, plan)
        runs, checkpoints = fit_arms(root, base, dataset, collate, identity, init, allowed, labels, reused)
        metrics = evaluate(root, base, checkpoints, init, reused)
        report(root, metrics, runs, plan)


if __name__ == '__main__':
    main()
