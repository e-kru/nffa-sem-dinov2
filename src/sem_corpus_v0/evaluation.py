"""Reuse frozen probes and publish a cumulative validation report."""

import csv
import json
from pathlib import Path

from scripts import run_sem_regularization_vits14 as pilot
from src.sem_cellstyle.regularization_report import scalar_metrics
from src.ssl.checkpoints import atomic_json
from src.ssl.data import sha256
from src.ssl.reporting import result_bundle
from src.ssl.upstream import ROOT


SMALL_ROOT = ROOT/'outputs/sem_ssl_regularization/nffa_split_a_vits14_500step_v1_seed42/evaluation/original'
LARGE_ROOT = ROOT/'outputs/encoder_evaluation/original_vitl14_vs_vits14_seed42/dinov2_vitl14_d5383ea8f4877b24'


def bundle(cfg, root, checksum, task):
    return result_bundle(root, checksum, task, classification_splits=('A',),
                         head_profile=cfg.evaluation.head_profile)


def evaluate(cfg, checkpoint, root, *, reference=None):
    checksum = sha256(checkpoint)
    suffix = f'dinov2_vits14_{checksum[:16]}'
    result, protocols, weights = {}, {}, set()
    for task in ('classification', 'segmentation'):
        prior = reference / suffix if reference else None
        try:
            metrics, contracts, missing, fingerprints = bundle(cfg, prior or root/suffix, checksum, task)
        except (OSError, ValueError, KeyError):
            if not prior:
                raise
            missing = ['historical artifacts are incompatible']
        if missing:
            # Prefer already completed artifacts in the new output on restart.
            metrics, contracts, missing, fingerprints = bundle(cfg, root/suffix, checksum, task)
        if missing:
            pilot.run_command(pilot.evaluation_command(cfg, checkpoint, root, task),
                              root.parent.parent/'logs'/f'{root.name}_{task}.log')
            metrics, contracts, missing, fingerprints = bundle(cfg, root/suffix, checksum, task)
        if missing:
            raise ValueError(f'Incomplete validation: {missing}')
        result.update(metrics)
        protocols.update(contracts)
        weights.update(fingerprints)
    if len(weights) != 1:
        raise ValueError('Classification and segmentation encoder fingerprints differ')
    return dict(metrics=scalar_metrics(result), protocols=protocols, bundle=result)


def comparable_profile(result):
    """Compare the validated profiles after removing encoder identity and feature width."""
    profile = {}
    for key,value in result.items():
        folder = Path(value['artifact']).parent
        cfg = json.loads((folder/'config.json').read_text())
        if key.startswith('nffa_'):
            # Artifact path is split_A/head/seed42/validation_metrics.json.
            data = json.loads((folder.parents[1]/'preflight.json').read_text())['contract']
            cfg.pop('cache_contract_sha256')
            data.pop('encoder')
        else:
            data = cfg.pop('cache_metadata')['contract']
            cfg.pop('feature_cache')
            for field in ('encoder','checkpoint','checkpoint_sha256'):
                data.pop(field)
        data.pop('feature_shape')
        cfg.pop('device', None)
        profile[key] = [cfg,data]
    return profile


def large_reference(cfg, small):
    expected = 'd5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428'
    metrics, weights = {}, set()
    try:
        for task in ('classification','segmentation'):
            result, _, missing, fingerprints = bundle(cfg, LARGE_ROOT, expected, task)
            if missing:
                raise ValueError(f'Incomplete ViT-L baseline: {missing}')
            metrics.update(result)
            weights.update(fingerprints)
        if len(weights) != 1 or comparable_profile(metrics) != comparable_profile(small['bundle']):
            raise ValueError('ViT-L baseline protocols differ from the current ViT-S reference')
    except (OSError, ValueError, KeyError) as error:
        return {'available':False,'reason':str(error)}
    return dict(available=True, metrics=scalar_metrics(metrics), artifacts={
        key:value['artifact'] for key,value in metrics.items()})


def training_summary(path, step, presentations):
    if step == 0:
        return dict(seconds=0., originals_per_second=None, peak_vram_bytes=0, last_update=None)
    elapsed = 0.
    peak = 0
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            elapsed += row['wall_seconds']
            peak = max(peak, row['peak_vram_bytes'])
            if row['step'] == step:
                break
        else:
            raise ValueError(f'Missing training metrics for step {step}')
    return dict(seconds=elapsed, originals_per_second=presentations/elapsed,
                peak_vram_bytes=peak, last_update=row)


def publish(root, large):
    destination = root/'report'
    destination.mkdir(exist_ok=True)
    entries = [json.loads(path.read_text()) for path in sorted((root/'milestones').glob('epoch_*.json'))]
    original = entries[0]['metrics']
    reference_l = large.get('metrics', {})
    temporary = destination/'comparison.partial.csv'
    with temporary.open('w', newline='') as stream:
        export = csv.writer(stream)
        export.writerow(('epoch','step','presentations','metric','value','delta_vs_original_s_pp','delta_vs_original_l_pp'))
        for entry in entries:
            for metric,value in sorted(entry['metrics'].items()):
                export.writerow((entry['epoch'], entry['step'], entry['image_presentations'], metric, value,
                                 (value-original[metric])*100,
                                 (value-reference_l[metric])*100 if metric in reference_l else None))
    temporary.replace(destination/'comparison.csv')
    contract_file = root/'frozen/contract.json'
    scope = json.loads(contract_file.read_text())['data'].get('validation_scope') if contract_file.exists() else None
    atomic_json(destination/'comparison.json', dict(partition='val', seed=42, original_vit_l=large,
                                                     validation_scope=scope, milestones=entries,
                                                     complete=entries[-1]['epoch']==300))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    panels = [
        ('nffa_A_logreg.macro_f1', 'NFFA LogReg Macro-F1'), ('nffa_A_mlp.macro_f1', 'NFFA MLP Macro-F1'),
        *((f'aachen_{head}.{threshold}.ma_iou',f'Aachen {head} MA-IoU · {threshold}')
          for head in ('linear','cnn8') for threshold in ('fixed','selected')),
    ]
    fig,grid = plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for ax,(metric,label) in zip(grid.flat,panels):
        ax.plot([e['epoch'] for e in entries],[100*e['metrics'][metric] for e in entries],
                'o-',label='SEM ViT-S')
        ax.axhline(100*original[metric], ls='--', color='grey', label='Original ViT-S')
        if metric in reference_l:
            ax.axhline(100*reference_l[metric], ls=':', color='black', label='Original ViT-L')
        ax.set(title=label, xlabel='Epoch', ylabel='%')
    grid.flat[0].legend(fontsize=8)
    image = destination/'validation_by_epoch.partial.png'
    fig.savefig(image,dpi=150)
    plt.close(fig)
    image.replace(destination/'validation_by_epoch.png')
