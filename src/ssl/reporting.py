"""Read validation artifacts, enforce common protocols and represent gaps as null."""
import copy
import json
from pathlib import Path

from src.ssl.checkpoints import atomic_json, validate_snapshot
from src.ssl.data import sha256
from src.ssl.experiment import digest
from src.ssl.upstream import PIN, ROOT


def read(path):
    return json.loads(path.read_text())


def verify_sources(mapping):
    for name, expected in mapping.items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f'Evaluation source changed: {name}')


def result_bundle(base, checkpoint_sha, task, *, check_sources=True, protocol_source_exclusions=(),
                  classification_splits=('A', 'B'), head_profile='legacy'):
    """Metrics and comparable full protocol fingerprints, without encoder fields."""
    results, protocols, missing = {}, {}, []
    fingerprints = set()
    if task == 'classification':
        from src.aachen_ablation.data import digest as feature_digest
        from src.downstream.probes import head_configurations
        head_configs = head_configurations(head_profile)
        for split in classification_splits:
            for head in ('logreg', 'mlp'):
                key = f'nffa_{split}_{head}'
                folder = base / 'classification' / f'split_{split}' / head / 'seed42'
                metric_path = folder / 'validation_metrics.json'
                results[key] = None
                if not metric_path.exists():
                    missing.append(str(metric_path))
                    continue
                metric, config = read(metric_path), read(folder / 'config.json')
                preflight = read(folder.parents[1] / 'preflight.json')
                contract = copy.deepcopy(preflight['contract'])
                encoder = contract.pop('encoder')
                if (encoder['checkpoint_sha256'] != checkpoint_sha or encoder['upstream_commit'] != PIN['commit']
                        or metric['encoder'] != encoder
                        or metric['partition'] != 'val' or metric['test_evaluated'] is not False
                        or metric['seed'] != 42 or metric['head'] != head or metric['split'] != split
                        or config['seed'] != 42 or config['head_config'] != head_configs[head]
                        or metric['head_config'] != head_configs[head]
                        or config['cache_contract_sha256'] != feature_digest(preflight['contract'])
                        or metric.get('nonscientific_smoke_test') or config.get('nonscientific_smoke_test')):
                    raise ValueError(f'Invalid classification artifact: {metric_path}')
                if check_sources:
                    verify_sources({k: v for k, v in contract['sources'].items() if k not in protocol_source_exclusions})
                comparison = {k: v for k, v in config.items() if k not in ('cache_contract_sha256', 'device')}
                comparable = copy.deepcopy(contract)
                for name in protocol_source_exclusions:
                    comparable['sources'].pop(name, None)
                protocols[key] = digest(dict(config=comparison, data=comparable))
                fingerprints.add(metric['encoder_state_sha256'])
                results[key] = dict(macro_f1=metric['macro_f1'], accuracy=metric['accuracy'], artifact=str(metric_path))
    elif task == 'segmentation':
        for head in ('linear', 'cnn8'):
            key = f'aachen_{head}'
            paths = sorted(base.glob(f'segmentation/runs/A3/{head}/seed42_*/validation_metrics.json'))
            results[key] = None
            if not paths:
                missing.append(f'{base}/segmentation/A3/{head}/seed42')
                continue
            if len(paths) != 1:
                raise ValueError(f'Ambiguous evaluation results: {paths}')
            metric_path = paths[0]
            metric, config = read(metric_path), read(metric_path.parent / 'config.json')
            cache = config['cache_metadata']
            contract = copy.deepcopy(cache['contract'])
            if (contract.pop('checkpoint_sha256') != checkpoint_sha or metric['split'] != 'val'
                    or metric['test_evaluated'] is not False or metric['seed'] != 42
                    or config['seed'] != 42 or config['head'] != head
                    or config['variant'] != dict(name='A3', width=896, height=672, quadrants=False)
                    or metric.get('nonscientific_smoke_test')):
                raise ValueError(f'Invalid segmentation artifact: {metric_path}')
            expected = dict(epochs=30, patience=8, batch_size=8, micro_batch_size=8,
                            validation_batch_size=8, lr=.001 if head == 'linear' else .0003,
                            weight_decay=.0001, optimizer='AdamW', scheduler=None)
            if any(config.get(key) != value for key, value in expected.items()):
                raise ValueError(f'Segmentation head differs from the fixed baseline: {metric_path}')
            contract.pop('checkpoint')
            if check_sources:
                verify_sources(config['training_sources'])
                verify_sources(contract['sources'])
                verify_sources(contract['pinned_encoder_sources'])
            comparison = {k: v for k, v in config.items() if k not in ('cache_metadata', 'feature_cache', 'device')}
            protocols[key] = digest(dict(config=comparison, data=contract))
            fingerprints.add(cache['encoder_state_sha256'])
            results[key] = dict(fixed=metric['validation_at_0_50'], selected=metric['validation_at_best_threshold'],
                                selection=metric['threshold_selection'], artifact=str(metric_path))
    else:
        raise ValueError(task)
    if len(fingerprints) > 1:
        raise ValueError('Evaluation heads use different backbone tensors')
    return results, protocols, missing, fingerprints


def collect(run, evaluation_root, steps=(0, 50, 100, 200), *, check_sources=True):
    rows, reference_protocols, missing, ssl_contracts = [], {}, [], set()
    for step in steps:
        path = run / 'checkpoints' / f'step_{step:06d}'
        if path.exists():
            metadata = validate_snapshot(path)
            sha = metadata['files']['teacher_backbone.pth']
            ssl_contracts.add(metadata['contract_sha256'])
            if len(ssl_contracts) > 1:
                raise ValueError('SSL milestones use different experiment contracts')
        elif step == 0:
            sha = PIN['backbone_sha256']
        else:
            sha = None
        row = dict(step=step, teacher_sha256=sha, milestone_available=path.exists(), metrics={})
        fingerprints = set()
        for task in ('classification', 'segmentation'):
            if sha:
                results, protocols, gaps, weights = result_bundle(
                    evaluation_root / f'dinov2_vits14_{sha[:16]}', sha, task, check_sources=check_sources)
                fingerprints.update(weights)
                for key, value in protocols.items():
                    previous = reference_protocols.setdefault(key, value)
                    if previous != value:
                        raise ValueError(f'Downstream protocol changed across SSL checkpoints: {key}')
                missing.extend(gaps)
            else:
                keys = ([f'nffa_{s}_{h}' for s in ('A', 'B') for h in ('logreg', 'mlp')]
                        if task == 'classification' else ['aachen_linear', 'aachen_cnn8'])
                results = dict.fromkeys(keys)
                missing.append(f'SSL milestone {step} not available: {task}')
            row['metrics'].update(results)
        if len(fingerprints) > 1:
            raise ValueError('Classification and segmentation backbones differ')
        rows.append(row)
    return dict(partition='val', test_evaluated=False, complete=not missing,
                rows=rows, missing=missing, protocol_sha256=reference_protocols,
                note='Step 0 may refer to the verified original frozen reference before SSL starts; missing results are null.')


def render_report(result, run, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    destination.mkdir(parents=True, exist_ok=False)
    atomic_json(destination / 'comparison.json', result)
    keys = ['nffa_A_logreg', 'nffa_A_mlp', 'nffa_B_logreg', 'nffa_B_mlp']
    columns = ['Step', 'A LogReg F1', 'A MLP F1', 'B LogReg F1', 'B MLP F1']
    for head in ('linear', 'cnn8'):
        for operating in ('fixed', 'selected'):
            columns.extend(f'{head} {operating} {name}' for name in ('threshold', 'MA-IoU', 'mIoU', 'precision', 'recall'))
    lines = ['# SEM SSL validation comparison', '', 'Scores are fractions; fixed threshold 0.50 and validation-selected thresholds are separate.', '',
             '| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    for row in result['rows']:
        m = row['metrics']
        values = [row['step'], *[(m[k] or {}).get('macro_f1') for k in keys]]
        for head in ('linear', 'cnn8'):
            for operating in ('fixed', 'selected'):
                point = (m[f'aachen_{head}'] or {}).get(operating, {})
                values.extend(point.get(k) for k in ('threshold', 'ma_iou', 'miou', 'ma_precision', 'ma_recall'))
        lines.append('| ' + ' | '.join('—' if v is None else str(v) if i == 0 else f'{v:.6f}' for i, v in enumerate(values)) + ' |')
    lines += ['', 'Missing artifacts:', '', *[f'- {item}' for item in result['missing']]]
    (destination / 'comparison.md').write_text('\n'.join(lines) + '\n')
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    steps = [row['step'] for row in result['rows']]
    for key in keys:
        axes[0].plot(steps, [(row['metrics'][key] or {}).get('macro_f1', float('nan')) for row in result['rows']], 'o-', label=key)
    axes[0].set_ylabel('Validation macro-F1')
    for ax, metric in zip(axes[1:], ('ma_iou', 'miou')):
        for head in ('linear', 'cnn8'):
            for operating, style in (('fixed', '--'), ('selected', '-')):
                values = [((row['metrics'][f'aachen_{head}'] or {}).get(operating) or {}).get(metric, float('nan')) for row in result['rows']]
                ax.plot(steps, values, 'o' + style, label=f'{head} {operating}')
        ax.set_ylabel(f'Validation {metric}')
    for ax in axes:
        ax.set_xlabel('SSL optimizer steps')
        ax.legend(fontsize=8)
        ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(destination / 'downstream_vs_steps.png', dpi=160)
    plt.close(fig)
    metrics_path = run / 'metrics.jsonl'
    if metrics_path.exists():
        metrics = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
        if metrics:
            fig, ax = plt.subplots(figsize=(9, 5))
            for key in ('total_loss', 'dino_global_crops_loss', 'dino_local_crops_loss', 'ibot_loss', 'koleo_loss'):
                ax.plot([r['step'] for r in metrics], [r[key] for r in metrics], label=key)
            ax.set_xlabel('SSL optimizer steps')
            ax.set_ylabel('Training objective contribution (not downstream quality)')
            ax.legend()
            fig.tight_layout()
            fig.savefig(destination / 'ssl_losses.png', dpi=160)
            plt.close(fig)
