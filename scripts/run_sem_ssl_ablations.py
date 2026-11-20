"""Three fixed SEM SSL ablations, existing validation runners, one sequential GPU."""
import argparse
import copy
import csv
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tempfile

import numpy as np
from omegaconf import OmegaConf
import torch

from src.ssl.checkpoints import atomic_json, run_lock, validate_snapshot
from src.ssl.data import sha256
from src.ssl.experiment import REFERENCE, configuration, contract, digest, step_schedules
from src.ssl.reporting import result_bundle
from src.ssl.upstream import PIN, ROOT
from src.ssl_benchmark import dataset_from_inventory

BASELINE = ROOT/'outputs/sem_ssl/nffa_5000step_seed42'
BASELINE_EVAL = ROOT/'outputs/encoder_evaluation/nffa_5000step_workers_v1'
DEFAULT_OUTPUT = ROOT/'outputs/sem_ssl_ablations/low_lr_freeze8_seed42_v1'
EXPERIMENTS = (('A_low_lr', 'low_lr', .1, 0), ('B_freeze8', 'freeze8', 1., 8),
               ('C_low_lr_freeze8', 'low_lr_freeze8', .1, 8))
STEPS = (0, 500, 1000)
# Only this CLI gained an opt-in inventory reader. Its old/new hashes remain
# recorded. Actual transforms, heads, training, metrics and geometry must match.
SOURCE_EXCLUSIONS = ('scripts/evaluate_sem_encoder.py',)
KEYS = ('nffa_A_logreg', 'nffa_A_mlp', 'nffa_B_logreg', 'nffa_B_mlp', 'aachen_linear', 'aachen_cnn8')


def read(path):
    return json.loads(Path(path).read_text())


def prepare(output):
    """CPU metadata checks only; no image/mask-content hashing or model forward."""
    baseline = read(BASELINE/'run.json')
    reference = configuration(ROOT/'configs/sem_ssl/nffa_5000step.yaml')
    original = OmegaConf.to_container(reference, resolve=True)
    if original != baseline['config'] or baseline['contract']['initialization_sha256'] != PIN['backbone_sha256']:
        raise ValueError('Reference YAML differs from the actual original 5000-step run')
    inventory = baseline['dataset']
    dataset_from_inventory(reference, inventory)  # existing manifest + selection checks
    if inventory['count'] != 8061 or digest(inventory) != baseline['contract']['data_sha256']:
        raise ValueError('Expected the original verified 8061-image inventory')
    configs, contracts = {}, {}
    base_schedules = step_schedules(reference)
    for label, suffix, factor, frozen in EXPERIMENTS:
        cfg = configuration(ROOT/f'configs/sem_ssl/nffa_ablation_{suffix}.yaml')
        expected = copy.deepcopy(original)
        expected['experiment'].update(name='nffa_sem_vits14_ablation_'+suffix, run_until_step=1000,
                                      milestones=list(STEPS))
        expected['student']['freeze_first_blocks'] = frozen
        for key in ('base_lr', 'lr', 'min_lr'):
            expected['optim'][key] *= factor
        actual = OmegaConf.to_container(cfg, resolve=True)
        if actual != expected:
            raise ValueError(f'Uncontrolled configuration difference: {label}')
        for key, schedule in step_schedules(cfg).items():
            expected_values = base_schedules[key].schedule * (factor if key in ('lr', 'last_layer_lr') else 1)
            np.testing.assert_allclose(schedule.schedule, expected_values, rtol=1e-14, atol=1e-20)
        configs[label] = actual
        contracts[label] = {**contract(cfg, inventory, REFERENCE, 'bf16'), 'device_type': 'cuda'}
    if any(c['initialization_sha256'] != PIN['backbone_sha256'] for c in contracts.values()):
        raise ValueError('Original initialization file differs from the pinned release')
    # Existing verified baseline, no retraining and no historical checkpoint rewrite.
    verified = read(ROOT/'outputs/sem_ssl_checks/nffa_5000step_verified.json')
    if not verified['passed']:
        raise ValueError('Baseline checkpoint verification failed')
    for step in STEPS:
        meta = read(BASELINE/f'checkpoints/step_{step:06d}/complete.json')
        item = next(r for r in verified['checkpoints'] if r['step'] == step)
        if (not item['passed'] or item['teacher_sha256'] != meta['files']['teacher_backbone.pth']
                or item['contract_sha256'] != meta['contract_sha256']):
            raise ValueError('Baseline verification receipt differs from milestone metadata')
    files = [Path(__file__), ROOT/'scripts/run_sem_ssl_ablations.sh', ROOT/'scripts/verify_sem_ssl.py',
             ROOT/'scripts/run_aachen_ablation.py', ROOT/'scripts/evaluate_sem_encoder.py']
    plan = dict(schema=1, seed=42, configs=configs, contracts=contracts,
                dataset_records_sha256=inventory['records_sha256'], baseline_run=str(BASELINE),
                baseline_evaluation=str(BASELINE_EVAL), protocol_source_exclusions=list(SOURCE_EXCLUSIONS),
                orchestration_sources={str(p.relative_to(ROOT)): sha256(p) for p in files})
    # Validate all 18 existing baseline task results against the shared protocol.
    collect_rows(output, baseline_only=True)
    from src.ssl import ablation_data
    from src.aachen_ablation.data import manifest_rows
    from src.aachen_ablation.geometry import VARIANTS
    baseline_encoder = BASELINE_EVAL/f'dinov2_vits14_{PIN["backbone_sha256"][:16]}'
    for split in ('A', 'B'):
        path = baseline_encoder/f'classification/split_{split}/preflight.json'
        stored = read(path)['contract']
        split_file = ROOT/f'outputs/dataset_v3/splits/split_{"A_exact" if split == "A" else "B_full"}.csv'
        ablation_data.classification(path, ROOT/'outputs/dataset_v3/manifest.csv', split_file, stored['image_root'])
    manifest = ROOT/'outputs/aachen_heerlen/splits/group_split_seed42.csv'
    ablation_data.segmentation(ablation_data.segmentation_reference(baseline_encoder), manifest,
                              manifest_rows(manifest), [VARIANTS['A3']])
    return plan, inventory


def commands(output, label, suffix, *, resume=False):
    folder = output/label
    train = [sys.executable, '-m', 'scripts.train_sem_ssl', '--stage', 'train',
             '--config', f'configs/sem_ssl/nffa_ablation_{suffix}.yaml', '--output', str(folder/'train'),
             '--inventory', str(output/'dataset_inventory.json'), '--device', 'cuda', '--precision', 'bf16']
    if resume:
        train.append('--resume')
    verify = [sys.executable, '-m', 'scripts.verify_sem_ssl', '--run', str(folder/'train'),
              '--steps', *map(str, STEPS), '--output', str(folder/'checkpoints_verified.json')]
    evaluation = []
    baseline_encoder = BASELINE_EVAL/f'dinov2_vits14_{PIN["backbone_sha256"][:16]}'
    for step in STEPS[1:]:
        for task, heads, workers in [('classification', ['logreg', 'mlp'], 8), ('segmentation', ['linear', 'cnn8'], 16)]:
            evaluation.append([sys.executable, '-m', 'scripts.evaluate_sem_encoder', '--task', task,
                '--heads', *heads, '--stage', 'all', '--workers', str(workers), '--seed', '42', '--device', 'cuda:0',
                '--encoder-checkpoint', str(folder/f'train/checkpoints/step_{step:06d}/teacher_backbone.pth'),
                '--output-root', str(folder/'evaluation'), '--verified-data-root', str(baseline_encoder), '--resume'])
    return train, verify, evaluation


def run_command(command, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f'\n{shlex.join(command)}\nLog: {log}', flush=True)
    with log.open('a', buffering=1) as stream:
        stream.write('\nCOMMAND '+shlex.join(command)+'\n')
        with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1) as child:
            try:
                for line in child.stdout:
                    print(line, end='', flush=True)
                    stream.write(line)
                if child.wait():
                    raise subprocess.CalledProcessError(child.returncode, command)
            except KeyboardInterrupt:
                if child.poll() is None:
                    child.send_signal(signal.SIGINT)
                    child.wait()
                raise


def run_study(output, plan, inventory, runner=run_command):
    with run_lock(output):
        path = output/'study.json'
        if path.exists():
            if read(path) != plan:
                raise ValueError('Study config/data/source/software changed; use a new output directory')
        else:
            if any(p.name != '.writer.lock' for p in output.iterdir()):
                raise FileExistsError('New study output must be empty')
            atomic_json(path, plan)
            atomic_json(output/'dataset_inventory.json', inventory)
        for label, suffix, _, frozen in EXPERIMENTS:
            folder = output/label
            training = folder/'train'
            train, verification, _ = commands(output, label, suffix, resume=(training/'run.json').exists())
            if (training/'run.json').exists() and read(training/'run.json')['contract'] != plan['contracts'][label]:
                raise ValueError(f'Run contract changed: {label}')
            status = read(training/'status.json') if (training/'status.json').exists() else {}
            if status.get('status') != 'completed' or status.get('step') != 1000:
                runner(train, folder/'logs/train.log')
            status = read(training/'status.json')
            if status.get('status') != 'completed' or status.get('step') != 1000:
                raise InterruptedError(f'{label} stopped safely; repeat the same command to resume')
            if read(training/'run.json')['contract'] != plan['contracts'][label]:
                raise ValueError(f'Trained run differs from prepared contract: {label}')
            receipt = folder/'checkpoints_verified.json'
            if not receipt.exists():
                runner(verification, folder/'logs/verify.log')
            else:
                # A receipt alone must not hide corrupted files on a later resume.
                for step in STEPS:
                    validate_snapshot(training/f'checkpoints/step_{step:06d}')
            verified = read(receipt)
            if (not verified['passed'] or [r['step'] for r in verified['checkpoints']] != list(STEPS)
                    or any(r.get('frozen_blocks') != frozen or (frozen and not r.get('frozen_blocks_unchanged'))
                           for r in verified['checkpoints'])):
                raise ValueError(f'Incomplete checkpoint verification: {label}')
        for label, suffix, _, _ in EXPERIMENTS:
            for index, command in enumerate(commands(output, label, suffix)[2]):
                runner(command, output/label/f'logs/evaluation_{index}.log')
        write_report(output, collect_rows(output))


def collect_rows(output, *, baseline_only=False):
    rows, common = [], {}
    experiments = [('reference', BASELINE, BASELINE_EVAL, STEPS)]
    if not baseline_only:
        experiments += [(label, output/label/'train', output/label/'evaluation', STEPS[1:])
                        for label, _, _, _ in EXPERIMENTS]
    for label, run, evaluation, steps in experiments:
        verification = (read(ROOT/'outputs/sem_ssl_checks/nffa_5000step_verified.json') if label == 'reference'
                        else read(output/label/'checkpoints_verified.json'))
        for step in steps:
            meta = read(run/f'checkpoints/step_{step:06d}/complete.json')
            receipt = next(r for r in verification['checkpoints'] if r['step'] == step)
            sha = meta['files']['teacher_backbone.pth']
            if not receipt['passed'] or receipt['teacher_sha256'] != sha or receipt['contract_sha256'] != meta['contract_sha256']:
                raise ValueError('Checkpoint verification/metadata mismatch')
            if label != 'reference' and any(r.get('frozen_blocks') and not r.get('frozen_blocks_unchanged')
                                            for r in verification['checkpoints']):
                raise ValueError('Frozen-block verification is missing')
            metrics, fingerprints = {}, set()
            for task in ('classification', 'segmentation'):
                bundle, protocols, missing, weights = result_bundle(
                    evaluation/f'dinov2_vits14_{sha[:16]}', sha, task,
                    protocol_source_exclusions=SOURCE_EXCLUSIONS)
                if missing:
                    raise ValueError('Missing validation artifacts: '+str(missing))
                for key, value in protocols.items():
                    if common.setdefault(key, value) != value:
                        raise ValueError(f'Downstream protocol changed: {label}/{step}/{key}')
                metrics.update(bundle)
                fingerprints.update(weights)
            if len(fingerprints) != 1:
                raise ValueError('Heads use different backbone tensors')
            rows.append(dict(experiment=label, step=step, teacher_sha256=sha, metrics=metrics))
    return rows


def scores(row):
    return {key: (value['macro_f1'] if key.startswith('nffa_') else value['fixed']['ma_iou'])
            for key, value in row['metrics'].items()}


def recommendation(rows):
    original = scores(next(r for r in rows if r['experiment'] == 'reference' and r['step'] == 0))
    candidates = [r for r in rows if r['experiment'] != 'reference']
    dominant = [r for r in candidates if all(scores(r)[k] >= original[k] for k in KEYS)
                and any(scores(r)[k] > original[k] for k in KEYS)]
    if dominant:
        best = max(dominant, key=lambda r: sum(scores(r)[k]-original[k] for k in KEYS))
        decision = f'{best["experiment"]} bei Schritt {best["step"]} als Kandidaten für mehrere Seeds prüfen; '
        decision += 'alle sechs primären Punktwerte erreichen mindestens die Originalreferenz.'
    else:
        decision = ('Die Originalgewichte als allgemeine Referenz beibehalten: keine Ablation erreicht '
                    'in allen sechs primären Punktwerten mindestens Schritt 0 und verbessert mindestens einen.')
    reference = {r['step']: scores(r) for r in rows if r['experiment'] == 'reference'}
    rescue = [dict(experiment=r['experiment'], step=r['step']) for r in candidates
              if all(scores(r)[k] >= reference[r['step']][k] for k in KEYS)
              and any(scores(r)[k] > reference[r['step']][k] for k in KEYS)]
    if rescue:
        decision += ' Gegenüber der gleich lang trainierten Referenz verbessern oder erhalten '
        decision += ', '.join(f'{r["experiment"]}@{r["step"]}' for r in rescue)+' alle sechs primären Punktwerte.'
    else:
        decision += ' Gegenüber der gleich lang trainierten Referenz gibt es keinen durchgängigen Gewinn über alle sechs Metriken.'
    winners = {key: max(rows, key=lambda r: scores(r)[key]) for key in KEYS}
    return dict(decision=decision, improved_over_matched_reference=rescue, best_by_metric={k: dict(experiment=r['experiment'], step=r['step'],
                score=scores(r)[k]) for k, r in winners.items()},
                limitation='Ein Seed; keine Signifikanzbehauptung. Feste Schwelle primär, Val-Schwellen separat. '
                           'Task-spezifische Gewinner stehen unten; keine automatische längere Trainingsfortsetzung.')


def write_report(output, rows):
    rec = recommendation(rows)
    original = scores(rows[0])
    baseline = {r['step']: scores(r) for r in rows if r['experiment'] == 'reference'}
    result = dict(complete=True, partition='val', test_evaluated=False, seed=42, rows=rows,
                  protocol_source_exclusions=list(SOURCE_EXCLUSIONS), recommendation=rec,
                  provenance='Old/new evaluator source hashes retained in baseline preflights and study.json; '
                             'the opt-in verified-inventory CLI change is excluded from protocol comparison only.')
    destination = output/'report'
    if destination.exists():
        if (read(destination/'comparison.json') != result
                or not all((destination/name).is_file() for name in ('comparison.md', 'comparison.csv'))):
            raise FileExistsError('Existing report differs/incomplete; preserve it and inspect before continuing')
        print(f'Existing complete report reused: {destination}', flush=True)
        return
    report = Path(tempfile.mkdtemp(prefix='.pending_report_', dir=output))
    atomic_json(report/'comparison.json', result)
    lines = ['# SEM SSL: Low LR und Freeze 8', '', 'Alle Werte in Prozent; Validation, Seed 42. '
             'Referenz = existierender 5.000-Schritt-Lauf. Original = Referenz bei Schritt 0.', '',
             '## Empfehlung aus den Messungen', '', rec['decision'], '', rec['limitation'], '',
             '## NFFA Macro-F1', '', '| Experiment | Schritt | A LogReg | A MLP | B LogReg | B MLP |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    flat = []
    for row in rows:
        current = scores(row)
        lines.append('| '+row['experiment']+' | '+str(row['step'])+' | '+
                     ' | '.join(f'{current[k]*100:.2f}' for k in KEYS[:4])+' |')
        entry = dict(experiment=row['experiment'], step=row['step'])
        for key in KEYS:
            entry[key] = current[key]
            entry[key+'_delta_original_pp'] = 100*(current[key]-original[key])
            entry[key+'_delta_matched_reference_pp'] = 100*(current[key]-baseline[row['step']][key])
        flat.append(entry)
    lines += ['', '## Aachen: feste und ausgewählte Schwellen getrennt', '',
              '| Experiment | Schritt | Head | MA-IoU 0,50 | mIoU 0,50 | Val-Schwelle | MA-IoU ausgewählt | mIoU ausgewählt |',
              '| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |']
    for row, entry in zip(rows, flat):
        for head in ('linear', 'cnn8'):
            point = row['metrics']['aachen_'+head]
            f, s = point['fixed'], point['selected']
            lines.append(f'| {row["experiment"]} | {row["step"]} | {head} | {f["ma_iou"]*100:.2f} | '
                         f'{f["miou"]*100:.2f} | {s["threshold"]:.2f} | {s["ma_iou"]*100:.2f} | {s["miou"]*100:.2f} |')
            for mode in ('fixed', 'selected'):
                for key, value in point[mode].items():
                    if isinstance(value, (int, float)):
                        entry[f'aachen_{head}_{mode}_{key}'] = value
    lines += ['', '## Beste Punktwerte je primärer Metrik', '',
              '| Metrik | Experiment | Schritt | Wert, % |', '| --- | --- | ---: | ---: |']
    for key, best in rec['best_by_metric'].items():
        lines.append(f'| {key} | {best["experiment"]} | {best["step"]} | {best["score"]*100:.2f} |')
    lines += ['', '## Änderung gegenüber Original und gleich lang trainierter Referenz', '',
              'Die CSV enthält für jede primäre Metrik beide Differenzen in Prozentpunkten. '
              'Die Auswahl berücksichtigt alle sechs primären Metriken; unterschiedliche Aufgaben werden '
              'nicht als ein gemeinsamer statistischer Endpunkt ausgegeben.', '',
              '| Experiment | Schritt | Δ A LogReg | Δ A MLP | Δ B LogReg | Δ B MLP | Δ Linear MA-IoU | Δ CNN8 MA-IoU |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        if row['experiment'] != 'reference':
            lines.append('| '+row['experiment']+' | '+str(row['step'])+' | '+
                         ' | '.join(f'{100*(scores(row)[k]-baseline[row["step"]][k]):+.2f}' for k in KEYS)+' |')
    lines += ['', 'Δ in Prozentpunkten gegenüber derselben Schrittzahl des Referenzlaufs. '
              'Eine Erholung unter Low LR/Freeze stützt die Hypothese innerhalb dieses Laufs; '
              'weitere Seeds und die gemeinsame LR-Änderung an Backbone und Head begrenzen die kausale Interpretation.', '',
              '## Provenienz und Grenzen', '',
              'Alle Runs: 5.000-Schritt-Schedule, Stopp bei 1.000; gleicher Seed, gleicher Korpus, originale Objectives. '
              'Low LR skaliert den gesamten LR-Schedule einschließlich Minimum und Head-Gruppen um 0,1. '
              'Freeze 8 fixiert Blöcke 0–7 in Student und Teacher; Embeddings, Tokens, Norm und Head bleiben trainierbar.', '',
              result['provenance'], '',
              'Einzelmetriken mit Artefaktpfaden stehen in comparison.json. Vollständige Laufkonfigurationen, '
              'Datenidentität und Quellen: ../study.json sowie ../<Experiment>/train/run.json. '
              'Checkpoint-Verifikation: ../<Experiment>/checkpoints_verified.json. Logs: ../<Experiment>/logs/.', '']
    (report/'comparison.md').write_text('\n'.join(lines))
    with (report/'comparison.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    os.rename(report, destination)
    print(rec['decision'], flush=True)
    print(f'Report: {destination/"comparison.md"}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true', help='Metadata/config/schedule/protocol checks and plan; no GPU or output writes')
    mode.add_argument('--report-only', action='store_true', help='Recollect already completed study results; no training')
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output == ROOT or any((parent/'run.json').exists() for parent in (output, *output.parents)):
        raise ValueError('Study output must be separate from existing training runs')
    os.chdir(ROOT)
    torch.set_num_threads(1)
    plan, inventory = prepare(output)
    if args.dry_run:
        for label, suffix, _, _ in EXPERIMENTS:
            training, verification, evaluations = commands(output, label, suffix)
            for command in [training, verification, *evaluations]:
                print(shlex.join(command))
        print('All metadata/config/schedule/protocol checks passed; GPU study NOT started.')
        return
    if args.report_only:
        if read(output/'study.json') != plan:
            raise ValueError('Study contract differs')
        write_report(output, collect_rows(output))
        return
    from src.ssl_benchmark import validate_gpu
    validate_gpu()
    def request_stop(*_args):
        raise KeyboardInterrupt()
    previous = signal.signal(signal.SIGTERM, request_stop)
    try:
        run_study(output, plan, inventory)
    except (KeyboardInterrupt, InterruptedError):
        print('Study interrupted; repeat the same command to resume compatible stages.', flush=True)
        return 130
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    raise SystemExit(main())
