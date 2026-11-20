"""Prepare/execute predeclared Large validation milestones using existing probes."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def commands(cfg, run, output, passes, verified_data_root=None, full=False):
    from src.ssl.checkpoints import validate_snapshot
    from src.sem_cellstyle.recipe import budgets
    meta = json.loads((run / 'run.json').read_text())
    expected = meta['config']['experiment']
    for key, value in budgets(cfg, expected['verified_train_originals']).items():
        if key not in ('run_until_step', 'initial_original_presentations') and expected[key] != value:
            raise ValueError('Evaluation recipe budget differs from the recorded scientific run')
    for point in passes:
        if point not in cfg.evaluation.quick_validation_passes and point not in cfg.evaluation.full_validation_passes:
            raise ValueError('Only predeclared milestones may be evaluated')
        step = expected['pass_to_step'][str(point)]
        folder = run / 'checkpoints' / f'step_{step:06d}'
        validate_snapshot(folder)
        full_scope = full or point in cfg.evaluation.full_validation_passes
        for task, heads, workers in (
            ('classification', ['logreg','mlp'] if full_scope else ['logreg'], cfg.evaluation.classification_workers),
            ('segmentation', ['linear','cnn8'] if full_scope else ['linear'], cfg.evaluation.segmentation_workers)):
            common = [sys.executable, '-m', 'scripts.evaluate_sem_encoder', '--encoder-name', 'dinov2_vitl14',
                      '--head-profile', cfg.evaluation.get('head_profile','legacy'),
                      '--encoder-checkpoint', str(folder / 'teacher_backbone.pth'), '--output-root', str(output),
                      '--task', task, '--heads', *heads, '--workers', str(workers), '--seed','42', '--resume']
            if task == 'classification':
                common += ['--splits', *cfg.evaluation.get('classification_splits', ['A','B'])]
            if verified_data_root:
                common += ['--verified-data-root', str(verified_data_root)]
            # Existing runner includes original-mask metrics, both threshold
            # protocols, components/islands and validation-only reports.
            yield point, task, common + ['--stage','all']
        yield point, 'report', [sys.executable, '-m', 'scripts.evaluate_sem_encoder', '--encoder-name','dinov2_vitl14',
            '--encoder-checkpoint', str(folder / 'teacher_backbone.pth'), '--output-root', str(output),
            '--stage','report', '--classification-heads', *(['logreg','mlp'] if full_scope else ['logreg']),
            '--segmentation-heads', *(['linear','cnn8'] if full_scope else ['linear']), '--seed','42',
            '--head-profile', cfg.evaluation.get('head_profile','legacy'),
            '--splits', *cfg.evaluation.get('classification_splits', ['A','B'])]


def main():
    from src.sem_cellstyle.recipe import configuration, DEFAULT_CONFIG
    from src.ssl.upstream import ROOT
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    p.add_argument('--run', type=Path)
    p.add_argument('--output', type=Path, default=Path('outputs/encoder_evaluation/sem_cellstyle_vitl14_v1'))
    p.add_argument('--passes', type=int, nargs='+', default=[0,1,5,10])
    p.add_argument('--verified-data-root', type=Path)
    p.add_argument('--full', action='store_true', help='All existing heads even at quick milestones; controlled regularization study')
    p.add_argument('--report-only', action='store_true')
    p.add_argument('--report-output', type=Path, default=Path('outputs/sem_ssl_reports/sem_cellstyle_vitl14_v1'))
    p.add_argument('--execute', action='store_true', help='Default prints commands only')
    args = p.parse_args(); cfg = configuration(args.config)
    run = (args.run or Path(cfg.runtime.output)).resolve()
    verified = args.verified_data_root or cfg.evaluation.verified_data_root
    if args.full:
        cfg.evaluation.full_validation_passes=sorted(set(cfg.evaluation.full_validation_passes)|set(args.passes))
    if args.report_only:
        from src.sem_cellstyle.reporting import write
        print(write(cfg, run, args.output.resolve(), args.report_output.resolve())); return
    for point, task, command in commands(cfg, run, args.output.resolve(), args.passes, verified, full=args.full):
        print(f'pass={point} task={task}: {shlex.join(command)}', flush=True)
        if args.execute:
            subprocess.run(command, cwd=ROOT, check=True)
    if args.execute:
        from src.sem_cellstyle.reporting import write
        print(write(cfg, run, args.output.resolve(), args.report_output.resolve()))


if __name__ == '__main__':
    main()
