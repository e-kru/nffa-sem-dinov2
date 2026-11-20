"""Explicit sequential checkpoint evaluation through the unchanged shared runner."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ['XFORMERS_DISABLED'] = '1'
from src.ssl.checkpoints import validate_snapshot
from src.ssl.data import sha256
from src.ssl.experiment import REFERENCE
from src.ssl.reporting import result_bundle
from src.ssl.upstream import PIN, ROOT


def commands(run, steps, task, stage, evaluation_root, workers=8):
    for step in steps:
        snapshot = run / 'checkpoints' / f'step_{step:06d}'
        meta = validate_snapshot(snapshot)
        sha = meta['files']['teacher_backbone.pth']
        if step == 0 and sha != PIN['backbone_sha256']:
            raise ValueError('Step 0 differs from the official baseline')
        if step == 0 and sha256(REFERENCE) != sha:
            raise ValueError('Cached original release differs from the step-0 export')
        checkpoint = REFERENCE if step == 0 else snapshot / 'teacher_backbone.pth'
        common = [sys.executable, str(ROOT / 'scripts/evaluate_sem_encoder.py'), '--task', task,
                  '--heads', *(['logreg', 'mlp'] if task == 'classification' else ['linear', 'cnn8']),
                  '--encoder-checkpoint', str(checkpoint.resolve()), '--output-root', str(evaluation_root.resolve()),
                  '--workers', str(workers), '--seed', '42', '--device', 'cuda:0', '--resume']
        yield step, sha, common, stage


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--steps', type=int, nargs='+', default=[0, 50, 100, 200])
    p.add_argument('--task', choices=('classification', 'segmentation'), required=True)
    p.add_argument('--stage', choices=('preflight', 'extract', 'train', 'evaluate', 'report', 'all'), default='preflight')
    p.add_argument('--evaluation-root', type=Path, default=Path('outputs/encoder_evaluation/v1'))
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    for step, sha, common, stage in commands(args.run, args.steps, args.task, args.stage, args.evaluation_root, args.workers):
        stages = ['preflight', 'all'] if stage == 'all' else [stage]
        for current in stages:
            # Preflight recomputes data/source/head contracts and checks loading.
            # Once it passes, existing step-0 metrics may replace an expensive rerun.
            if step == 0 and current == 'all' and not args.dry_run:
                base = args.evaluation_root / f'dinov2_vits14_{sha[:16]}'
                _, _, missing, _ = result_bundle(base, sha, args.task)
                if not missing:
                    from scripts.evaluate_sem_encoder import software
                    provenance = json.loads((base / 'provenance.json').read_text())
                    if provenance['software'] != software():
                        raise ValueError('Step-0 software differs from reproduced baseline')
                    print(f'step=0 {args.task}: verified existing baseline reused', flush=True)
                    continue
            command = [*common, '--stage', current]
            print(f'step={step}: {shlex.join(command)}', flush=True)
            if not args.dry_run:
                env = dict(os.environ)
                env['PYTHONPATH'] = str(ROOT) + os.pathsep + env.get('PYTHONPATH', '')
                subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == '__main__':
    main()
