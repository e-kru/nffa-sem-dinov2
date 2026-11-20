"""Preflight, short A100 measurement and sequential 300-epoch Aug-A training."""

import argparse
from pathlib import Path
import shutil
import signal

from omegaconf import OmegaConf
import torch

from scripts import run_sem_regularization_vits14 as pilot
from src.sem_cellstyle.recipe import contract
from src.sem_cellstyle.runner import require_a100
from src.sem_corpus_v0.benchmark import benchmark
from src.sem_corpus_v0.epochs import MILESTONE_EPOCHS, full_schedule, train_to, verified_milestone
from src.sem_corpus_v0.evaluation import SMALL_ROOT, evaluate, large_reference, publish, training_summary
from src.sem_corpus_v0.preflight import MANIFEST_DIR, aug_a_dataset, preflight, read
from src.ssl.adapter import load_config
from src.ssl.checkpoints import atomic_json, run_lock
from src.ssl.data import sha256
from src.ssl.experiment import digest
from src.ssl.upstream import PIN, ROOT


OUTPUT = ROOT/'outputs/sem_ssl/sem_v0_vits14_aug_a_300epochs_decode_verified_seed42'


def configuration(root, manifest, report):
    cfg = load_config(pilot.CONFIG)
    cfg.recipe = 'sem_v0_vits14_aug_a_300epochs'
    cfg.optim.lr = cfg.optim.backbone_lr
    cfg.runtime.output = str(root)
    cfg.dataset = dict(provider='frozen_sem_posttraining_v0', manifest=str(manifest))
    cfg.checkpointing.recovery_every_optimizer_updates = 1000
    cfg.checkpointing.keep_recent_recovery_snapshots = 2
    return cfg, full_schedule(cfg, report['identity']['count'])


def disk_plan(cfg, root):
    # Payload sizes measured on the previous identical unregularized ViT-S.
    checkpoint = ROOT/'outputs/sem_ssl_regularization/nffa_split_a_vits14_500step_v1_seed42/runs/A/checkpoints/step_000500'
    files = [checkpoint/name for name in ('training.pth','teacher_backbone.pth')]
    size = sum(path.stat().st_size for path in files) if all(path.is_file() for path in files) else 800*1024**2
    size += 5*cfg.experiment.schedule_total_steps*9  # extra serialized FP64 schedule values
    snapshot_bytes = size*(len(MILESTONE_EPOCHS)+cfg.checkpointing.keep_recent_recovery_snapshots+1)
    # One optional original fallback plus eight independent milestone encoders.
    evaluation_bytes = len(MILESTONE_EPOCHS)*(1457*384*48*64*4 + 15543*384*4 + 128*1024**2)
    metrics_bytes = cfg.experiment.schedule_total_steps*1500
    reserve = 5*1024**3
    return dict(checkpoint_bytes=snapshot_bytes, feature_and_eval_bytes=evaluation_bytes,
                metrics_bytes=metrics_bytes, transient_reserve_bytes=reserve,
                required_free_bytes=snapshot_bytes+evaluation_bytes+metrics_bytes+reserve,
                observed_free_bytes=shutil.disk_usage(root).free)


def freeze(root, cfg, manifest, report, initialization):
    frozen = root/'frozen'
    frozen.mkdir(exist_ok=True)
    copy = frozen/'training_manifest.csv'
    expected_sha = report['identity']['manifest_sha256']
    if not copy.exists():
        temp = frozen/'training_manifest.partial.csv'
        shutil.copyfile(manifest,temp)
        if sha256(temp) != expected_sha:
            raise ValueError('Frozen copy differs from the verified input export')
        temp.replace(copy)
    elif sha256(copy) != expected_sha:
        raise ValueError('Frozen manifest differs from the current approved export')
    cfg.dataset.manifest = str(copy)
    expected = contract(cfg, report['identity'], initialization, 'cuda')
    for path in [Path(__file__).resolve(), *sorted((ROOT/'src/sem_corpus_v0').glob('*.py'))]:
        expected['sources'][str(path.relative_to(ROOT))] = sha256(path)
    receipt = frozen/'contract.json'
    if receipt.exists() and read(receipt) != expected:
        raise ValueError('Existing training contract differs. Use its original input or a new output directory.')
    if not receipt.exists():
        atomic_json(receipt,expected)
    pilot.write_config(frozen/'config.yaml',cfg)
    return expected


def run_milestones(root, cfg, data, collate, layout, initialization, expected):
    training = root/'training'
    markers = root/'milestones'
    completed = []
    large = None
    for epoch in MILESTONE_EPOCHS:
        step = cfg.experiment.epoch_to_step[str(epoch)]
        folder = training/'checkpoints'/f'step_{step:06d}'
        if not (folder/'complete.json').is_file():
            print(f'Training to epoch {epoch}, update {step}',flush=True)
            train_to(cfg,data,collate,layout,initialization,expected,training,step)
            torch.cuda.empty_cache()
        metadata = verified_milestone(folder,expected,layout,PIN['backbone_sha256'])
        marker = markers/f'epoch_{epoch:03d}.json'
        if marker.exists():
            result = read(marker)
            if result['snapshot'] != metadata:
                raise ValueError(f'Previously evaluated checkpoint changed: epoch {epoch}')
            # Verify the checkpoint and trust the atomic validation receipt.
            if result['epoch'] != epoch or result['step'] != step:
                raise ValueError(f'Invalid epoch completion receipt: {marker}')
        else:
            print(f'Frozen downstream validation for epoch {epoch}',flush=True)
            current = evaluate(cfg,folder/'teacher_backbone.pth',root/'evaluation'/f'epoch_{epoch:03d}',
                               reference=SMALL_ROOT if epoch == 0 else None)
            if completed and current['protocols'] != completed[0]['protocols']:
                raise ValueError(f'Validation protocol changed at epoch {epoch}')
            result = dict(epoch=epoch, step=step, image_presentations=epoch*layout.originals,
                          checkpoint=str(folder), snapshot=metadata, metrics=current['metrics'],
                          protocols=current['protocols'],
                          evaluation_artifacts={k:v['artifact'] for k,v in current['bundle'].items()},
                          training=training_summary(training/'metrics.jsonl',step,epoch*layout.originals))
            if epoch == 0:
                result['original_vit_l'] = large_reference(cfg,current)
            atomic_json(marker,result)
        if epoch == 0:
            large = result['original_vit_l']
        completed.append(result)
        publish(root,large)
        atomic_json(root/'status.json',dict(status='complete' if epoch == 300 else 'milestone_complete',
                                           epoch=epoch, step=step))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('preflight','benchmark','run'), default='run')
    parser.add_argument('--manifest-dir', type=Path, default=MANIFEST_DIR)
    parser.add_argument('--leakage-review', type=Path)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    root = args.output.resolve()
    with run_lock(root):
        manifest, records, report = preflight(args.manifest_dir,args.leakage_review)
        cfg,layout = configuration(root,manifest,report)
        initialization = Path(cfg.initialization.checkpoint).resolve()
        if sha256(initialization) != PIN['backbone_sha256'] or cfg.initialization.sha256 != PIN['backbone_sha256']:
            raise ValueError('Training requires the pinned original ViT-S/14 checkpoint')
        disk = disk_plan(cfg,root)
        if disk['required_free_bytes'] > disk['observed_free_bytes']:
            report['blockers'].append('Insufficient filesystem space for milestones, recovery and validation caches')
        report.update(ready=not report['blockers'], schedule=OmegaConf.to_container(cfg.experiment,resolve=True),
                      physical_batch=layout.physical_batch, effective_batch=layout.effective_batch,
                      final_microbatches_per_epoch=[
                          (layout.originals-layout.offset(layout.updates-1)+1)//2,
                          (layout.originals-layout.offset(layout.updates-1))//2],
                      total_presentations=300*layout.originals, disk=disk,
                      initialization_sha256=PIN['backbone_sha256'],
                      recipe={key:OmegaConf.to_container(cfg[key],resolve=True) for key in
                              ('student','runtime','train','crops','optim','teacher','dino','ibot','evaluation')})
        atomic_json(root/'preflight.json',report)
        print(f'Preflight: {layout.originals:,} images, {cfg.experiment.schedule_total_steps:,} updates, '
              f'{disk["required_free_bytes"]/2**30:.1f} GiB planned local storage',flush=True)
        if not report['ready']:
            atomic_json(root/'status.json',dict(status='preflight_blocked', blockers=report['blockers']))
            raise SystemExit('Preflight blocked:\n- '+'\n- '.join(report['blockers'])+
                             f'\nDetails: {root/"preflight.json"}')
        if args.stage == 'preflight':
            if not (root/'frozen/contract.json').exists():
                atomic_json(root/'status.json',dict(status='preflight_ready',
                            manifest_sha256=report['identity']['manifest_sha256']))
            return
        gpu = require_a100()
        expected = freeze(root,cfg,manifest,report,initialization)
        data,collate = aug_a_dataset(cfg,records)
        benchmark_path = root/'benchmark/benchmark.json'
        if benchmark_path.exists():
            measurement = read(benchmark_path)
            if measurement['contract_sha256'] != digest(expected) or measurement['gpu'] != gpu.name:
                raise ValueError('A100 benchmark differs from the frozen training contract or device')
        else:
            measurement = benchmark(root/'benchmark',cfg,data,collate,layout,initialization,expected,gpu)
            torch.cuda.empty_cache()
        if not (measurement['resume_verified'] and measurement['export_verified'] and measurement['finite_losses_and_gradients']):
            raise ValueError('Real-corpus A100 probe did not pass')
        print(f'A100: {measurement["originals_per_second"]:.1f} originals/s, '
              f'{measurement["estimated_training_only_seconds"]/3600:.1f} h training, '
              f'{measurement["estimated_total_wall_seconds"]/3600:.1f} h total estimate',flush=True)
        if args.stage == 'run':
            free, _ = torch.cuda.mem_get_info()
            if free < measurement['peak_reserved_bytes']:
                raise ValueError('Available A100 VRAM is below the measured SSL requirement')
            run_milestones(root,cfg,data,collate,layout,initialization,expected)


if __name__ == '__main__':
    def termination(_signal, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM,termination)
    try:
        main()
    except (KeyboardInterrupt, InterruptedError):
        raise SystemExit('Interrupted. Repeat the same command to resume from the last verified checkpoint.')
