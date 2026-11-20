"""CPU-only synthetic checks; reduced Meta fixture is never a Phase-A model."""
import copy
import csv
import json
from pathlib import Path
import random
from unittest import mock

import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader

pytest.importorskip('omegaconf')
from src.ssl.upstream import ROOT, activate, checkout_path
if not checkout_path().is_dir():
    pytest.skip('Pinned Meta checkout unavailable', allow_module_level=True)
activate()
from src.ssl.adapter import SingleGPUAdapter
from src.ssl.checkpoints import (find_latest, reconcile_metrics, restore_rng, rng_state,
                                trim_recovery, validate_snapshot)
from src.ssl.data import SEMTrainDataset, augmentation_and_collate
from src.ssl.experiment import configuration, contract, dataset_plan, digest, step_schedules
from src.ssl.stream import CursorBatches, DrawCollate, DrawDataset, data_rng
from src.ssl.trainer import execute


@pytest.fixture
def cfg():
    torch.set_num_threads(1)
    return configuration(ROOT / 'configs/sem_ssl/nffa_200step.yaml')


def small_backbone(**_kwargs):
    from dinov2.models.vision_transformer import DinoVisionTransformer
    return DinoVisionTransformer(img_size=56, patch_size=14, embed_dim=24, depth=1, num_heads=3,
                                 block_chunks=0, init_values=1, drop_path_rate=.15)


@pytest.fixture
def fixture(cfg, tmp_path, monkeypatch):
    import dinov2.hub.backbones
    monkeypatch.setattr(dinov2.hub.backbones, 'dinov2_vits14', small_backbone)
    cfg.crops.global_crops_size, cfg.crops.local_crops_size, cfg.crops.local_crops_number = 56, 28, 2
    cfg.train.batch_size_per_gpu, cfg.train.num_workers = 2, 0
    cfg.dino.head_n_prototypes, cfg.dino.head_hidden_dim, cfg.dino.head_bottleneck_dim = 32, 64, 16
    cfg.ibot.mask_ratio_min_max = [.4, .5]
    cfg.experiment.schedule_total_steps = 6
    cfg.experiment.run_until_step = 6
    cfg.experiment.warmup_steps = 2
    cfg.experiment.freeze_last_layer_steps = 1
    cfg.experiment.teacher_temp_warmup_steps = 2
    cfg.experiment.milestones = [0, 2, 4, 6]
    cfg.experiment.recovery_interval = 2
    checkpoint = tmp_path / 'initialization.pth'
    torch.manual_seed(7)
    torch.save(small_backbone().state_dict(), checkpoint)
    records = []
    for i in range(7):
        path = tmp_path / f'image{i}.png'
        Image.fromarray(np.random.default_rng(i).integers(0, 256, (81, 97, 3), dtype=np.uint8)).save(path)
        records.append(dict(image_id=path.name, group_id=str(i)))
    transform, collate = augmentation_and_collate(cfg)
    dataset = SEMTrainDataset(tmp_path, records, transform)
    identity = dict(records=records, count=len(records), provider='NONSCIENTIFIC_SYNTHETIC_FIXTURE')
    return cfg, dataset, collate, identity, checkpoint


def test_phase_a_original_augmentations_and_hyperparameters(cfg):
    from dinov2.data import DataAugmentationDINO
    transform, _ = augmentation_and_collate(cfg)
    assert type(transform) is DataAugmentationDINO
    assert transform.global_crops_size == 224 and transform.local_crops_size == 98
    assert transform.local_crops_number == 8
    assert list(transform.global_crops_scale) == [.32, 1]
    assert list(transform.local_crops_scale) == [.05, .32]
    assert transform.geometric_augmentation_global.transforms[1].p == .5
    jitter = transform.global_transfo1.transforms[0].transforms
    assert jitter[0].p == .8 and jitter[1].p == .2
    assert transform.global_transfo2.transforms[1].transforms[1].p == .2
    normalized = transform.normalize.transforms[-1]
    assert list(normalized.mean) == [.485, .456, .406]
    assert list(normalized.std) == [.229, .224, .225]
    assert cfg.optim.lr == pytest.approx(.0002 * (cfg.train.batch_size_per_gpu / 1024)**.5)
    assert cfg.teacher.momentum_teacher == .996


def test_exact_original_step_schedules_and_stop_independence(cfg):
    sched = step_schedules(cfg)
    assert all(len(value.schedule) == 200 for value in sched.values())
    assert all(type(v).__module__ == 'dinov2.utils.utils' for v in sched.values())
    assert sched['lr'][0] == 0
    assert sched['last_layer_lr'][9] == 0 and sched['last_layer_lr'][10] > 0
    assert sched['teacher_temp'][0] == .04 and sched['teacher_temp'][19] == .07
    cfg.experiment.run_until_step = 3
    for key, value in step_schedules(cfg).items():
        np.testing.assert_array_equal(value.schedule, sched[key].schedule)


def test_rng_scope_does_not_change_model_rng():
    before = rng_state()
    with data_rng(123):
        random.random(); np.random.rand(); torch.rand(10)
    after = rng_state()
    assert before['python'] == after['python']
    assert np.array_equal(before['numpy'][1], after['numpy'][1])
    assert torch.equal(before['torch'], after['torch'])


def test_seekable_shuffle_and_epoch_drop_last():
    complete = list(CursorBatches(7, 2, 42, 0, 10))
    assert complete[4:] == list(CursorBatches(7, 2, 42, 4, 10))
    assert len({index for batch in complete[:3] for index, _ in batch}) == 6
    assert [draw for batch in complete for _, draw in batch] == list(range(20))
    assert complete != list(CursorBatches(7, 2, 43, 0, 10))


def test_augmentation_and_masking_seek_exactly(fixture):
    cfg, dataset, collate, _, _ = fixture
    def rows(start, workers=0):
        return list(DataLoader(DrawDataset(dataset, 42), batch_sampler=CursorBatches(7, 2, 42, start, 4),
                              collate_fn=DrawCollate(collate, 42), num_workers=workers,
                              generator=torch.Generator().manual_seed(42)))
    full, resumed = rows(0), rows(2)
    for a, b in zip(full[2:], resumed):
        for key in a:
            if isinstance(a[key], torch.Tensor):
                assert torch.equal(a[key], b[key]), key
            else:
                assert a[key] == b[key]


def test_worker_count_does_not_change_draws(fixture):
    from multiprocessing.connection import Listener
    # Torch's tensor FD transfer starts this listener in a queue feeder thread;
    # test capability here so sandbox denial cannot become a misleading timeout.
    try:
        listener = Listener(authkey=b'sem-ssl-test')
        listener.close()
    except PermissionError:
        pytest.skip('Sandbox denies multiprocessing tensor-transfer listener')
    _, dataset, collate, _, _ = fixture
    def loader(workers):
        return DataLoader(DrawDataset(dataset, 42), batch_sampler=CursorBatches(7, 2, 42, 2, 4),
                          collate_fn=DrawCollate(collate, 42), num_workers=workers,
                          generator=torch.Generator().manual_seed(42),
                          **(dict(multiprocessing_context='spawn', timeout=60) if workers else {}))
    try:
        parallel = list(loader(2))
    except PermissionError:
        pytest.skip('Sandbox does not permit multiprocessing semaphores')
    for a, b in zip(loader(0), parallel):
        assert torch.equal(a['collated_global_crops'], b['collated_global_crops'])
        assert torch.equal(a['collated_masks'], b['collated_masks'])


def test_resume_bitwise_optimizer_ema_centers_and_metrics(fixture, tmp_path):
    cfg, dataset, collate, identity, initialization = fixture
    full, interrupted = tmp_path / 'full', tmp_path / 'interrupted'
    model_full, optimizer_full = execute(cfg, dataset, collate, identity, initialization, full)
    execute(cfg, dataset, collate, identity, initialization, interrupted, interrupt_after=3)
    assert json.loads((interrupted / 'status.json').read_text())['status'] == 'interrupted'
    model_resume, optimizer_resume = execute(cfg, dataset, collate, identity, initialization, interrupted, resume=True)
    for key, value in model_full.state_dict().items():
        assert torch.equal(value, model_resume.state_dict()[key]), key
    for key, value in optimizer_full.state_dict()['state'].items():
        for name, tensor in value.items():
            assert torch.equal(tensor, optimizer_resume.state_dict()['state'][key][name])
    for folder in (full, interrupted):
        assert [p.name for p in sorted((folder / 'checkpoints').glob('step_*'))] == [f'step_{s:06d}' for s in (0, 2, 4, 6)]
    full_metrics = [json.loads(line) for line in (full / 'metrics.jsonl').read_text().splitlines()]
    resume_metrics = [json.loads(line) for line in (interrupted / 'metrics.jsonl').read_text().splitlines()]
    for a, b in zip(full_metrics, resume_metrics):
        for key in ('total_loss', 'dino_local_crops_loss', 'ibot_loss', 'gradient_norm', 'record_indices', 'draw_ids'):
            assert a[key] == b[key], key
    latest, payload, _ = find_latest(interrupted)
    export = torch.load(latest / 'teacher_backbone.pth', weights_only=True)
    fresh = small_backbone().eval()
    fresh.load_state_dict(export, strict=True)
    inputs = torch.randn(1, 3, 56, 56)
    assert torch.equal(fresh(inputs), model_resume.teacher.backbone(inputs))
    assert payload['data_cursor']['image_presentations'] == 12
    assert all(p.grad is None for p in model_resume.teacher.parameters())
    from scripts.verify_sem_ssl import verify
    assert verify(latest)['forward_max_abs_error'] == 0


def test_planned_horizon_resume_and_contract_rejection(fixture, tmp_path):
    cfg, dataset, collate, identity, initialization = fixture
    root = tmp_path / 'run'
    cfg.experiment.run_until_step = 2
    execute(cfg, dataset, collate, identity, initialization, root)
    cfg.experiment.run_until_step = 6
    execute(cfg, dataset, collate, identity, initialization, root, resume=True)
    with pytest.raises(FileExistsError):
        execute(cfg, dataset, collate, identity, initialization, root)
    altered = copy.deepcopy(identity)
    altered['records'][0]['image_id'] = 'another'
    with pytest.raises(ValueError, match='differs'):
        execute(cfg, dataset, collate, altered, initialization, root, resume=True)
    cfg.experiment.schedule_total_steps = 10
    with pytest.raises(ValueError, match='differs'):
        execute(cfg, dataset, collate, identity, initialization, root, resume=True)


def test_corrupt_latest_fallback_and_metrics_tail(fixture, tmp_path):
    cfg, dataset, collate, identity, initialization = fixture
    root = tmp_path / 'run'
    execute(cfg, dataset, collate, identity, initialization, root)
    (root / 'checkpoints/step_000006/training.pth').write_bytes(b'corrupt')
    latest, payload, rejected = find_latest(root)
    assert payload['step'] == 4 and rejected
    with (root / 'metrics.jsonl').open('a') as stream:
        stream.write('{partial')
    reconcile_metrics(root, 4)
    assert len((root / 'metrics.jsonl').read_text().splitlines()) == 4
    assert list(root.glob('metrics_before_resume_*.jsonl'))
    assert not list(root.glob('checkpoints/.pending_*'))


def test_nonfinite_stops_without_partial_checkpoint(fixture, tmp_path):
    cfg, dataset, collate, identity, initialization = fixture
    root = tmp_path / 'run'
    with mock.patch('src.ssl.trainer.train_step', side_effect=FloatingPointError('injected nonfinite')):
        with pytest.raises(FloatingPointError):
            execute(cfg, dataset, collate, identity, initialization, root)
    _, saved, _ = find_latest(root)
    assert saved['step'] == 0
    assert json.loads((root / 'status.json').read_text())['status'] == 'failed'


def test_exact_200_updates_and_milestone_retention(fixture, tmp_path):
    cfg, dataset, collate, identity, initialization = fixture
    cfg.experiment.schedule_total_steps = cfg.experiment.run_until_step = 200
    cfg.experiment.milestones = [0, 50, 100, 200]
    cfg.experiment.recovery_interval = 25
    cfg.experiment.log_interval = 100
    root = tmp_path / 'synthetic_200'
    execute(cfg, dataset, collate, identity, initialization, root)
    assert len((root / 'metrics.jsonl').read_text().splitlines()) == 200
    assert len(list((root / 'checkpoints').glob('step_*'))) == 4
    assert len(list((root / 'recovery').glob('step_*'))) == 2
    assert json.loads((root / 'status.json').read_text())['step'] == 200


def test_selected_images_only_opened_and_hashed(cfg, tmp_path):
    def csv_file(path, rows):
        with path.open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    manifest, primary, holdout = [tmp_path / name for name in ('manifest.csv', 'B.csv', 'A.csv')]
    names = ['train.png', 'val_missing.png', 'test_missing.png']
    csv_file(manifest, [dict(image_id=n, included='1', crop_mode='crop_v2') for n in names])
    rows = [dict(image_id=n, split=s, group_id=str(i)) for i, (n, s) in enumerate(zip(names, ['train', 'val', 'test']))]
    csv_file(primary, rows); csv_file(holdout, rows)
    Image.new('L', (60, 60), 100).save(tmp_path / 'train.png')
    cfg.dataset.root = str(tmp_path)
    cfg.dataset.manifest, cfg.dataset.primary_split, cfg.dataset.holdout_splits = str(manifest), str(primary), [str(holdout)]
    dataset, _, identity = dataset_plan(cfg, 2)
    assert identity['count'] == 1 and dataset[0][1] == ()
    assert [r['image_id'] for r in identity['records']] == ['train.png']


def test_aggregation_keeps_missing_null_and_rejects_protocol_mismatch(tmp_path):
    from src.ssl.reporting import collect, render_report
    from src.ssl.upstream import PIN
    fake = dict(nffa_A_logreg=dict(macro_f1=.9, accuracy=.91))
    with mock.patch('src.ssl.reporting.result_bundle', return_value=(fake, {'key': 'same'}, [], {'weights'})):
        result = collect(tmp_path / 'not_started', tmp_path / 'eval')
    assert result['rows'][0]['metrics']['nffa_A_logreg']['macro_f1'] == .9
    assert result['rows'][1]['metrics']['nffa_A_logreg'] is None
    assert not result['complete']
    run = tmp_path / 'run'
    for step in (0, 50):
        (run / 'checkpoints' / f'step_{step:06d}').mkdir(parents=True)
    meta = dict(files={'teacher_backbone.pth': PIN['backbone_sha256']}, contract_sha256='same-ssl-contract')
    bundles = [(fake, {'key': 'first'}, [], {'weights'}),
               ({}, {}, [], {'weights'}),
               (fake, {'key': 'different'}, [], {'weights'})]
    with mock.patch('src.ssl.reporting.validate_snapshot', return_value=meta), \
            mock.patch('src.ssl.reporting.result_bundle', side_effect=bundles):
        with pytest.raises(ValueError, match='protocol changed'):
            collect(run, tmp_path / 'eval', steps=(0, 50))


def test_evaluation_commands_delegate_original_heads_and_preserve_baseline_path(tmp_path):
    from scripts.evaluate_sem_ssl import commands
    from src.ssl.experiment import REFERENCE
    from src.ssl.upstream import PIN
    meta = dict(files={'teacher_backbone.pth': PIN['backbone_sha256']})
    with mock.patch('scripts.evaluate_sem_ssl.validate_snapshot', return_value=meta), \
            mock.patch('scripts.evaluate_sem_ssl.sha256', return_value=PIN['backbone_sha256']):
        rows = list(commands(tmp_path, [0, 50], 'classification', 'all', tmp_path / 'eval'))
    assert str(REFERENCE.resolve()) in rows[0][2]
    assert str((tmp_path / 'checkpoints/step_000050/teacher_backbone.pth').resolve()) in rows[1][2]
    assert rows[0][2][rows[0][2].index('--heads') + 1:rows[0][2].index('--heads') + 3] == ['logreg', 'mlp']
