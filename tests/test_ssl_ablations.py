"""CPU-only control, freezing, recovery, inventory reuse and study reporting checks."""
import copy
import csv
import json
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image
import pytest
import torch
from omegaconf import OmegaConf

from scripts import run_sem_ssl_ablations as study
from scripts.verify_sem_ssl import verify
from src.ssl.adapter import SingleGPUAdapter, train_step
from src.ssl import ablation_data
from src.ssl.data import SEMTrainDataset, augmentation_and_collate
from src.ssl.experiment import configuration, step_schedules
from src.ssl.trainer import execute
from src.ssl.upstream import ROOT


def tiny_backbone(**_kwargs):
    from dinov2.models.vision_transformer import DinoVisionTransformer
    return DinoVisionTransformer(img_size=56, patch_size=14, embed_dim=24, depth=12, num_heads=3,
                                 block_chunks=0, init_values=1, drop_path_rate=.1)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    import dinov2.hub.backbones
    torch.set_num_threads(1)
    monkeypatch.setattr(dinov2.hub.backbones, 'dinov2_vits14', tiny_backbone)
    cfg = configuration(ROOT/'configs/sem_ssl/nffa_ablation_low_lr_freeze8.yaml')
    cfg.crops.global_crops_size, cfg.crops.local_crops_size, cfg.crops.local_crops_number = 56, 28, 2
    cfg.train.batch_size_per_gpu, cfg.train.num_workers = 2, 0
    cfg.dino.head_n_prototypes, cfg.dino.head_hidden_dim, cfg.dino.head_bottleneck_dim = 32, 64, 16
    cfg.experiment.schedule_total_steps = cfg.experiment.run_until_step = 6
    cfg.experiment.warmup_steps = cfg.experiment.freeze_last_layer_steps = 1
    cfg.experiment.teacher_temp_warmup_steps = 2
    cfg.experiment.milestones, cfg.experiment.recovery_interval = [0, 3, 6], 2
    cfg.optim.lr, cfg.optim.min_lr = .001, .0001
    initialization = tmp_path/'initialization.pth'
    torch.manual_seed(7)
    torch.save(tiny_backbone().state_dict(), initialization)
    records = []
    for i in range(5):
        name = f'{i}.png'
        Image.fromarray(np.random.default_rng(i).integers(0, 180, (64, 84, 3), np.uint8)).save(tmp_path/name)
        records.append(dict(image_id=name, group_id=str(i)))
    transform, collate = augmentation_and_collate(cfg)
    dataset = SEMTrainDataset(tmp_path, records, transform)
    identity = dict(provider='NONSCIENTIFIC_SYNTHETIC_FIXTURE', count=5, records=records)
    return cfg, dataset, collate, identity, initialization


def test_exact_lr_scaling_and_same_5000_step_non_lr_schedules():
    base = configuration(ROOT/'configs/sem_ssl/nffa_5000step.yaml')
    reference = step_schedules(base)
    for _, suffix, factor, frozen in study.EXPERIMENTS:
        cfg = configuration(ROOT/f'configs/sem_ssl/nffa_ablation_{suffix}.yaml')
        assert cfg.experiment.schedule_total_steps == 5000 and cfg.experiment.run_until_step == 1000
        assert list(cfg.experiment.milestones) == [0, 500, 1000]
        assert cfg.train.seed == 42 and cfg.train.batch_size_per_gpu == 64
        assert cfg.student.freeze_first_blocks == frozen
        assert cfg.optim.lr == pytest.approx(5e-5*factor)
        assert cfg.optim.min_lr == pytest.approx(1e-6*factor)
        for key, value in step_schedules(cfg).items():
            expected = reference[key].schedule*(factor if key in ('lr', 'last_layer_lr') else 1)
            np.testing.assert_allclose(value.schedule, expected, rtol=1e-14, atol=1e-20)


def test_pretrained_backbone_and_random_head_are_copied_and_seed_matched(fixture):
    cfg, _, _, _, initialization = fixture
    expected = torch.load(initialization, weights_only=True)
    initial_states = []
    for frozen in (0, 8):
        cfg.student.freeze_first_blocks = frozen
        torch.manual_seed(42)
        model = SingleGPUAdapter(cfg, initialization)
        assert model.pretrained_verified
        assert all(torch.equal(v, expected[k]) for k, v in model.student.backbone.state_dict().items())
        assert all(torch.equal(v, model.teacher.state_dict()[k]) for k, v in model.student.state_dict().items())
        assert not model.teacher.training and all(not p.requires_grad for p in model.teacher.parameters())
        initial_states.append({k: v.clone() for k, v in model.student.state_dict().items()})
    assert all(torch.equal(v, initial_states[1][k]) for k, v in initial_states[0].items())


def test_frozen_blocks_have_no_grad_optimizer_or_ema_updates(fixture):
    cfg, dataset, collate, _, initialization = fixture
    model = SingleGPUAdapter(cfg, initialization)
    initial = {k: v.clone() for k, v in model.student.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameter_groups())
    optimized = {id(p) for group in optimizer.param_groups for p in group['params']}
    frozen = list(model.student.backbone.blocks[:8].parameters())
    assert all(not p.requires_grad and id(p) not in optimized for p in frozen)
    batch = collate([dataset[0], dataset[1]])
    sched = {k: [v] for k, v in dict(lr=.001, last_layer_lr=.001, weight_decay=.1,
                                     momentum=.996, teacher_temp=.04).items()}
    row = train_step(model, optimizer, batch, sched, 0)
    assert np.isfinite(row['total_loss']) and all(p.grad is None for p in frozen)
    for momentum in (.996, .9997, .5, 0., 1.):
        model.update_teacher(momentum)
    prefixes = tuple(f'backbone.blocks.{i}.' for i in range(8))
    for network in (model.student, model.teacher):
        for key, value in network.state_dict().items():
            if key.startswith(prefixes):
                assert torch.equal(value, initial[key]), key
    for group in ('backbone.blocks.8.', 'backbone.patch_embed.', 'dino_head.'):
        assert any(not torch.equal(v, initial[k]) for k, v in model.student.state_dict().items() if k.startswith(group))


def test_frozen_resume_is_bitwise_and_export_verified(fixture, tmp_path):
    cfg, dataset, collate, identity, initialization = fixture
    full, interrupted = tmp_path/'full', tmp_path/'interrupted'
    expected, _ = execute(cfg, dataset, collate, identity, initialization, full)
    execute(cfg, dataset, collate, identity, initialization, interrupted, interrupt_after=3)
    actual, _ = execute(cfg, dataset, collate, identity, initialization, interrupted, resume=True)
    assert all(torch.equal(v, actual.state_dict()[k]) for k, v in expected.state_dict().items())
    receipt = verify(interrupted/'checkpoints/step_000006', initialization=initialization)
    assert receipt['frozen_blocks'] == 8 and receipt['frozen_blocks_unchanged']
    assert receipt['forward_max_abs_error'] == 0
    assert json.loads((interrupted/'status.json').read_text())['status'] == 'completed'


def test_verified_data_readers_never_open_images_or_hash_masks(tmp_path, monkeypatch):
    manifest, split, reference = tmp_path/'manifest.csv', tmp_path/'split.csv', tmp_path/'preflight.json'
    manifest.write_text('existing verified manifest')
    split.write_text('existing verified split')
    image_root = tmp_path/'absent_images'
    records = [dict(image_id=str(i), split=s, label=str(i), group_id=str(i), image_sha256='stored')
               for i, s in enumerate(('train', 'val'))]
    expected_hash = ablation_data.sha256_file
    reference.write_text(json.dumps(dict(test_files_opened=False, counts=dict(train=1, val=1), contract=dict(
        nonscientific_smoke_test=False, manifest_sha256=expected_hash(manifest), split_sha256=expected_hash(split),
        image_root=str(image_root.resolve()), records=records))))
    def only_manifests(path):
        assert Path(path) in (manifest, split), 'Dataset pixel file rehashed'
        return expected_hash(path)
    monkeypatch.setattr(ablation_data, 'sha256_file', only_manifests)
    assert ablation_data.classification(reference, manifest, split, image_root) == records
    from src.aachen_ablation.geometry import VARIANTS
    segrows = [dict(image_id=r['image_id'], split=r['split'], group_id=r['group_id'],
                   image_path='missing'+r['image_id'], mask_path='missing_mask'+r['image_id']) for r in records]
    segrecords = [dict(r, original_size=[100, 75], image_sha256='stored', mask_sha256='stored') for r in segrows]
    config = tmp_path/'config.json'
    config.write_text(json.dumps(dict(cache_metadata=dict(contract=dict(nonscientific_smoke_test=False,
        manifest_sha256=expected_hash(manifest), records=segrecords)))))
    actual, info = ablation_data.segmentation(config, manifest, segrows, [VARIANTS['A3']])
    assert actual == segrecords and info['test_files_opened'] is False
    split.write_text('modified')
    with pytest.raises(ValueError, match='does not match'):
        ablation_data.classification(reference, manifest, split, image_root)
    poison = copy.deepcopy(segrows)
    poison[0]['split'] = 'test'
    with pytest.raises(ValueError, match='differ'):
        ablation_data.segmentation(config, manifest, poison, [VARIANTS['A3']])


def test_sequential_orchestration_resume_and_interruption(tmp_path, monkeypatch):
    plan = dict(contracts={label: dict(label=label) for label, *_ in study.EXPERIMENTS})
    seen = []
    monkeypatch.setattr(study, 'collect_rows', lambda *_args, **_kwargs: [])
    monkeypatch.setattr(study, 'validate_snapshot', lambda *_args: None)
    report = mock.Mock()
    monkeypatch.setattr(study, 'write_report', report)
    def runner(command, _log):
        seen.append(command)
        if command[2] == 'scripts.train_sem_ssl':
            out = Path(command[command.index('--output')+1]);out.mkdir(parents=True, exist_ok=True)
            label = out.parent.name
            (out/'run.json').write_text(json.dumps(dict(contract=plan['contracts'][label])))
            (out/'status.json').write_text(json.dumps(dict(status='completed', step=1000)))
        elif command[2] == 'scripts.verify_sem_ssl':
            out = Path(command[command.index('--output')+1])
            frozen=next(f for label, _, _, f in study.EXPERIMENTS if label==out.parent.name)
            out.write_text(json.dumps(dict(passed=True, checkpoints=[dict(step=s, frozen_blocks=frozen,
                           frozen_blocks_unchanged=bool(frozen)) for s in study.STEPS])))
    output = tmp_path/'study'
    study.run_study(output, plan, {}, runner)
    modules = [c[2] for c in seen]
    assert modules[:6] == ['scripts.train_sem_ssl', 'scripts.verify_sem_ssl']*3
    assert modules[6:] == ['scripts.evaluate_sem_encoder']*12
    for command in seen[6:]:
        assert '--resume' in command and '--verified-data-root' in command
        task = command[command.index('--task')+1]
        assert command[command.index('--workers')+1] == ('8' if task == 'classification' else '16')
    seen.clear()
    study.run_study(output, plan, {}, runner)
    assert len(seen) == 12 and all(c[2] == 'scripts.evaluate_sem_encoder' for c in seen)
    changed = copy.deepcopy(plan); changed['new_source'] = True
    with pytest.raises(ValueError, match='changed'):
        study.run_study(output, changed, {}, runner)
    # A graceful trainer stop must prevent verification/evaluation and later runs.
    seen.clear()
    def stop(command, _log):
        runner(command, _log)
        out = Path(command[command.index('--output')+1])
        (out/'status.json').write_text(json.dumps(dict(status='interrupted', step=250)))
    with pytest.raises(InterruptedError):
        study.run_study(tmp_path/'stopped', plan, {}, stop)
    assert len(seen) == 1


def result_rows():
    rows = []
    for label, steps in [('reference', study.STEPS), *[(r[0], (500, 1000)) for r in study.EXPERIMENTS]]:
        for step in steps:
            metrics = {k: dict(macro_f1=.9, accuracy=.91) for k in study.KEYS[:4]}
            metrics.update({k: dict(fixed=dict(ma_iou=.4, miou=.65, threshold=.5),
                                     selected=dict(ma_iou=.42, miou=.66, threshold=.35)) for k in study.KEYS[4:]})
            rows.append(dict(experiment=label, step=step, metrics=metrics, teacher_sha256='fixture'))
    return rows


def test_report_serialization_recommendation_and_no_mixed_thresholds(tmp_path):
    rows = result_rows()
    chosen = next(r for r in rows if r['experiment'] == 'A_low_lr' and r['step'] == 500)
    chosen['metrics']['nffa_A_logreg']['macro_f1'] += .01
    assert 'A_low_lr' in study.recommendation(rows)['decision']
    study.write_report(tmp_path, rows)
    result = json.loads((tmp_path/'report/comparison.json').read_text())
    assert result['test_evaluated'] is False and result['partition'] == 'val'
    assert result['recommendation']['best_by_metric']['nffa_A_logreg']['experiment'] == 'A_low_lr'
    with (tmp_path/'report/comparison.csv').open() as stream:
        table = list(csv.DictReader(stream))
    assert len(table) == 9
    assert float(table[3]['nffa_A_logreg_delta_original_pp']) == pytest.approx(1.)
    assert float(table[3]['aachen_cnn8_selected_ma_iou']) == .42
    assert float(table[3]['aachen_cnn8_fixed_ma_iou']) == .4
    study.write_report(tmp_path, rows)  # safe, identical report reuse
    chosen['metrics']['nffa_B_logreg']['macro_f1'] -= .1
    with pytest.raises(FileExistsError, match='differs'):
        study.write_report(tmp_path, rows)
    # High selected-threshold IoU alone must not produce a universal recommendation.
    rows = result_rows()
    rows[3]['metrics']['aachen_cnn8']['selected']['ma_iou'] = .99
    assert study.recommendation(rows)['decision'].startswith('Die Originalgewichte')


def test_protocol_excludes_only_the_explicit_inventory_cli_source(tmp_path):
    from src.ssl.reporting import result_bundle
    from src.ssl.upstream import PIN
    from src.aachen_ablation.data import digest
    from src.downstream.probes import HEAD_CONFIGS
    encoder = dict(checkpoint_sha256='checkpoint', upstream_commit=PIN['commit'])
    def write(source, preprocessing):
        for split in ('A', 'B'):
            base = tmp_path/f'classification/split_{split}'
            contract = dict(encoder=encoder, sources={'scripts/evaluate_sem_encoder.py': source,
                                                      'src/preprocessing_v2.py': preprocessing}, records=['verified'])
            for head in ('logreg', 'mlp'):
                folder=base/head/'seed42';folder.mkdir(parents=True, exist_ok=True)
                config=dict(seed=42, head_config=HEAD_CONFIGS[head], cache_contract_sha256=digest(contract))
                metric=dict(encoder=encoder, partition='val', test_evaluated=False, seed=42, head=head,
                            split=split, head_config=HEAD_CONFIGS[head], encoder_state_sha256='weights',
                            macro_f1=.9, accuracy=.9)
                (folder/'config.json').write_text(json.dumps(config))
                (folder/'validation_metrics.json').write_text(json.dumps(metric))
            (base/'preflight.json').write_text(json.dumps(dict(contract=contract)))
    def protocols(exclude):
        return result_bundle(tmp_path, 'checkpoint', 'classification', check_sources=False,
                             protocol_source_exclusions=exclude)[1]
    write('old CLI', 'same preprocessing')
    old_default, old_excluded = protocols(()), protocols(study.SOURCE_EXCLUSIONS)
    write('new inventory CLI', 'same preprocessing')
    assert protocols(()) != old_default
    assert protocols(study.SOURCE_EXCLUSIONS) == old_excluded
    write('new inventory CLI', 'changed preprocessing')
    assert protocols(study.SOURCE_EXCLUSIONS) != old_excluded


def test_verified_data_option_reaches_existing_aachen_runner(tmp_path):
    from scripts import evaluate_sem_encoder
    from scripts import run_aachen_ablation
    reference=tmp_path/'reference'
    config=reference/'segmentation/runs/A3/linear/seed42_fixture/config.json'
    config.parent.mkdir(parents=True);config.write_text('{}')
    args=evaluate_sem_encoder.parser().parse_args(['--task','segmentation','--verified-data-root',str(reference)])
    with mock.patch.object(run_aachen_ablation,'main') as called:
        evaluate_sem_encoder.run_segmentation(args,tmp_path/'new',tmp_path/'checkpoint.pth',['linear','cnn8'])
    command=called.call_args.args[0]
    assert command[command.index('--verified-data')+1]==str(config)
