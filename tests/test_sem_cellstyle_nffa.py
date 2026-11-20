"""NFFA pilot wiring, using synthetic manifests and no held-out image reads."""
import copy
import csv
import json
from pathlib import Path
from unittest import mock

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import pytest
import torch

from src.sem_cellstyle.data import provider, SEMViews
from src.sem_cellstyle.recipe import budgets, configuration
from src.ssl.data import sha256, train_records
from src.ssl.experiment import digest
from src.ssl.ablation_data import nffa_training_inventory


@pytest.fixture
def nffa(tmp_path):
    cfg = configuration()
    cfg.dataset.provider = 'verified_nffa_inventory_v1'
    cfg.dataset.root = str(tmp_path)
    cfg.crops.global_crops_size = 28
    cfg.crops.local_crops_size = 14
    ids = ['a.png', 'b.png', 'f.png', 'c.png', 'd.png', 'e.png']

    def write(name, rows):
        path = tmp_path / name
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        return path

    manifest = write('manifest.csv', [dict(image_id=i, included='1', crop_mode='crop_v2') for i in ids])
    primary = write('B.csv', [dict(image_id=i, group_id='shared' if i in ids[1:3] else i,
                                  split='val' if i == 'c.png' else 'test' if i == 'd.png' else 'train') for i in ids])
    holdout = write('A.csv', [dict(image_id=i, group_id=i,
                                  split='val' if i in ('f.png', 'c.png') else 'test' if i == 'd.png' else 'train') for i in ids])
    selected = train_records(manifest, primary, [holdout])
    assert [r['image_id'] for r in selected] == ['a.png', 'e.png']
    for row in selected:
        path = tmp_path / row['image_id']
        Image.fromarray(np.random.default_rng(42).integers(0,256,(53,77,3),dtype=np.uint8)).save(path)
        row.update(bytes=path.stat().st_size, sha256=sha256(path))
    inventory = dict(provider='nffa_v3', root=str(tmp_path), count=len(selected), records=selected,
                     records_sha256=digest(selected), manifests=[dict(path=str(p), sha256=sha256(p))
                     for p in (manifest, primary, holdout)])
    path = tmp_path / 'run.json'
    path.write_text(json.dumps(dict(dataset=inventory)))
    cfg.dataset.update(dict(manifest=str(manifest), primary_split=str(primary),
                            holdout_splits=[str(holdout)], inventory=str(path)))
    return cfg, inventory, path


def test_nffa_reuses_exact_safe_selection_without_rehashing_images(nffa):
    cfg, inventory, path = nffa
    with mock.patch('src.ssl_benchmark.sha256', wraps=sha256) as hashing, \
         mock.patch('src.sem_cellstyle.data.sha256', wraps=sha256) as identity_hashing:
        dataset, collate, identity = provider(cfg)
        assert all(call.args[0].suffix == '.csv' for call in hashing.call_args_list)
        assert [call.args[0] for call in identity_hashing.call_args_list] == [path]
    assert len(dataset) == identity['count'] == 2
    assert dataset.records == [{k:r[k] for k in ('image_id', 'group_id')} for r in inventory['records']]
    assert identity['audited_inventory'] == inventory
    assert identity['image_contents_rehashed'] is False
    assert identity['orientation_invariant'] is False
    assert cfg.dataset.provider == 'verified_nffa_inventory_v1'
    assert isinstance(dataset.transform, SEMViews)
    with mock.patch('src.ssl.loading.crop_metadata_strip', side_effect=lambda image: image.crop((0,0,77,40))) as strip, \
         mock.patch('src.sem_cellstyle.data.ImageOps.mirror') as mirror:
        samples = [dataset[0], dataset[1]]
        batch = collate(samples)
    assert strip.call_count == 2 and not mirror.called
    assert batch['collated_global_crops'].shape == (4,3,28,28)
    assert batch['collated_local_crops'].shape == (16,3,14,14)
    assert batch['collated_masks'].shape == (4,4)


@pytest.mark.parametrize('change', ['manifest', 'records', 'selection', 'count', 'root', 'holdout'])
def test_nffa_rejects_changed_verification_or_selection(nffa, change):
    cfg, inventory, path = nffa
    if change == 'manifest':
        with Path(cfg.dataset.manifest).open('a') as stream: stream.write('\n')
    elif change == 'records': inventory['records'][0]['sha256'] = '0'*64
    elif change == 'selection':
        inventory['records'][0]['image_id'] = 'c.png'
        inventory['records_sha256'] = digest(inventory['records'])
    elif change == 'count': inventory['count'] += 1
    elif change == 'root': cfg.dataset.root = str(path.parent / 'other')
    elif change == 'holdout': cfg.dataset.holdout_splits = []
    path.write_text(json.dumps(inventory))
    with pytest.raises(ValueError): provider(cfg)


def test_nffa_pilot_keeps_large_scientific_recipe_and_actual_budget():
    pilot = configuration('configs/sem_ssl/sem_cellstyle_nffa_vitl14_pilot.yaml')
    original = configuration()
    left, right = [OmegaConf.to_container(copy.deepcopy(c)) for c in (pilot, original)]
    for value in (left, right):
        for key in ('dataset', 'initialization'): value.pop(key)
        value['runtime'].pop('output'); value['evaluation'].pop('verified_data_root')
        value['runtime'].pop('allow_cuda_resume_update_drift', None)
        value['evaluation'].pop('classification_splits', None)
        value['evaluation'].pop('head_profile', None)
    assert left == right
    assert pilot.runtime.allow_cuda_resume_update_drift is True
    assert not original.runtime.get('allow_cuda_resume_update_drift', False)
    assert pilot.dataset.selection_protocol == 'split_A_train_v1'
    assert not pilot.dataset.holdout_splits
    assert Path(pilot.dataset.primary_split).name == 'split_A_exact.csv'
    assert list(pilot.evaluation.classification_splits) == ['A']
    assert pilot.evaluation.head_profile == 'residual_v1'
    actual = budgets(pilot, 12801)
    assert actual['schedule_total_steps'] == 10001
    assert actual['run_until_step'] == 1001
    assert actual['milestones'] == [0,101,501,1001,2501,5001,10001]
    assert (actual['warmup_steps'], actual['freeze_last_layer_steps'],
            actual['teacher_temp_warmup_steps']) == (501,101,2001)


@pytest.fixture
def split_a_nffa(nffa):
    cfg, _, old = nffa
    split = Path(cfg.dataset.holdout_splits[0])
    with split.open(newline='') as stream:
        rows = [r for r in csv.DictReader(stream) if r['split'] != 'test']
    Image.new('RGB', (77,53)).save(old.parent/'b.png')
    for row in rows:
        path = old.parent/row['image_id']
        row.update(label='0',image_sha256=sha256(path) if row['split']=='train' else 'f'*64)
    reference = old.parent/'classification_preflight.json'
    proof = dict(test_files_opened=False,counts={'train':3,'val':2},contract=dict(
        nonscientific_smoke_test=False,manifest_sha256=sha256(Path(cfg.dataset.manifest)),
        split_sha256=sha256(split),image_root=cfg.dataset.root,records=rows))
    reference.write_text(json.dumps(proof))
    cfg.dataset.primary_split = str(split)
    cfg.dataset.holdout_splits = []
    cfg.dataset.selection_protocol = 'split_A_train_v1'
    cfg.evaluation.classification_splits = ['A']
    inventory = nffa_training_inventory(reference, Path(cfg.dataset.manifest), split, Path(cfg.dataset.root))
    path = old.parent/'split_a_inventory.json'; path.write_text(json.dumps(inventory))
    cfg.dataset.inventory = str(path)
    return cfg, reference, inventory


def test_split_a_uses_all_a_train_and_reuses_previous_hashes(split_a_nffa):
    cfg, reference, inventory = split_a_nffa
    with mock.patch('src.ssl.ablation_data.sha256_file',wraps=sha256) as hashing:
        rebuilt = nffa_training_inventory(reference, Path(cfg.dataset.manifest),
                                          Path(cfg.dataset.primary_split), Path(cfg.dataset.root))
        assert {call.args[0].suffix for call in hashing.call_args_list} <= {'.csv','.json'}
    assert rebuilt == inventory
    dataset, _, identity = provider(cfg)
    assert [r['image_id'] for r in dataset.records] == ['a.png','b.png','e.png']
    assert identity['selection_protocol'] == 'split_A_train_v1'
    assert identity['image_contents_rehashed'] is False
    assert identity['count'] == 3


def test_split_a_rejects_b_evaluation_and_old_intersection_inventory(split_a_nffa):
    cfg, _, _ = split_a_nffa
    cfg.evaluation.classification_splits = ['A','B']
    with pytest.raises(ValueError,match='B holdouts overlap'): provider(cfg)
    cfg.evaluation.classification_splits = ['A']
    cfg.dataset.inventory = str(Path(cfg.dataset.inventory).parent/'run.json')
    with pytest.raises(ValueError,match='changed after audit'): provider(cfg)


def test_split_a_checks_the_verified_source_provenance(split_a_nffa):
    cfg, reference, _ = split_a_nffa
    proof = json.loads(reference.read_text()); proof['contract']['records'][0]['group_id'] = 'wrong'
    reference.write_text(json.dumps(proof))
    with pytest.raises(ValueError,match='training groups'):
        nffa_training_inventory(reference, Path(cfg.dataset.manifest), Path(cfg.dataset.primary_split), cfg.dataset.root)


def test_new_small_ssl_default_is_a_only():
    from scripts.benchmark_sem_ssl_a100 import parser
    from src.ssl.experiment import configuration as small_configuration
    path = parser().parse_args([]).config
    assert path.name == 'nffa_default.yaml'
    cfg = small_configuration(path)
    assert Path(cfg.dataset.primary_split).name == 'split_A_exact.csv'
    assert not cfg.dataset.holdout_splits
    assert cfg.dataset.inventory.endswith('nffa_split_a_v1.json')


def test_split_a_evaluation_commands_and_reports_do_not_request_b(tmp_path, monkeypatch):
    from scripts.evaluate_sem_cellstyle import commands
    from src.sem_cellstyle.recipe import resolve
    from src.ssl.reporting import result_bundle
    from src.sem_cellstyle.reporting import collect
    cfg = configuration(); cfg.evaluation.classification_splits = ['A']
    resolved = resolve(cfg, 12801, 32)
    run = tmp_path/'run'; run.mkdir()
    (run/'run.json').write_text(json.dumps(dict(config=OmegaConf.to_container(resolved))))
    monkeypatch.setattr('src.ssl.checkpoints.validate_snapshot', lambda path: {})
    for _, task, command in commands(cfg, run, tmp_path/'eval', [0,1], full=True):
        if task in ('classification','report'):
            index = command.index('--splits')
            assert command[index+1:] == (['A','--stage','all'] if task=='classification' else ['A'])
    metrics, _, missing, _ = result_bundle(tmp_path/'eval', '0'*64, 'classification', classification_splits=['A'])
    assert set(metrics) == {'nffa_A_logreg','nffa_A_mlp'}
    assert all('split_A' in path for path in missing)
    result = collect(cfg, run, tmp_path/'eval')
    assert result['classification_splits'] == ['A']
    assert 'nffa_B_logreg' in result['rows'][0]['deliberately_omitted']
