"""Bounded worker sweep selection, command generation and real-loader CPU checks."""
import csv
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import benchmark_sem_eval_workers as bench


def test_recommendation_uses_measured_task_rates_and_prefers_fewer_workers():
    rows = [dict(task=task, workers=w, status='ok', images_per_second=rate)
            for task in ('classification', 'segmentation')
            for w, rate in [(8, 98), (16, 100), (24, 90)]]
    rows.append(dict(task='classification', workers=32, status='error'))
    assert bench.recommend(rows) == dict(classification=8, segmentation=8)
    rows = [r for r in rows if r['task'] != 'segmentation']
    with pytest.raises(ValueError, match='segmentation'):
        bench.recommend(rows)


def test_generated_evaluation_covers_all_six_checkpoints_tasks_and_final_report():
    commands = bench.evaluation_commands('/env/python', Path('/run'), Path('/eval'), Path('/report'),
                                          dict(classification=8, segmentation=24))
    assert len(commands) == 3
    for command in commands:
        assert command[0] == '/env/python'
        start = command.index('--steps')+1
        assert command[start:start+6] == ['0', '500', '1000', '2000', '3000', '5000']
    assert commands[0][commands[0].index('--workers')+1] == '8'
    assert commands[1][commands[1].index('--workers')+1] == '24'
    assert commands[0][commands[0].index('--task')+1] == 'classification'
    assert commands[1][commands[1].index('--task')+1] == 'segmentation'
    assert commands[-1][1].endswith('report_sem_ssl.py')


def test_resource_report_contains_affinity_and_memory():
    report = bench.resources()
    assert 0 < report['effective_cpus'] <= len(report['affinity']) <= report['logical_cpus']
    assert 0 < report['physical_cores'] <= report['logical_cpus']
    assert 0 <= report['MemAvailable'] <= report['MemTotal']


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_original_datasets_cpu_extraction_and_no_dataset_hashing(tmp_path, monkeypatch):
    import numpy as np
    import torch
    from PIL import Image
    from src import config
    from src.aachen_ablation import data as aachen_data
    from src.downstream import benchmark as downstream
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(bench, 'ROOT', tmp_path)
    monkeypatch.setattr(config, 'DATASET_ROOT', tmp_path)
    def forbidden(*_args, **_kwargs):
        raise AssertionError('The worker benchmark must not hash datasets')
    monkeypatch.setattr(aachen_data, 'sha256_file', forbidden)
    monkeypatch.setattr(downstream, 'sha256_file', forbidden)
    classification, segmentation = [], []
    for i, split in enumerate(('train', 'train', 'val', 'test')):
        image = tmp_path/f'{i}.png'
        mask = tmp_path/f'{i}_mask.png'
        if split != 'test':
            Image.fromarray(np.full((48, 64, 3), i*30, np.uint8)).save(image)
            Image.fromarray(np.zeros((48, 64), np.uint8)).save(mask)
        classification.append(dict(image_id=image.name, group_id=str(i), split=split,
                                   label=str(i % 2) if split != 'test' else 'FORBIDDEN'))
        segmentation.append(dict(image_id=image.name, image_path=str(image), mask_path=str(mask),
                                  group_id=str(i), split=split))
    write_csv(tmp_path/'outputs/dataset_v3/splits/split_A_exact.csv', classification)
    write_csv(tmp_path/'outputs/aachen_heerlen/splits/group_split_seed42.csv', segmentation)
    datasets = bench.datasets(SimpleNamespace(nffa_images=3, aachen_images=3))
    assert all(len(dataset) == 3 for dataset in datasets.values())
    class Encoder:
        def forward_global(self, images):
            return torch.zeros(len(images), 384)
        def forward_dense(self, images):
            return torch.zeros(len(images), 384, 48, 64)
    for task, dataset in datasets.items():
        result = bench.measure(task, dataset, Encoder(), workers=0, device=torch.device('cpu'))
        assert result['images'] == 3
        assert result['wall_seconds'] > 0 and result['images_per_second'] > 0
        assert result['cpu_seconds'] >= 0 and result['first_batch_seconds'] > 0
    assert not list(tmp_path.rglob('cache_metadata.json'))


def test_existing_checkpoint_runner_forwards_workers_to_both_tasks(tmp_path):
    from unittest.mock import patch
    from scripts.evaluate_sem_ssl import commands
    from src.ssl.upstream import PIN
    meta = dict(files={'teacher_backbone.pth': PIN['backbone_sha256']})
    with patch('scripts.evaluate_sem_ssl.validate_snapshot', return_value=meta), \
            patch('scripts.evaluate_sem_ssl.sha256', return_value=PIN['backbone_sha256']):
        for task, heads in [('classification', ['logreg', 'mlp']), ('segmentation', ['linear', 'cnn8'])]:
            rows = list(commands(tmp_path, bench.STEPS, task, 'all', tmp_path/'eval', workers=24))
            assert len(rows) == 6
            for _, _, command, _ in rows:
                assert command[command.index('--workers')+1] == '24'
                assert command[command.index('--heads')+1:command.index('--heads')+3] == heads
