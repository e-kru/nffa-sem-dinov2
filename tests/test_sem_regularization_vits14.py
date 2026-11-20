from pathlib import Path

import pytest
import torch

from scripts import run_sem_regularization_vits14 as pilot
from src.sem_cellstyle.recipe import contract
from src.sem_cellstyle.regularization_study import calibrate
from src.sem_cellstyle.runner import execute
from src.ssl.adapter import load_config
from src.ssl.data import sha256

from test_sem_cellstyle import fixture, tiny


def test_small_pilot_config_and_evaluation_commands():
    cfg = load_config(pilot.CONFIG)
    assert cfg.student.arch == 'vit_small'
    assert cfg.crops.global_crops_size == 224 and cfg.crops.local_crops_size == 98
    assert cfg.runtime.physical_batch * cfg.runtime.accumulation_steps == cfg.runtime.effective_batch == 64
    assert cfg.experiment.schedule_total_steps == cfg.experiment.run_until_step == 500
    assert list(cfg.evaluation.classification_splits) == ['A']
    for task, heads, worker_count in [
        ('classification', ['logreg', 'mlp'], '8'),
        ('segmentation', ['linear', 'cnn8'], '16'),
    ]:
        command = pilot.evaluation_command(cfg, Path('weights.pth'), Path('evaluation'), task)
        assert 'scripts.evaluate_sem_encoder' in command
        assert command[command.index('--heads') + 1:command.index('--splits')] == heads
        assert command[command.index('--splits') + 1] == 'A'
        assert command[command.index('--workers') + 1] == worker_count
        assert '--verified-data-root' in command


def test_four_small_arms_restart_and_resume_from_common_calibration(fixture, tmp_path, monkeypatch):
    import dinov2.hub.backbones

    cfg, dataset, collate, identity, initialization = fixture
    cfg.student.arch = 'vit_small'
    cfg.initialization.sha256 = sha256(initialization)
    cfg.experiment.milestones = [0, 4]
    monkeypatch.setattr(dinov2.hub.backbones, 'dinov2_vits14', tiny)
    monkeypatch.setattr(pilot, 'DRIFT_UPDATES', 2)
    monkeypatch.setattr(pilot, 'PROBES', 2)
    monkeypatch.setattr(pilot, 'contract', lambda cfg, data, init, _: contract(cfg, data, init, 'cpu'))
    monkeypatch.setattr(pilot, 'calibrate', lambda *args, **kwargs:
                        calibrate(*args, **kwargs, device='cpu', target=.01, minimum=1e-8, maximum=1.))
    starts = []

    def run(cfg, dataset, collate, identity, initialization, destination, *, resume):
        starts.append((destination.name, resume))
        stop = 2 if destination.name == 'B' and not resume else None
        return execute(cfg, dataset, collate, identity, initialization, destination,
                       resume=resume, device='cpu', interrupt_after=stop)

    monkeypatch.setattr(pilot, 'execute', run)
    root = tmp_path / 'pilot'
    root.mkdir()
    with pytest.raises(InterruptedError, match='Arm B'):
        pilot.train_arms(root, cfg, dataset, collate, identity, initialization)
    assert not (root / 'runs/C').exists()
    paths, coeffs = pilot.train_arms(root, cfg, dataset, collate, identity, initialization)
    assert starts == [('A', False), ('B', False), ('B', True), ('C', False), ('D', False)]
    assert set(paths) == {'A', 'B', 'C', 'D'}
    assert coeffs['lambda_sp'] > 0 and coeffs['lambda_feat'] > 0
    for path in paths.values():
        assert sha256(path.parent / 'step_000000' / 'teacher_backbone.pth') == cfg.initialization.sha256
    original = torch.load(initialization, map_location='cpu', weights_only=True)
    d = torch.load(paths['D'] / 'training.pth', map_location='cpu', weights_only=False)
    assert all(torch.equal(d['model'][f'regularization.reference.{key}'], original[key]) for key in original)
