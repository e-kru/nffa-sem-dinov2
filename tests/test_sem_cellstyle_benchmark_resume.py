"""Regression for the real benchmark's repeated in-memory checkpoint restore."""
import copy

import pytest
import torch

from test_sem_cellstyle import fixture
from src.sem_cellstyle.benchmark import _restore_probe_state, recommendation
from src.sem_cellstyle.recipe import schedules
from src.sem_cellstyle.runner import initialize, loader
from src.sem_cellstyle.training import accumulated_step
from src.ssl.checkpoints import save_snapshot


@pytest.mark.parametrize('sp,feature', [(False, False), (True, False), (False, True), (True, True)])
def test_benchmark_restore_keeps_checkpoint_immutable_and_matches_uninterrupted_update(
        fixture, tmp_path, sp, feature):
    cfg, dataset, collate, identity, initialization = fixture
    cfg.regularization = dict(l2_sp=sp, feature_distillation=feature,
                              lambda_sp=2., lambda_feat=.3, lambda_cls=1., lambda_patch=1.)
    model, optimizer = initialize(cfg, initialization, 'cpu')
    sched = schedules(cfg)
    stream = iter(loader(cfg, dataset, collate, 'cpu', 0, 2))
    for update in range(2):
        accumulated_step(model, optimizer, stream, sched, update, precision='fp32')
    run = tmp_path / 'benchmark_checkpoint'
    run.mkdir()
    from omegaconf import OmegaConf
    path = save_snapshot(run, model, optimizer, 2, OmegaConf.to_container(cfg),
                         identity, sched, initialization, milestone=False)
    payload = torch.load(path / 'training.pth', map_location='cpu', weights_only=False)
    saved_optimizer = copy.deepcopy(payload['optimizer'])

    def step():
        batches = iter(loader(cfg, dataset, collate, 'cpu', 2, 3))
        return accumulated_step(model, optimizer, batches, sched, 2, precision='fp32')

    # Include a genuinely uninterrupted update, not just two restored updates.
    uninterrupted = step()
    expected_model = {name: value.clone() for name, value in model.state_dict().items()}
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    for _ in range(2):
        _restore_probe_state(model, optimizer, payload)
        # CPU checkpoint tensors must remain independent even for AdamW.step.
        for key, state in optimizer.state_dict()['state'].items():
            for name, value in state.items():
                assert value.data_ptr() != payload['optimizer']['state'][key][name].data_ptr()
        restored = step()
        assert restored['draw_ids'] == uninterrupted['draw_ids']
        assert restored['total_loss'] == uninterrupted['total_loss']
        for name, value in expected_model.items():
            assert torch.equal(value, model.state_dict()[name]), name
        for key, state in expected_optimizer['state'].items():
            for name, value in state.items():
                assert torch.equal(value, optimizer.state_dict()['state'][key][name]), (key, name)
        for key, state in saved_optimizer['state'].items():
            for name, value in state.items():
                assert torch.equal(value, payload['optimizer']['state'][key][name]), (key, name)


def test_failed_resume_does_not_claim_oom_or_require_a_new_training_recipe():
    result = recommendation([dict(status='error', physical_batch=16, finite=False,
                                  error='Restored update differs')])
    assert result['physical_batch'] is None
    assert 'resume' in result['reason']
    assert 'inspect candidate errors' in result['reason']
    assert 'revise recipe' not in result['reason']
