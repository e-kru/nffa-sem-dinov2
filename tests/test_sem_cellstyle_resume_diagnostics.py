"""Zero-tolerance resume reports and error serialization on tiny CPU Meta models."""
import json
from unittest import mock

from omegaconf import OmegaConf
import pytest
import torch

from test_sem_cellstyle import fixture
from src.sem_cellstyle.recipe import schedules
from src.sem_cellstyle.runner import initialize, loader
from src.sem_cellstyle.training import accumulated_step
from src.sem_cellstyle import resume_check
from src.ssl.checkpoints import save_snapshot


def warmed(fixture, tmp_path, sp=False, feature=False):
    cfg, dataset, collate, identity, initialization = fixture
    cfg.regularization = dict(l2_sp=sp, feature_distillation=feature,
                              lambda_sp=2., lambda_feat=.3, lambda_cls=1., lambda_patch=1.)
    model, optimizer = initialize(cfg, initialization, 'cpu')
    sched = schedules(cfg)
    stream = iter(loader(cfg, dataset, collate, 'cpu', 0, 2))
    for update in range(2):
        accumulated_step(model, optimizer, stream, sched, update, precision='fp32')
    root = tmp_path/'state'; root.mkdir()
    path = save_snapshot(root, model, optimizer, 2, OmegaConf.to_container(cfg), identity,
                         sched, initialization, milestone=False)
    payload = torch.load(path/'training.pth', map_location='cpu', weights_only=False)
    return model, optimizer, dataset, collate, sched, payload


@pytest.mark.parametrize('sp,feature', [(False, False), (True, False), (False, True), (True, True)])
def test_report_checks_uninterrupted_state_rng_batches_and_all_meta_states(fixture, tmp_path, sp, feature):
    model, optimizer, dataset, collate, sched, payload = warmed(fixture, tmp_path, sp, feature)
    destination = tmp_path/'diagnostics.json'
    report = resume_check.verify_resume_update(model, optimizer, dataset, collate, sched, 2,
                                                payload, destination, precision='fp32')
    assert report['passed'] and all(report['checks'].values())
    assert report['zero_tolerance'] is True
    assert report['model_difference']['max_absolute_error'] == 0
    assert report['optimizer_difference']['max_absolute_error'] == 0
    assert report['first_batch_signatures'] == report['restored_batch_signatures']
    assert json.loads(destination.read_text())['passed']


def test_one_ulp_restore_error_is_rejected_and_kept_in_report(fixture, tmp_path, monkeypatch):
    args = warmed(fixture, tmp_path)
    original = resume_check.restore_probe_state
    def corrupt(model, optimizer, payload):
        original(model, optimizer, payload)
        with torch.no_grad():
            parameter = next(model.student.parameters())
            parameter.copy_(torch.nextafter(parameter, torch.full_like(parameter, float('inf'))))
    monkeypatch.setattr(resume_check, 'restore_probe_state', corrupt)
    path = tmp_path/'failure.json'
    with pytest.raises(resume_check.ResumeCheckError):
        resume_check.verify_resume_update(*args[:5], 2, args[5], path, precision='fp32')
    report = json.loads(path.read_text())
    assert not report['passed'] and report['zero_tolerance']
    assert not report['checks']['restored_checkpoint_state_exact']
    assert report['restored_pre_state']['model']['different_tensor_count'] > 0


def test_trace_detects_different_augmented_views_despite_identical_draw_ids():
    a = resume_check.BatchTrace([dict(draw_ids=[42], collated_global_crops=torch.zeros(2, 3))])
    b = resume_check.BatchTrace([dict(draw_ids=[42], collated_global_crops=torch.ones(2, 3))])
    next(a); next(b)
    assert a.signatures != b.signatures


def test_approved_drift_accepts_only_finite_update_values_after_exact_restore():
    value = torch.tensor([1.0])
    changed = torch.tensor([1.0000001])
    model_key = 'student.backbone.pos_embed'
    moment_key = '0.exp_avg'
    model_may_differ = lambda key: key.startswith(('student.', 'teacher.'))
    moment_may_differ = lambda key: key.endswith(('.exp_avg', '.exp_avg_sq'))
    saved_model = {model_key: value, 'dino_loss.center': value}
    saved_optimizer = {moment_key: value, '0.step': value}
    model = resume_check.differences(saved_model, saved_model | {model_key: changed},
                                     may_differ=model_may_differ)
    optimizer = resume_check.differences(saved_optimizer, saved_optimizer | {moment_key: changed},
                                         may_differ=moment_may_differ)
    hard_checks = dict(first_checkpoint_state_exact=True, restored_checkpoint_state_exact=True,
                       batch_inputs_equal=True, rng_before_equal=True, rng_after_equal=True,
                       module_modes_equal=True, total_loss_exact=True, optimizer_groups_exact=True)

    def verdict(model, optimizer, *, allow=True, **overrides):
        checks = hard_checks | dict(model_exact=model['exact'], optimizer_exact=optimizer['exact']) | overrides
        return resume_check.resume_verdict(checks, model, optimizer, allow_cuda_update_drift=allow)

    assert verdict(model, optimizer) == dict(passed=True, update_bitwise_exact=False,
                                             cuda_update_drift_accepted=True)
    assert not verdict(model, optimizer, allow=False)['passed']
    for check in hard_checks:
        assert not verdict(model, optimizer, **{check: False})['passed']
    for changed_model in [{model_key: torch.full_like(value, float('inf'))},
                          {model_key: torch.zeros(2)}, {'dino_loss.center': changed}]:
        bad = resume_check.differences(saved_model, saved_model | changed_model, may_differ=model_may_differ)
        assert not verdict(bad, optimizer)['passed']
    bad = resume_check.differences(saved_optimizer, saved_optimizer | {'0.step': changed}, may_differ=moment_may_differ)
    assert not verdict(model, bad)['passed']


def test_benchmark_keeps_finite_metrics_when_resume_fails(fixture, tmp_path, monkeypatch):
    import scripts.benchmark_sem_cellstyle_a100 as cli
    import src.sem_cellstyle.recipe as recipe
    import src.sem_cellstyle.data as data
    import src.sem_cellstyle.benchmark as benchmark
    cfg, dataset, collate, identity, initialization = fixture
    monkeypatch.setattr(recipe, 'configuration', lambda *a: cfg)
    monkeypatch.setattr(recipe, 'require_initialization', lambda *a: initialization)
    monkeypatch.setattr(data, 'provider', lambda *a: (dataset, collate, identity))
    error = resume_check.ResumeCheckError('fixture resume failure', {})
    error.result = dict(status='resume_error', physical_batch=16, finite=True,
                        resume_verified=False, original_images_per_second=5., peak_reserved_bytes=10,
                        error=str(error))
    def fail(*a, **kw):
        raise error
    monkeypatch.setattr(benchmark, 'measure', fail)
    cli.main(['--output', str(tmp_path/'benchmark'), '--candidate', '16'])
    result = json.loads((tmp_path/'benchmark/candidate_16.json').read_text())
    assert result['status'] == 'resume_error' and result['finite'] is True
    assert result['original_images_per_second'] == 5. and not result['resume_verified']


@pytest.mark.parametrize('enabled', [False, True])
def test_diagnostic_numerics_restore_flags_even_after_operator_failure(enabled):
    from scripts.diagnose_sem_cellstyle_resume import deterministic_probe
    previous = (torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled(),
                torch.backends.cudnn.deterministic)
    with pytest.raises(RuntimeError, match='diagnostic operator failure'):
        with deterministic_probe(enabled):
            assert torch.are_deterministic_algorithms_enabled() == (True if enabled else previous[0])
            if enabled:
                assert not torch.is_deterministic_algorithms_warn_only_enabled()
                assert torch.backends.cudnn.deterministic
            raise RuntimeError('diagnostic operator failure')
    assert previous == (torch.are_deterministic_algorithms_enabled(),
                        torch.is_deterministic_algorithms_warn_only_enabled(),
                        torch.backends.cudnn.deterministic)


def test_original_cpu_update_stays_exact_in_strict_diagnostic_mode(fixture, tmp_path):
    from scripts.diagnose_sem_cellstyle_resume import deterministic_probe
    model, optimizer, dataset, collate, sched, payload = warmed(fixture, tmp_path, True, True)
    with deterministic_probe(True):
        report = resume_check.verify_resume_update(model, optimizer, dataset, collate, sched, 2,
                                                   payload, tmp_path/'strict.json', precision='fp32')
    assert report['passed'] and report['deterministic_algorithms'] and report['zero_tolerance']


def test_diagnostic_cli_persists_operator_error_instead_of_losing_evidence(fixture, tmp_path, monkeypatch):
    import scripts.diagnose_sem_cellstyle_resume as cli
    import src.sem_cellstyle.recipe as recipe
    import src.sem_cellstyle.data as data
    import src.sem_cellstyle.runner as runner
    import src.sem_cellstyle.training as training
    cfg, dataset, collate, identity, initialization = fixture
    monkeypatch.setattr(recipe, 'configuration', lambda *a: cfg)
    monkeypatch.setattr(recipe, 'require_initialization', lambda *a: initialization)
    monkeypatch.setattr(recipe, 'resolve', lambda cfg, *a: cfg)
    monkeypatch.setattr(recipe, 'contract', lambda *a: {'fixture': True})
    monkeypatch.setattr(data, 'provider', lambda *a: (dataset, collate, identity))
    monkeypatch.setattr(runner, 'require_a100', lambda: None)
    monkeypatch.setattr(runner, 'initialize', lambda *a: (object(), object()))
    monkeypatch.setattr(runner, 'loader', lambda *a: [])
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '2')
    def unsupported(*args, **kwargs):
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        raise RuntimeError('test_backward_cuda does not have a deterministic implementation')
    monkeypatch.setattr(training, 'accumulated_step', unsupported)
    destination = tmp_path/'diagnostic'
    result = cli.main(['--config', 'fixture.yaml', '--output', str(destination), '--deterministic'])
    assert result == 2
    stored = json.loads((destination/'result.json').read_text())
    details = json.loads((destination/'runtime_error.json').read_text())
    assert stored['status'] == 'runtime_error' and stored['phase'] == 'warmup'
    assert 'test_backward_cuda' in stored['error']
    assert 'RuntimeError' in details['traceback']
    assert stored['diagnostic_deterministic_override'] and stored['cuda_visible_devices'] == '2'
    assert not stored['scientific_training_started'] and not stored['weights_promoted']
