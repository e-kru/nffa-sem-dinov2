"""CPU-only benchmark orchestration and reduced original-Meta integration."""
import copy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader

pytest.importorskip('omegaconf')
from src import ssl_benchmark as bench
from src.ssl.adapter import SingleGPUAdapter, train_step
from src.ssl.data import SEMTrainDataset, augmentation_and_collate, sha256
from src.ssl.experiment import configuration, digest, step_schedules
from src.ssl.stream import CursorBatches, DrawCollate, DrawDataset
from src.ssl.upstream import ROOT, activate, checkout_path
from scripts.benchmark_sem_ssl_a100 import parser, validate_options


class CpuTimer:
    """Test backend only; production CLI always constructs CudaWindowTimer."""
    def begin(self):
        self.count = 0

    def before_step(self):
        pass

    def after_step(self):
        self.count += 1

    def finish(self):
        return dict(wall_seconds=self.count * .1, step_seconds=[.1] * self.count,
                    train_step_seconds=[.08] * self.count)


def good_result(batch=8, throughput=100, reserved=30, tail=1.1):
    return dict(batch_size=batch, status='ok', measured_steps=20, measured_originals=20 * batch,
                originals_per_second=throughput, window_wall_seconds=20 * batch / throughput,
                finite_training=True, step_seconds=dict(mean=.1, median=.1, p95=.1 * tail),
                train_step_seconds=dict(mean=.08, median=.08, p95=.09),
                dataloader_wait_seconds=dict(mean=.02, median=.01, p95=.03),
                peak_allocated_bytes=(reserved-1) * 1024**3, peak_reserved_bytes=reserved * 1024**3,
                p95_to_median=tail, gpu=dict(total_bytes=80 * 1024**3, free_before_model_bytes=78 * 1024**3, uuid='same'),
                initial_state_sha256='same')


def test_statistics_and_original_image_throughput():
    assert bench.aggregate([1, 2, 3, 4]) == dict(mean=2.5, median=2.5, p95=3.8499999999999996)
    rows = [dict(iteration=i, total_loss=torch.tensor(2.), gradient_norm=torch.tensor(1.)) for i in range(3)]
    result = bench.summarize_window(dict(rows=rows, timing=dict(wall_seconds=3., step_seconds=[.5, 1., 1.5],
        train_step_seconds=[.4, .9, 1.4], dataloader_wait_seconds=[.1, .1, .1])), 8)
    assert result['originals_per_second'] == 8
    assert result['measured_originals'] == 24  # Never multiply by 10 multicrop views.
    assert result['step_seconds']['median'] == 1
    for bad in ([], [float('nan')], [-1]):
        with pytest.raises(ValueError):
            bench.aggregate(bad)


def test_warmup_and_measurement_use_contiguous_optimizer_indices():
    called = []
    def step(batch, index):
        called.append((batch, index))
        return dict(iteration=index, total_loss=torch.tensor(1.), gradient_norm=torch.tensor(.2))
    iterator = iter(range(25))
    warm = bench.training_window(iterator, step, 5, 0, CpuTimer())
    measured = bench.training_window(iterator, step, 20, 5, CpuTimer())
    assert called == list(zip(range(25), range(25)))
    assert len(warm['rows']) == 5 and len(measured['rows']) == 20
    assert bench.summarize_window(measured, 8)['measured_originals'] == 160


def test_cuda_events_read_only_after_one_boundary_sync(monkeypatch):
    state = dict(sync=0, event=0)
    class Event:
        def __init__(self, **kwargs):
            self.number = state['event']; state['event'] += 1
        def record(self):
            assert state['sync'] == 0
        def elapsed_time(self, _other):
            assert state['sync'] == 1
            return 10.
    monkeypatch.setattr(torch.cuda, 'Event', Event)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda:state.update(sync=state['sync'] + 1))
    timer = bench.CudaWindowTimer()
    timer.begin()
    for _ in range(20):
        timer.before_step(); timer.after_step()
    assert state['sync'] == 0
    values = timer.finish()
    assert state['sync'] == 1 and len(values['step_seconds']) == 20
    assert len(values['train_step_seconds']) == 20


@pytest.mark.parametrize('key,value', [('total_loss', float('nan')), ('gradient_norm', float('inf'))])
def test_nonfinite_loss_and_gradient_detected_at_boundary(key, value):
    with pytest.raises(FloatingPointError, match=key):
        bench.materialize([dict(iteration=3, **{key:torch.tensor(value)})])


@pytest.mark.parametrize('error,status', [(torch.cuda.OutOfMemoryError('test OOM'), 'oom'),
    (RuntimeError('CUDA out of memory. test'), 'oom'), (FloatingPointError('nonfinite gradient'), 'nonfinite'),
    (RuntimeError('other failure'), 'error')])
def test_failure_classification_and_serialization(tmp_path, error, status):
    def action(progress):
        progress['phase'] = 'warmup'
        raise error
    result = bench.run_case(action, 16, tmp_path / 'result.json')
    assert result['status'] == status and result['phase'] == 'warmup'
    assert json.loads((tmp_path / 'result.json').read_text())['status'] == status
    assert 'originals_per_second' not in result


def test_memory_and_stability_aware_recommendation():
    results = [good_result(8, 100), good_result(16, 102), good_result(32, 150, reserved=75),
               good_result(64, 200, tail=3)]
    chosen = bench.recommendation(results)
    assert chosen['batch_size'] == 8  # Within 3% of B16; larger cases violate criteria.
    assert '32' in chosen['excluded'] and '64' in chosen['excluded']
    assert bench.recommendation([dict(batch_size=64, status='oom')])['batch_size'] is None
    constrained = good_result()
    constrained['gpu']['free_before_model_bytes'] = 31 * 1024**3
    assert bench.recommendation([constrained])['batch_size'] is None


def test_optional_128_requires_headroom_and_improving_throughput():
    assert bench.may_try_128([good_result(32, 100), good_result(64, 110, reserved=40)])
    assert not bench.may_try_128([good_result(32, 100), good_result(64, 101, reserved=40)])
    assert not bench.may_try_128([good_result(32, 100), good_result(64, 110, reserved=70)])
    assert not bench.may_try_128([dict(batch_size=64, status='oom')])


def test_isolated_cpu_subprocesses_continue_after_oom(tmp_path):
    pids = []
    def runner(command, log):
        batch = int(command[command.index('--worker-batch') + 1])
        row = good_result(batch, throughput=batch*10)
        if batch == 16:
            row = dict(batch_size=batch, status='oom')
        # Exercise real subprocess isolation, with CPU-only JSON work.
        code = 'import json,os,sys; d=json.loads(sys.argv[1]); d["pid"]=os.getpid(); open(sys.argv[2],"x").write(json.dumps(d))'
        result = bench.subprocess_case([sys.executable, '-c', code, json.dumps(row), str(log.parent / 'result.json')], log)
        pids.append(json.loads((log.parent / 'result.json').read_text())['pid'])
        return result
    results, optional = bench.orchestrate(dict(batch_sizes=[8, 16, 32], try_128=False), tmp_path, runner=runner)
    assert [row['status'] for row in results] == ['ok', 'oom', 'ok']
    assert len(set(pids)) == 3 and os.getpid() not in pids
    assert len(json.loads((tmp_path / 'partial_results.json').read_text())['results']) == 3
    assert not optional['executed']


def test_missing_worker_result_and_crash_are_not_success(tmp_path):
    rows, _ = bench.orchestrate(dict(batch_sizes=[8], try_128=False), tmp_path,
                                runner=lambda command,log:137)
    assert rows[0]['status'] == 'error' and rows[0]['worker_returncode'] == 137
    assert bench.recommendation(rows)['batch_size'] is None


def test_subprocess_interruption_terminates_its_process_group(tmp_path, monkeypatch):
    process = SimpleNamespace(pid=1234, wait=mock.Mock(side_effect=[KeyboardInterrupt(), 0]))
    monkeypatch.setattr(bench.subprocess, 'Popen', lambda *args, **kwargs:process)
    terminate = mock.Mock()
    monkeypatch.setattr(bench.os, 'killpg', terminate)
    with pytest.raises(KeyboardInterrupt):
        bench.subprocess_case(['CPU-test'], tmp_path / 'worker.log')
    terminate.assert_called_once_with(1234, bench.signal.SIGTERM)


def test_orchestration_runs_optional_128_only_after_qualifying_64(tmp_path):
    seen = []
    def runner(command, log):
        batch = int(command[command.index('--worker-batch')+1]); seen.append(batch)
        row = good_result(batch, throughput=100 if batch == 32 else 110 if batch == 64 else 115)
        (log.parent / 'result.json').write_text(json.dumps(row))
        return 0
    rows, optional = bench.orchestrate(dict(batch_sizes=[32,64],try_128=True),tmp_path,runner=runner)
    assert seen == [32,64,128] and optional['executed']
    assert rows[-1]['batch_size'] == 128


def test_csv_json_and_plot_outputs_preserve_failed_nulls(tmp_path):
    results = [good_result(), dict(batch_size=16, status='oom', error='test allocation failure')]
    report = bench.write_report(tmp_path, dict(max_memory_fraction=.85), results, dict(requested=False, executed=False))
    assert report['recommendation']['batch_size'] == 8
    assert json.loads((tmp_path / 'benchmark.json').read_text())['benchmark_only']
    with (tmp_path / 'benchmark.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows[1]['originals_per_second'] == '' and rows[0]['step_median_seconds'] == '0.1'
    assert (tmp_path / 'throughput_vs_batch.png').stat().st_size > 0
    assert (tmp_path / 'memory_vs_batch.png').stat().st_size > 0
    assert not list(tmp_path.glob('*.pth'))


def test_cli_requires_new_separate_output_and_one_visible_gpu(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1')
    args = parser().parse_args(['--output', str(tmp_path / 'new')])
    with pytest.raises(ValueError, match='CUDA_VISIBLE_DEVICES'):
        validate_options(args)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    args = parser().parse_args(['--output', str(tmp_path)])
    with pytest.raises(FileExistsError):
        validate_options(args)
    run = tmp_path / 'training'; (run / 'checkpoints').mkdir(parents=True); (run / 'run.json').write_text('{}')
    args = parser().parse_args(['--output', str(run / 'benchmark')])
    with pytest.raises(ValueError, match='outside'):
        validate_options(args)
    args = parser().parse_args(['--output', str(tmp_path / 'new'), '--batch-sizes', '8', '64', '128'])
    validate_options(args)
    assert args.batch_sizes == [8, 64] and args.try_128


@pytest.mark.parametrize('name,total,uuid', [('A100 80GB', 80, 'MIG-uuid'), ('RTX', 80, 'gpu'), ('A100', 40, 'gpu')])
def test_gpu_guard_rejects_mig_wrong_gpu_and_memory(monkeypatch, name, total, uuid):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda:True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda:1)
    monkeypatch.setattr(torch.cuda, 'set_device', lambda _:None)
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda _:SimpleNamespace(name=name,total_memory=total*1024**3,uuid=uuid))
    with pytest.raises(RuntimeError, match='A100 80GB'):
        bench.validate_gpu()


@pytest.fixture
def meta_fixture(tmp_path, monkeypatch):
    if not checkout_path().is_dir():
        pytest.skip('Pinned Meta checkout not installed')
    activate()
    import dinov2.hub.backbones
    from dinov2.models.vision_transformer import DinoVisionTransformer
    def backbone(**kwargs):
        return DinoVisionTransformer(img_size=56, patch_size=14, embed_dim=24, depth=1, num_heads=3, block_chunks=0)
    monkeypatch.setattr(dinov2.hub.backbones, 'dinov2_vits14', backbone)
    torch.set_num_threads(1)
    cfg = configuration(ROOT / 'configs/sem_ssl/nffa_200step.yaml')
    cfg.train.batch_size_per_gpu, cfg.train.num_workers = 2, 0
    cfg.crops.global_crops_size, cfg.crops.local_crops_size, cfg.crops.local_crops_number = 56, 28, 2
    cfg.ibot.mask_ratio_min_max = [.4, .5]
    cfg.dino.head_n_prototypes, cfg.dino.head_hidden_dim, cfg.dino.head_bottleneck_dim = 32, 64, 16
    records = []
    for index in range(4):
        path = tmp_path / f'{index}.png'
        Image.fromarray(np.random.default_rng(index).integers(0, 256, (90, 100, 3), dtype=np.uint8)).save(path)
        records.append(dict(image_id=path.name,group_id=str(index)))
    transform, collate = augmentation_and_collate(cfg)
    dataset = SEMTrainDataset(tmp_path, records, transform)
    loader = DataLoader(DrawDataset(dataset, 42), batch_sampler=CursorBatches(4, 2, 42, 0, 3),
                        collate_fn=DrawCollate(collate, 42), num_workers=0, generator=torch.Generator().manual_seed(42))
    return cfg, loader


def test_deferred_full_ssl_step_matches_normal_and_has_no_scalar_readback(meta_fixture):
    cfg, loader = meta_fixture
    torch.manual_seed(42)
    normal = SingleGPUAdapter(cfg, None, allow_random_init=True)
    torch.manual_seed(42)
    deferred = SingleGPUAdapter(cfg, None, allow_random_init=True)
    assert bench.state_fingerprint(normal) == bench.state_fingerprint(deferred)
    optimizer_a, optimizer_b = [torch.optim.AdamW(model.parameter_groups()) for model in (normal, deferred)]
    schedules = step_schedules(cfg)
    batch = next(iter(loader))
    with pytest.raises(ValueError, match='diagnostics=True'):
        train_step(deferred, optimizer_b, batch, schedules, 10, defer_metrics=True)
    regular = train_step(normal, optimizer_a, batch, schedules, 10, diagnostics=True)
    with mock.patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('Per-step CPU readback')), \
            mock.patch.object(torch.Tensor, '__float__', side_effect=AssertionError('Per-step scalar readback')):
        row = train_step(deferred, optimizer_b, batch, schedules, 10, diagnostics=True, defer_metrics=True)
    actual = bench.materialize([row])[0]
    assert actual == regular
    for key, value in normal.state_dict().items():
        assert torch.equal(value, deferred.state_dict()[key]), key
    assert all(parameter.grad is None for parameter in deferred.teacher.parameters())


def test_cpu_synthetic_complete_training_windows(meta_fixture):
    cfg, loader = meta_fixture
    model = SingleGPUAdapter(cfg, None, allow_random_init=True)
    optimizer = torch.optim.AdamW(model.parameter_groups())
    schedules = step_schedules(cfg)
    def step(batch, iteration):
        return train_step(model, optimizer, batch, schedules, iteration, diagnostics=True, defer_metrics=True)
    iterator = iter(loader)
    bench.training_window(iterator, step, 1, 0, CpuTimer())
    measured = bench.training_window(iterator, step, 2, 1, CpuTimer())
    summary = bench.summarize_window(measured, 2)
    assert summary['measured_steps'] == 2 and summary['measured_originals'] == 4
    assert optimizer.state and all(row['gradient_norm'] > 0 for row in measured['rows'])
    assert not model.teacher.training
