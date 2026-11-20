"""Numerical and protocol regressions for batched original-raster training."""
import copy
import csv
import hashlib
import json
import subprocess
from unittest.mock import patch

import pytest
import torch

import test_aachen_ablation as fixtures

fixture_manifest = fixtures.fixture_manifest
make_cache = fixtures.make_cache
from scripts.benchmark_aachen_ablation import REFERENCE_REPOSITORY, REFERENCE_REVISION, reference_module
from scripts.run_aachen_ablation import parser
from src.aachen_ablation import cache, engine
from src.aachen_ablation.geometry import VARIANTS
from src.aachen_ablation.loading import make_loader


REFERENCE_GIT = ["git", "-C", str(REFERENCE_REPOSITORY)]
reference_test = pytest.mark.skipif(
    subprocess.run([*REFERENCE_GIT, "cat-file", "-e", f"{REFERENCE_REVISION}^{{commit}}"],
                   capture_output=True).returncode != 0,
    reason="Set SEM_AACHEN_REFERENCE_REPO to a checkout with the audited legacy revision",
)


@pytest.mark.parametrize("name", list(VARIANTS))
@pytest.mark.parametrize("kind", ["linear", "cnn8"])
@pytest.mark.parametrize("ragged", [False, True])
@reference_test
def test_batched_original_loss_gradients_and_metrics_match_reference(name, kind, ragged):
    v = VARIANTS[name]
    torch.manual_seed(231)
    # Same fixed tensors/weights; eval disables stochastic dropout, not gradients.
    original = engine.make_decoder(kind).eval()
    batched = copy.deepcopy(original)
    sizes = [(9, 13), (11, 15) if ragged else (9, 13)]
    samples = []
    for i, size in enumerate(sizes):
        y = torch.randint(0, 2, size, dtype=torch.long)
        y[0, :3] = 255
        y[:size[0]//2, :size[1]//2] = 255  # Entire ignored quadrant is legal.
        samples.append({"features": torch.randn(4 if v.quadrants else 1, 384, *v.grid),
                        "target": y, "record": {"image_id": str(i)}})
    reference = reference_module("engine", REFERENCE_REVISION)
    old_grid = torch.stack([original(s["features"]) for s in samples])
    expected = torch.stack([reference.original_loss(z, s["target"], v) for z, s in zip(old_grid, samples)])
    batch = engine.pack_originals(samples)
    grid = engine.forward_batch(batched, batch, "cpu")
    full = engine.project_batch(grid, batch["targets"], v)
    actual = engine.full_resolution_losses(full, batch["targets"], torch.device("cpu"))
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
    expected.mean().backward()
    actual.mean().backward()
    for a, b in zip(original.parameters(), batched.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=3e-4, atol=3e-6)
    old = reference.evaluate(original, samples, v, torch.device("cpu"), thresholds=(.05, .5, .95))
    new = engine.evaluate(batched, samples, v, "cpu", thresholds=(.05, .5, .95), batch_size=2)
    assert old["thresholds"] == new["thresholds"]
    assert new["loss"] == pytest.approx(old["loss"], rel=2e-6, abs=2e-7)


def test_batched_validation_has_one_forward_per_batch_and_no_per_image_sync():
    v = VARIANTS["A1"]
    samples = [{"features": torch.randn(1, 384, *v.grid), "target": torch.zeros(9, 13, dtype=torch.long),
                "record": {"image_id": str(i)}} for i in range(5)]
    model = engine.make_decoder("linear")
    with patch.object(model, "forward", wraps=model.forward) as forward, \
         patch.object(engine, "sync", side_effect=AssertionError("per-image sync")):
        result = engine.evaluate(model, samples, v, "cpu", batch_size=2)
    assert forward.call_count == 3
    assert result["gpu_computation_seconds"] is None
    assert result["phase_seconds"] >= result["data_loading_seconds"] >= 0


def test_sampler_matches_old_order_and_does_not_touch_main_rng():
    from torch.utils.data import DataLoader
    data = list(range(17))
    before = torch.get_rng_state().clone()
    loader = make_loader(data, 8, 0, torch.device("cpu"), shuffle=True, seed=42)
    for epoch in (1, 2, 9):
        loader.sampler.epoch = epoch
        old = DataLoader(data, batch_size=8, shuffle=True, generator=torch.Generator().manual_seed(42+epoch))
        assert list(loader.sampler) == torch.cat(list(old)).tolist()
    assert torch.equal(before, torch.get_rng_state())
    persistent = make_loader(data, 8, 8, torch.device("cpu"))
    assert persistent.persistent_workers and persistent.num_workers == 8
    assert persistent.worker_init_fn is not None
    assert not loader.persistent_workers
    defaults = parser().parse_args([])
    assert defaults.micro_batch_size == defaults.batch_size == defaults.validation_batch_size == 8
    assert defaults.determinism == "off"


def test_cached_training_reads_masks_but_never_original_rgb(tmp_path, fixture_manifest):
    path, contract = make_cache(tmp_path, fixture_manifest, "A1")
    dataset = cache.OriginalCachedDataset(path, contract, fixture_manifest, "train")
    with patch.object(cache, "load_original", side_effect=AssertionError("original RGB loaded")):
        sample = dataset[0]
        assert sample["target"].shape == (9, 13)
        model = engine.make_decoder("linear")
        engine.train_epoch(model, dataset, torch.optim.AdamW(model.parameters()),
                           VARIANTS["A1"], "cpu", 8, 8, 42, 1)


@pytest.mark.parametrize("optimizer_class", [torch.optim.SGD, torch.optim.AdamW])
def test_microbatch_accumulation_and_oom_retry_preserve_optimizer_steps(optimizer_class):
    v = VARIANTS["A1"]
    torch.manual_seed(6)
    samples = [{"features": torch.randn(1, 384, *v.grid), "target": torch.randint(0, 2, (9, 13)),
                "record": {"image_id": str(i)}} for i in range(9)]
    one, eight = engine.make_decoder("linear"), engine.make_decoder("linear")
    eight.load_state_dict(one.state_dict())
    for model, micro in ((one, 1), (eight, 8)):
        optimizer = optimizer_class(model.parameters(), lr=.01)
        result = engine.train_epoch(model, samples, optimizer, v, "cpu", 8, micro, 42, 1)
        assert result["optimizer_steps"] == 2 and result["originals_seen"] == 9
    for a, b in zip(one.parameters(), eight.parameters()):
        torch.testing.assert_close(a, b, rtol=2e-6, atol=2e-7)
    control = copy.deepcopy(eight)
    engine.train_epoch(control, samples, torch.optim.SGD(control.parameters(), lr=.01), v, "cpu", 8, 4, 42, 1)
    retried = copy.deepcopy(eight)
    optimizer = torch.optim.SGD(retried.parameters(), lr=.01)
    real = engine._train_effective_batch
    calls = []
    def simulate_oom(model, batch, opt, variant, device, micro):
        calls.append(micro)
        if len(calls) == 1:
            # Simulate partial gradients before the failed attempt; retry must discard them.
            for p in model.parameters():
                p.grad = torch.ones_like(p)
            raise torch.cuda.OutOfMemoryError("synthetic OOM")
        return real(model, batch, opt, variant, device, micro)
    with patch.object(engine, "_train_effective_batch", side_effect=simulate_oom), patch("torch.cuda.empty_cache"):
        result = engine.train_epoch(retried, samples, optimizer, v, "cpu", 8, 8, 42, 1)
    assert calls == [8, 4, 4]
    assert result["optimizer_steps"] == 2 and result["oom_retries"] == 1
    assert result["actual_micro_batch_size"] == 4
    for a, b in zip(control.parameters(), retried.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_training_reuses_loaders_and_records_timings(tmp_path, fixture_manifest):
    path, contract = make_cache(tmp_path, fixture_manifest, "A1")
    train = cache.OriginalCachedDataset(path, contract, fixture_manifest, "train")
    val = cache.OriginalCachedDataset(path, contract, fixture_manifest, "val")
    config = dict(cache_metadata={}, head="linear", lr=.001, weight_decay=.0001, seed=42,
                  epochs=2, patience=8, batch_size=8, micro_batch_size=8, workers=0, determinism_policy="off")
    with patch.object(engine, "make_loader", wraps=make_loader) as constructors:
        engine.train_run(tmp_path / "run", config, train, val, VARIANTS["A1"], "cpu")
    assert constructors.call_count == 2  # train and validation, not per epoch
    with (tmp_path / "run/history.csv").open() as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        assert float(row["epoch_seconds"]) >= float(row["train_seconds"]) > 0
        assert float(row["validation_seconds"]) > 0
        assert float(row["data_loading_seconds"]) >= 0
        assert row["gpu_computation_seconds"] == ""


@reference_test
def test_audited_legacy_cache_reuse_and_new_feature_source_invalidation(tmp_path, fixture_manifest):
    path, expected = make_cache(tmp_path, fixture_manifest, "A1")
    legacy = tmp_path / "A1_legacy"
    path.rename(legacy)
    tree = subprocess.check_output([*REFERENCE_GIT, "ls-tree", "-r", "--name-only", REFERENCE_REVISION,
                                    "src/aachen_ablation"], text=True).splitlines()
    files = [p for p in tree if p.endswith(".py")] + ["src/preprocessing_v2.py", "src/dinov2_encoder.py",
                                                    "src/downstream/model.py", "src/downstream/heads.py"]
    sources = {name: hashlib.sha256(subprocess.check_output([*REFERENCE_GIT, "show", f"{REFERENCE_REVISION}:{name}"])).hexdigest()
               for name in files}
    meta_path = legacy / "cache_metadata.json"
    meta = json.loads(meta_path.read_text())
    del meta["contract"]["feature_sources"]
    meta["contract"]["sources"] = sources
    assert cache.digest(sources) in cache.LEGACY_FEATURE_SOURCE_SETS
    meta_path.write_text(json.dumps(meta))
    timestamp = meta_path.stat().st_mtime_ns
    assert cache.find_cache(tmp_path, expected) == legacy
    assert cache.validate_cache(legacy, expected)["complete"]
    assert meta_path.stat().st_mtime_ns == timestamp
    # Changing training-only provenance is allowed; extraction provenance/data is not.
    current = copy.deepcopy(expected)
    current["sources"]["src/aachen_ablation/engine.py"] = "training-only-change"
    assert cache.contracts_compatible(expected, current)
    current["feature_sources"]["files"]["src/aachen_ablation/geometry.py"] = "changed"
    assert not cache.contracts_compatible(expected, current)
    bad = copy.deepcopy(meta["contract"])
    bad["sources"]["src/aachen_ablation/cache.py"] = "unknown-extraction-code"
    assert not cache.contracts_compatible(bad, expected)
    bad = copy.deepcopy(meta["contract"])
    bad["records"][0]["mask_sha256"] = "changed-mask"
    assert not cache.contracts_compatible(bad, expected)



def test_real_persistent_workers_keep_processes_between_epochs():
    from multiprocessing.connection import Listener
    try:
        with Listener(family="AF_UNIX"):
            pass
    except PermissionError:
        pytest.skip("Local sandbox forbids IPC sockets for real worker reuse test")
    v = VARIANTS["A1"]
    samples = [{"features": torch.zeros(1, 384, *v.grid), "target": torch.zeros(9, 13, dtype=torch.long),
                "record": {"image_id": str(i)}} for i in range(3)]
    loader = make_loader(samples, 2, 2, torch.device("cpu"), shuffle=True)
    try:
        list(loader)
        first = [worker.pid for worker in loader._iterator._workers]
        loader.sampler.epoch = 2
        list(loader)
        assert first == [worker.pid for worker in loader._iterator._workers]
    finally:
        if loader._iterator is not None:
            loader._iterator._shutdown_workers()


def test_gpu_timing_uses_events_and_one_phase_completion_wait():
    from src.aachen_ablation.timing import PhaseTimer
    class Event:
        def record(self, stream):
            assert stream == "selected-stream"
        def elapsed_time(self, other):
            return 25.
    with patch("torch.cuda.Event", side_effect=lambda **kwargs: Event()), \
         patch("torch.cuda.current_stream", return_value="selected-stream") as stream, \
         patch("torch.cuda.synchronize") as sync:
        timer = PhaseTimer(torch.device("cuda:2"))
        for _ in range(3):
            timer.loaded()
            timer.decoder_done()
            timer.computed()
        sync.assert_not_called()
        result = timer.finish()
        sync.assert_called_once_with(2)
        assert all(call.args == (2,) for call in stream.call_args_list)
        assert result["gpu_computation_seconds"] == .075
        assert result["decoder_seconds"] == .075
    with patch("torch.cuda.Event", side_effect=RuntimeError("unsupported timing")), \
         patch("torch.cuda.synchronize"):
        timer = PhaseTimer(torch.device("cuda:0"))
        timer.loaded()
        timer.decoder_done()
        timer.computed()
        assert timer.finish()["gpu_computation_seconds"] is None


def test_validation_rejects_nonfinite_predictions_even_in_ignore_regions():
    v = VARIANTS["A1"]
    data = [{"features": torch.full((1, 384, *v.grid), float("nan")),
             "target": torch.full((9, 13), 255, dtype=torch.long), "record": {"image_id": "bad"}}]
    with pytest.raises(ValueError, match="Nonfinite"):
        engine.evaluate(engine.make_decoder("linear"), data, v, "cpu")
