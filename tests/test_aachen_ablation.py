"""Offline synthetic coverage of geometry, scientific protocol and stage lifecycle."""
import csv
import json
import math
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from scripts.run_aachen_ablation import main
from src.aachen_ablation.cache import (
    OriginalCachedDataset, SyntheticEncoder, cache_contract, extract_cache, validate_cache,
)
from src.aachen_ablation.data import inspect_data, load_original, manifest_rows
from src.aachen_ablation.engine import (evaluate, forward_batch, make_decoder, original_loss,
                                       train_epoch, train_run)
from src.aachen_ablation.geometry import (VARIANTS, content_box, prepare_region,
                                         project_logits, reconstruct, regions)
from src.aachen_ablation.metrics import (component_rows, confusion, scores,
                                        summarize_components, training_size_bins)
from src.downstream.model import DINOv2Backbone


@pytest.fixture
def fixture_manifest(tmp_path):
    rows = []
    for i, split in enumerate(["train", "train", "train", "val", "val", "test"]):
        row = dict(image_id=f"i{i}", image_path=f"im{i}.png", mask_path=f"mask{i}.png",
                   group_id=f"g{i}", split=split)
        if split != "test":  # Missing test files MUST never be opened, including reports.
            h, w = 9+i%2, 13+i%2
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[1:3, 1:3] = 1
            mask[h-2, w-2] = 1
            mask[0, 0] = 255
            image = np.repeat((mask == 1)[:, :, None] * np.uint8(255), 3, axis=2)
            Image.fromarray(image).save(tmp_path / row["image_path"])
            Image.fromarray(mask).save(tmp_path / row["mask_path"])
        rows.append(row)
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, rows)
    return manifest


def write_manifest(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class FakeDino(nn.Module):
    def get_intermediate_layers(self, images, n, reshape, norm):
        assert (n, reshape, norm) == (1, True, True)
        return (nn.functional.avg_pool2d(images, 14).repeat(1, 128, 1, 1),)


@pytest.mark.parametrize("name,grid,tokens", [("A1", (24, 32), 768), ("A2", (36, 48), 1728),
                                            ("A3", (48, 64), 3072), ("B1", (24, 32), 3072)])
def test_feature_grids_and_frozen_encoder(name, grid, tokens):
    variant = VARIANTS[name]
    model = DINOv2Backbone(model=FakeDino(), frozen=True)
    model.train()
    assert not model.model.training
    with torch.inference_mode():
        feature = model.forward_dense(torch.zeros(1, 3, variant.height, variant.width))
    assert feature.shape == (1, 384, *grid)
    assert variant.tokens == tokens
    for kind in ("linear", "cnn8"):
        decoder = make_decoder(kind)
        assert decoder(feature.clone()).shape == (1, 2, *grid)


@pytest.mark.parametrize("size", [(12, 8), (13, 9), (13, 8), (12, 9)])
def test_quadrants_cover_exactly_once_and_reconstruct(size):
    w, h = size
    coverage = np.zeros((h, w), dtype=int)
    boxes = regions(w, h, True)
    for x0, y0, x1, y1 in boxes:
        coverage[y0:y1, x0:x1] += 1
    np.testing.assert_array_equal(coverage, np.ones_like(coverage))
    logits = torch.stack([torch.full((2, 24, 32), float(i)) for i in range(4)])
    full = reconstruct(logits, size, VARIANTS["B1"])
    assert full.shape == (2, h, w)
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        torch.testing.assert_close(full[:, y0:y1, x0:x1], torch.full((2, y1-y0, x1-x0), float(i)))


def test_alignment_original_masks_and_no_distortion(fixture_manifest):
    rows = manifest_rows(fixture_manifest)
    image, target = load_original(fixture_manifest, rows[0])
    reference = target.clone()
    for variant in VARIANTS.values():
        for box in regions(*image.size, variant.quadrants):
            crop_mask = target[box[1]:box[3], box[0]:box[2]]
            # Recover raw red intensity; compare with a bilinear visualization of
            # the SAME original label coordinates (categorical GT itself never resized).
            x = prepare_region(image, box, variant)
            red = x[0] * .229 + .485
            inner = content_box(box[2]-box[0], box[3]-box[1], variant)
            x0, y0, x1, y1 = inner
            rendered = Image.fromarray((crop_mask.numpy() == 1).astype(np.uint8)*255)
            expected = np.asarray(rendered.resize((x1-x0, y1-y0), Image.Resampling.BILINEAR))/255
            np.testing.assert_allclose(red[y0:y1, x0:x1], expected, atol=1e-6)
            assert abs((x1-x0) / (box[2]-box[0]) - (y1-y0)/(box[3]-box[1])) < .2
        _, actual = load_original(fixture_manifest, rows[0])
        assert torch.equal(actual, reference)
    assert set(target.unique().tolist()) == {0, 1, 255}


def test_padding_removed_before_original_projection():
    variant = VARIANTS["A1"]
    # 2:1 image: vertical padding should not be retained in reconstructed content.
    box = content_box(28, 14, variant)
    assert box == (0, 56, 448, 280)
    z = torch.zeros(2, 24, 32)
    z[:, 4:20] = 1
    output = project_logits(z, (28, 14), variant)
    assert output.shape == (2, 14, 28)
    assert output[:, 7, 14].min() == 1


@pytest.mark.parametrize("mutation", ["group", "image", "mask", "id"])
def test_manifest_identity_and_leakage_rejection(fixture_manifest, mutation):
    with fixture_manifest.open() as f:
        rows = list(csv.DictReader(f))
    field = {"group": "group_id", "image": "image_path", "mask": "mask_path", "id": "image_id"}[mutation]
    rows[3][field] = rows[0][field]
    write_manifest(fixture_manifest, rows)
    with pytest.raises(ValueError):
        manifest_rows(fixture_manifest)


def test_test_files_are_never_opened_and_invalid_labels_rejected(fixture_manifest):
    rows = manifest_rows(fixture_manifest)
    assert len(rows) == 5
    assert {r["split"] for r in rows} == {"train", "val"}
    records, info = inspect_data(fixture_manifest, rows, list(VARIANTS.values()))
    assert len(records) == 5 and info["test_files_opened"] is False
    with pytest.raises(ValueError, match="Test data"):
        load_original(fixture_manifest, dict(rows[0], split="test"))
    Image.fromarray(np.full((9, 13), 2, dtype=np.uint8)).save(fixture_manifest.parent / rows[0]["mask_path"])
    with pytest.raises(ValueError, match="Invalid labels"):
        load_original(fixture_manifest, rows[0])


def test_confusion_ignore_and_ma_metrics():
    target = np.array([[0, 0, 1], [1, 1, 255]])
    probability = np.array([[.1, .8, .8], [.2, .9, np.nan]])
    cm = confusion(probability, target, .5)
    assert cm.tolist() == [[1, 1], [1, 2]]
    result = scores(cm)
    assert result["ma_iou"] == .5
    assert result["ma_recall"] == 2/3
    assert result["background_iou"] == 1/3
    assert result["ma_precision"] == 2/3


def test_component_bins_recall_and_connectivity():
    train = np.zeros((12, 15), dtype=np.uint8)
    train[0, 0] = 1
    train[3:5, 3:5] = 1
    train[8:11, 8:11] = 1
    bins = training_size_bins([train])
    assert bins["train_component_count"] == 3
    prediction = np.zeros_like(train, dtype=bool)
    prediction[0, 0] = True
    prediction[3, 3] = True
    rows = component_rows(train, prediction, bins, "test", .5)
    summary = summarize_components(rows)
    assert [r["area_pixels"] for r in rows] == [1, 4, 9]
    assert summary["small"]["pixel_weighted_recall"] == 1
    assert summary["medium"]["mean_island_pixel_recall"] == .25
    assert summary["medium"]["fraction_detected_at_25pct"] == 1
    assert summary["large"]["pixel_weighted_recall"] == 0
    # Diagonal pixels join under 8-connectivity; ignore is never foreground.
    assert training_size_bins([np.array([[1, 255], [0, 1]])])["train_component_count"] == 1
    equal = training_size_bins([np.array([[1, 0, 1]])])
    assert summarize_components(component_rows(np.array([[1]]), np.array([[True]]), equal, "x", .5))["medium"]["gt_islands"] == 0


def test_original_loss_uses_all_valid_pixels_of_odd_quadrants():
    target = torch.zeros(9, 13, dtype=torch.long)
    target[:4, :6] = 1
    target[0, 0] = 255
    z = torch.randn(4, 2, 24, 32, requires_grad=True)
    full = reconstruct(z, (13, 9), VARIANTS["B1"])
    expected = nn.functional.cross_entropy(full[None], target[None], ignore_index=255)
    actual = original_loss(z, target, VARIANTS["B1"])
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert z.grad is not None


@pytest.mark.parametrize("name", ["A1", "B1"])
def test_shared_decoder_and_equal_optimizer_step_budget(name):
    v = VARIANTS[name]
    q = 4 if v.quadrants else 1
    dataset = [{"features": torch.randn(q, 384, *v.grid), "target": torch.zeros(9, 13, dtype=torch.long),
                "record": {"image_id": str(i)}} for i in range(5)]
    model = make_decoder("linear")
    optimizer = torch.optim.AdamW(model.parameters())
    with patch.object(optimizer, "step", wraps=optimizer.step) as steps:
        result = train_epoch(model, dataset, optimizer, v, torch.device("cpu"), 2, 1, 42, 1)
    assert steps.call_count == math.ceil(5/2)
    assert result["originals_seen"] == 5
    assert sum(p.numel() for p in model.parameters()) == 384*2+2
    seen = []
    hook = model.register_forward_hook(lambda module, args, output: seen.append(args[0].shape[0]))
    forward_batch(model, dataset[:2], torch.device("cpu"))
    hook.remove()
    assert seen == [2*q]


def make_cache(tmp_path, manifest, name="B1"):
    rows = manifest_rows(manifest)
    variant = VARIANTS[name]
    records, _ = inspect_data(manifest, rows, [variant])
    contract = cache_contract(manifest, records, variant, "float32", None, True)
    path = tmp_path / f"cache_{name}"
    extract_cache(path, contract, manifest, variant, torch.device("cpu"), SyntheticEncoder().eval(), workers=0)
    return path, contract


def test_cache_metadata_identity_corruption_and_resume(tmp_path, fixture_manifest):
    path, contract = make_cache(tmp_path, fixture_manifest)
    ds = OriginalCachedDataset(path, contract, fixture_manifest, "train")
    assert len(ds) == 3
    assert ds[0]["features"].shape == (4, 384, 24, 32)
    assert ds[0]["record"]["group_id"] == "g0"
    with pytest.raises(ValueError):
        OriginalCachedDataset(path, contract, fixture_manifest, "test")
    changed = json.loads(json.dumps(contract))
    changed["geometry"]["width"] = 896
    with pytest.raises(ValueError, match="metadata"):
        validate_cache(path, changed)
    legacy = tmp_path / "old"
    legacy.mkdir()
    (legacy / "cache_metadata.json").write_text('{"input_shape_per_image": [3,336,448]}')
    with pytest.raises(ValueError):
        validate_cache(legacy, contract)
    meta_path = path / "cache_metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["samples"][0]["record"]["group_id"] = "wrong"
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="identity"):
        validate_cache(path, contract)


def test_interrupted_extraction_resumes_without_recomputing(tmp_path, fixture_manifest):
    rows = manifest_rows(fixture_manifest)
    variant = VARIANTS["A1"]
    records, _ = inspect_data(fixture_manifest, rows, [variant])
    contract = cache_contract(fixture_manifest, records, variant, "float32", None, True)
    encoder = SyntheticEncoder().eval()
    path = tmp_path / "partial"
    calls = 0
    real = encoder.forward_dense
    def interrupted(x):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("interrupted")
        return real(x)
    with patch.object(encoder, "forward_dense", side_effect=interrupted), pytest.raises(RuntimeError):
        extract_cache(path, contract, fixture_manifest, variant, torch.device("cpu"), encoder, workers=0)
    original_timestamp = (path / "000000.npy").stat().st_mtime_ns
    extract_cache(path, contract, fixture_manifest, variant, torch.device("cpu"), encoder, resume=True, workers=0)
    assert (path / "000000.npy").stat().st_mtime_ns == original_timestamp
    assert validate_cache(path, contract)["complete"]
    array = np.load(path / "000000.npy")
    array.flat[0] += 1
    np.save(path / "000000.npy", array)
    with pytest.raises(ValueError, match="Corrupt"):
        validate_cache(path, contract)


def test_epoch_resume_matches_uninterrupted_training(tmp_path, fixture_manifest):
    from src.aachen_ablation import engine
    path, contract = make_cache(tmp_path, fixture_manifest, "A1")
    train = OriginalCachedDataset(path, contract, fixture_manifest, "train")
    val = OriginalCachedDataset(path, contract, fixture_manifest, "val")
    config = dict(cache_metadata={}, head="linear", lr=.001, weight_decay=.0001, seed=42,
                  epochs=2, patience=8, batch_size=2, micro_batch_size=1, workers=0)
    v, device = VARIANTS["A1"], torch.device("cpu")
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    before = (torch.are_deterministic_algorithms_enabled(),
              torch.is_deterministic_algorithms_warn_only_enabled(),
              torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
    train_run(full, config, train, val, v, device)
    assert before == (torch.are_deterministic_algorithms_enabled(),
                      torch.is_deterministic_algorithms_warn_only_enabled(),
                      torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
    real = engine.train_epoch
    def interrupted(*args, **kwargs):
        if args[8] == 2:
            raise RuntimeError("interruption")
        return real(*args, **kwargs)
    with patch.object(engine, "train_epoch", side_effect=interrupted), pytest.raises(RuntimeError):
        train_run(resumed, config, train, val, v, device)
    with pytest.raises(FileExistsError):
        train_run(resumed, config, train, val, v, device)
    train_run(resumed, config, train, val, v, device, resume=True)
    a = torch.load(full / "best_model.pt", weights_only=True)
    b = torch.load(resumed / "best_model.pt", weights_only=True)
    def scientific_history(path):
        with (path / "history.csv").open() as handle:
            return [{k: v for k, v in row.items() if not k.endswith("seconds")} for row in csv.DictReader(handle)]
    assert scientific_history(full) == scientific_history(resumed)
    for key in a["model_state_dict"]:
        torch.testing.assert_close(a["model_state_dict"][key], b["model_state_dict"][key], rtol=0, atol=0)


def test_evaluation_thresholds_on_original_raster_and_ignore():
    variant = VARIANTS["B1"]
    features = torch.zeros(4, 384, 24, 32)
    target = torch.zeros(9, 13, dtype=torch.long)
    target[:4, :6] = 1
    target[0, 0] = 255
    features[0, 1] = 10
    features[1:, 0] = 10
    model = nn.Conv2d(384, 2, 1)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
        model.weight[0, 0, 0, 0] = 1
        model.weight[1, 1, 0, 0] = 1
    result = evaluate(model, [{"features": features, "target": target, "record": {"image_id": "x"}}],
                      variant, torch.device("cpu"))
    assert result["thresholds"][0]["ma_iou"] == 1
    assert result["thresholds"][0]["confusion_matrix"] == [[93, 0], [0, 23]]


def test_all_eight_cli_smoke_dry_run_and_reports(tmp_path, fixture_manifest, capsys):
    root = tmp_path / "experiment"
    base = ["--manifest", str(fixture_manifest), "--output-root", str(root),
            "--device", "cpu", "--smoke-test", "--epochs", "1", "--example-count", "1", "--timing-images", "1"]
    with patch("torch.hub.load", side_effect=AssertionError("NO NETWORK")):
        main(base + ["--dry-run"])
        assert not root.exists()
        main(base)
        runs = sorted(root.glob("NONSCIENTIFIC_SMOKE/runs/*/*/*"))
        assert len(runs) == 8
        for run in runs:
            for name in ("config.json", "cache_metadata.json", "history.csv", "best_model.pt",
                         "validation_metrics.json", "threshold_metrics.csv", "component_metrics.csv", "run.log"):
                assert (run / name).is_file()
            metrics = json.loads((run / "validation_metrics.json").read_text())
            assert metrics["test_evaluated"] is False
            assert metrics["nonscientific_smoke_test"] is True
            assert metrics["end_to_end_seconds_per_original"] > 0
        timestamps = {p: p.stat().st_mtime_ns for p in root.rglob("best_model.pt")}
        main(base)  # Safe completed-stage skips; no training/report overwrite.
        assert timestamps == {p: p.stat().st_mtime_ns for p in timestamps}
        reports = list(root.glob("NONSCIENTIFIC_SMOKE/reports/*/report_complete.json"))
        assert len(reports) == 1
        report = json.loads(reports[0].read_text())
        assert report["complete_eight_configurations"] is True
        assert (reports[0].parent / "example_00.png").exists()
        assert (reports[0].parent / "ma_iou_resolution_fixed_0.5.pdf").exists()


def test_default_eight_workers_and_synthetic_parallel_loader(tmp_path, fixture_manifest):
    from scripts.run_aachen_ablation import parser
    assert parser().parse_args([]).workers == 8
    assert parser().parse_args([]).device == "cuda:0"
    assert parser().parse_args([]).determinism == "off"
    from multiprocessing.connection import Listener
    try:
        with Listener(str(tmp_path / "worker.sock"), family="AF_UNIX"):
            pass
    except PermissionError:
        pytest.skip("Local sandbox forbids the Unix sockets required by Torch multiprocessing")
    variant = VARIANTS["A1"]
    records, _ = inspect_data(fixture_manifest, manifest_rows(fixture_manifest), [variant])
    contract = cache_contract(fixture_manifest, records, variant, "float32", None, True)
    path = tmp_path / "parallel"
    extract_cache(path, contract, fixture_manifest, variant, torch.device("cpu"),
                  SyntheticEncoder().eval(), workers=2)
    train_ds = OriginalCachedDataset(path, contract, fixture_manifest, "train")
    val_ds = OriginalCachedDataset(path, contract, fixture_manifest, "val")
    model = make_decoder("linear")
    optimizer = torch.optim.AdamW(model.parameters())
    result = train_epoch(model, train_ds, optimizer, variant, torch.device("cpu"), 2, 1, 42, 1, workers=2)
    assert result["optimizer_steps"] == 2
    assert evaluate(model, val_ds, variant, torch.device("cpu"), workers=2)["loss"] > 0


@pytest.mark.parametrize("entry", ["train", "evaluate"])
@pytest.mark.parametrize("policy", ["off", "strict", "warn"])
@pytest.mark.parametrize("initial", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("fail", [False, True])
def test_determinism_policy_is_scoped_even_on_failure(entry, policy, initial, fail):
    from src.aachen_ablation import engine
    before = (torch.are_deterministic_algorithms_enabled(),
              torch.is_deterministic_algorithms_warn_only_enabled(),
              torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
    try:
        torch.use_deterministic_algorithms(initial[0], warn_only=initial[1])
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        def operation(*args, **kwargs):
            assert torch.are_deterministic_algorithms_enabled() == (policy != "off")
            assert torch.is_deterministic_algorithms_warn_only_enabled() == (policy == "warn")
            assert not torch.backends.cudnn.benchmark
            assert torch.backends.cudnn.deterministic == (policy != "off")
            if fail:
                raise RuntimeError("synthetic failure")
            return "complete"
        args = (None, {"determinism_policy": policy}, None, None, None, None)
        with patch.object(engine, f"_{entry}_run", side_effect=operation):
            if fail:
                with pytest.raises(RuntimeError, match="synthetic failure"):
                    getattr(engine, f"{entry}_run")(*args)
            else:
                assert getattr(engine, f"{entry}_run")(*args) == "complete"
        assert torch.are_deterministic_algorithms_enabled() == initial[0]
        assert torch.is_deterministic_algorithms_warn_only_enabled() == initial[1]
        assert torch.backends.cudnn.benchmark
        assert not torch.backends.cudnn.deterministic
    finally:
        torch.use_deterministic_algorithms(before[0], warn_only=before[1])
        torch.backends.cudnn.benchmark = before[2]
        torch.backends.cudnn.deterministic = before[3]


@pytest.mark.parametrize("spec,expected", [
    ("cpu", "cpu"), ("cpu:0", "cpu"), (torch.device("cpu"), "cpu"),
    ("cuda:0", "cuda:0"), (torch.device("cuda:2"), "cuda:2"), (0, "cuda:0"),
    (2, "cuda:2"), ("cuda", "cuda:1"), (torch.device("cuda"), "cuda:1"),
])
def test_device_normalization(spec, expected):
    from src.aachen_ablation.devices import normalize_device
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.current_device", return_value=1):
        assert str(normalize_device(spec)) == expected


@pytest.mark.parametrize("spec", [-1, True, "cuda:-1", "not-a-device", "mps", None])
def test_invalid_device_specification(spec):
    from src.aachen_ablation.devices import normalize_device
    with pytest.raises(ValueError):
        normalize_device(spec)


def test_visible_cuda_ordinal_is_validated_and_selected(monkeypatch):
    from src.aachen_ablation.devices import select_device
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.device_count", return_value=1), \
         patch("torch.cuda.set_device") as select:
        assert select_device("cuda:0") == torch.device("cuda:0")
        select.assert_called_once_with(0)  # Logical 0, never physical 7.
        with pytest.raises(ValueError, match="logical devices visible"):
            select_device("cuda:7")
        select.assert_called_once_with(0)
    with patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(RuntimeError, match="CUDA unavailable"):
            select_device("cuda:0")


def test_lazy_cuda_is_initialized_before_allocator_reset():
    from src.aachen_ablation.devices import CudaMemoryProfiler
    events = []
    def init():
        events.append("init")
    def reset(index):
        if events != ["init"]:
            raise RuntimeError("Invalid device argument")
        assert index == 2 and isinstance(index, int)
        events.append("reset")
    with patch("torch.cuda.init", side_effect=init), \
         patch("torch.cuda.reset_peak_memory_stats", side_effect=reset), \
         patch("torch.cuda.max_memory_allocated", return_value=1234) as peak:
        profile = CudaMemoryProfiler(torch.device("cuda:2"))
        assert profile.start()
        assert profile.peak_bytes() == 1234
        peak.assert_called_once_with(2)
        assert profile.metadata() == {"device": "cuda:2", "available": True, "errors": []}


@pytest.mark.parametrize("operation", ["init", "reset_peak_memory_stats", "max_memory_allocated"])
def test_optional_cuda_profile_failures_return_null(operation, caplog):
    from src.aachen_ablation.devices import CudaMemoryProfiler
    with patch("torch.cuda.init") as init, \
         patch("torch.cuda.reset_peak_memory_stats") as reset, \
         patch("torch.cuda.max_memory_allocated", return_value=1234) as peak:
        {"init": init, "reset_peak_memory_stats": reset, "max_memory_allocated": peak}[operation].side_effect = RuntimeError("Invalid device argument")
        profile = CudaMemoryProfiler(0)
        profile.start()
        assert profile.peak_bytes() is None
        assert not profile.metadata()["available"]
        assert "Invalid device argument" in profile.metadata()["errors"][0]
        assert "Optional CUDA memory profiling unavailable" in caplog.text
        if operation != "max_memory_allocated":
            peak.assert_not_called()  # A failed reset must never report a stale peak.


def test_cpu_device_helpers_never_touch_cuda():
    from src.aachen_ablation.devices import CudaMemoryProfiler, select_device
    from src.aachen_ablation.engine import rng_state, restore_rng, sync
    with patch("torch.cuda.is_available", side_effect=AssertionError("CUDA query")), \
         patch("torch.cuda.init", side_effect=AssertionError("CUDA init")), \
         patch("torch.cuda.reset_peak_memory_stats", side_effect=AssertionError("CUDA reset")), \
         patch("torch.cuda.max_memory_allocated", side_effect=AssertionError("CUDA peak")), \
         patch("torch.cuda.get_rng_state", side_effect=AssertionError("CUDA RNG")), \
         patch("torch.cuda.set_rng_state", side_effect=AssertionError("CUDA RNG")), \
         patch("torch.cuda.synchronize", side_effect=AssertionError("CUDA sync")):
        assert select_device("cpu:0") == torch.device("cpu")
        profile = CudaMemoryProfiler("cpu")
        assert not profile.start()
        assert profile.peak_bytes() is None
        sync("cpu")
        restore_rng(rng_state("cpu"), "cpu")


def test_cuda_sync_and_rng_use_only_selected_logical_device():
    from src.aachen_ablation.engine import rng_state, restore_rng, sync
    value = torch.get_rng_state()
    with patch("torch.cuda.synchronize") as synchronize, \
         patch("torch.cuda.get_rng_state", return_value=value) as get_rng, \
         patch("torch.cuda.set_rng_state") as set_rng:
        sync("cuda:2")
        synchronize.assert_called_once_with(2)
        state = rng_state(2)
        get_rng.assert_called_once_with(2)
        restore_rng(state, torch.device("cuda:2"))
        assert set_rng.call_args.args[1] == 2
        assert torch.equal(set_rng.call_args.args[0], value)
        # Required synchronization failures are NOT treated as optional telemetry.
        synchronize.side_effect = RuntimeError("real CUDA failure")
        with pytest.raises(RuntimeError, match="real CUDA failure"):
            sync(2)


@pytest.mark.parametrize("failure", ["reset", "read"])
def test_mock_cuda_extraction_completes_despite_profiling_failure(tmp_path, fixture_manifest, failure):
    """Run real synthetic extraction; only device transfer/driver APIs are mocked."""
    from src.aachen_ablation import cache
    variant = VARIANTS["A1"]
    records, _ = inspect_data(fixture_manifest, manifest_rows(fixture_manifest), [variant])
    contract = cache_contract(fixture_manifest, records, variant, "float32", None, True)
    path = tmp_path / "mock_cuda"
    real_to, real_loader = torch.Tensor.to, cache.DataLoader
    transfers = []
    def transfer(tensor, *args, **kwargs):
        if args and isinstance(args[0], torch.device) and args[0].type == "cuda":
            transfers.append(str(args[0]))
            args = (torch.device("cpu"), *args[1:])
        return real_to(tensor, *args, **kwargs)
    def loader(*args, **kwargs):
        kwargs["pin_memory"] = False  # No actual accelerator in this offline test.
        return real_loader(*args, **kwargs)
    with patch.object(torch.Tensor, "to", transfer), patch.object(cache, "DataLoader", loader), \
         patch("torch.cuda.init"), \
         patch("torch.cuda.reset_peak_memory_stats", side_effect=RuntimeError("Invalid device argument") if failure == "reset" else None), \
         patch("torch.cuda.max_memory_allocated", side_effect=RuntimeError("unavailable peak")) as peak:
        meta = extract_cache(path, contract, fixture_manifest, variant, "cuda:0",
                             SyntheticEncoder().eval(), workers=0)
        assert meta["complete"]
        assert meta["extraction_peak_gpu_bytes"] is None
        assert meta["extraction_memory_profiling"]["errors"]
        assert not meta["extraction_memory_profiling"]["available"]
        assert transfers == ["cuda:0"] * len(records)
        assert validate_cache(path, contract)["complete"]
        if failure == "reset":
            peak.assert_not_called()
        # Compute failures must still stop extraction, even when profiling fails.
        with patch.object(SyntheticEncoder, "forward_dense", side_effect=RuntimeError("real compute failure")), \
             pytest.raises(RuntimeError, match="real compute failure"):
            extract_cache(tmp_path / "broken_compute", contract, fixture_manifest, variant,
                          "cuda:0", SyntheticEncoder().eval(), workers=0)


@pytest.mark.parametrize("failure", ["reset", "read"])
def test_every_stage_records_optional_profiling_failure(tmp_path, fixture_manifest, failure):
    """Inject mocked CUDA telemetry into real CPU stages; no CUDA compute needed."""
    from scripts import run_aachen_ablation as runner
    from src.aachen_ablation import cache, engine
    from src.aachen_ablation.devices import CudaMemoryProfiler
    root = tmp_path / "profiling_failure"
    def profiler(device):
        return CudaMemoryProfiler("cuda:0")
    with patch.object(cache, "CudaMemoryProfiler", side_effect=profiler), \
         patch.object(engine, "CudaMemoryProfiler", side_effect=profiler), \
         patch.object(runner, "CudaMemoryProfiler", side_effect=profiler), \
         patch("torch.cuda.init"), \
         patch("torch.cuda.reset_peak_memory_stats", side_effect=RuntimeError("Invalid device argument") if failure == "reset" else None) as reset, \
         patch("torch.cuda.max_memory_allocated", side_effect=RuntimeError("unavailable peak")) as peak, \
         patch("torch.hub.load", side_effect=AssertionError("NO MODEL DOWNLOAD")):
        main(["--manifest", str(fixture_manifest), "--output-root", str(root),
              "--smoke-test", "--device", "cpu", "--variants", "A1", "--heads", "linear",
              "--epochs", "1", "--timing-images", "1", "--example-count", "0"])
        assert reset.call_count == 4  # extraction, training, validation, timing
        assert all(call.args == (0,) for call in reset.call_args_list)
        assert peak.call_count == (4 if failure == "read" else 0)
    path = next(root.glob("NONSCIENTIFIC_SMOKE/runs/A1/linear/*/validation_metrics.json"))
    metrics = json.loads(path.read_text())
    for prefix in ("train", "eval", "timing"):
        assert metrics[prefix+"_peak_gpu_bytes"] is None
        assert metrics[prefix+"_memory_profiling"]["errors"]
        assert not metrics[prefix+"_memory_profiling"]["available"]
    assert metrics["end_to_end_seconds_per_original"] > 0
