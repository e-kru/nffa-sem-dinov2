"""Frozen benchmark integration: existing heads, cache isolation and no test access."""
import copy
import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import joblib
import numpy as np
import pytest
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from scripts import evaluate_sem_encoder as runner
from src.aachen_ablation.engine import make_decoder
from src.downstream.benchmark import (
    arrays, classification_records, extract_features, feature_contract, validate_features,
)
from src.downstream.heads import ClassificationHead, DeepCNN, SegmentationHead
from src.downstream.probes import HEAD_CONFIGS, fit_logreg, fit_mlp, mlp_model


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def fixtures(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    rows, inventory = [], []
    for i, split in enumerate(["train"]*8 + ["val"]*4 + ["test"]):
        name = f"{i}.png"
        if split != "test":
            Image.fromarray(np.full((28, 42, 3), (i % 2)*90+30, np.uint8)).save(images / name)
        # A poison test label must never be converted to int, and the file is absent.
        label = str(i % 2) if split != "test" else "FORBIDDEN TEST LABEL"
        rows.append(dict(image_id=name, label=label, group_id=f"g{i}", split=split))
        inventory.append(dict(image_id=name, label=label, included="1"))
    split_a, split_b, manifest = tmp_path / "a.csv", tmp_path / "b.csv", tmp_path / "manifest.csv"
    write_csv(split_a, rows)
    write_csv(split_b, rows)
    write_csv(manifest, inventory)
    segrows = []
    for i, split in enumerate(["train", "train", "val", "val", "test"]):
        row = dict(image_id=f"seg{i}", image_path=f"rgb{i}.png", mask_path=f"mask{i}.png",
                   group_id=f"sg{i}", split=split)
        if split != "test":
            mask = np.zeros((15+i % 2, 21+i % 2), dtype=np.uint8)
            mask[2:5, 2:5] = 1
            mask[0, 0] = 255
            Image.fromarray(mask).save(tmp_path / row["mask_path"])
            Image.fromarray(np.repeat((mask == 1)[:, :, None]*np.uint8(190), 3, axis=2)).save(tmp_path / row["image_path"])
        segrows.append(row)
    seg = tmp_path / "seg.csv"
    write_csv(seg, segrows)
    return SimpleNamespace(images=images, manifest=manifest, a=split_a, b=split_b,
                           seg=seg, rows=rows, root=tmp_path)


def arguments(f):
    return ["--classification-manifest", str(f.manifest), "--split-a", str(f.a),
            "--split-b", str(f.b), "--image-root", str(f.images),
            "--segmentation-manifest", str(f.seg), "--output-root", str(f.root / "out"),
            "--device", "cpu", "--smoke-test"]


def test_selection_defaults_explicit_and_invalid_heads():
    p = runner.parser()
    assert runner.selected_heads(p.parse_args([])) == runner.DEFAULT_HEADS
    assert p.parse_args([]).workers == 8
    for task, heads in runner.DEFAULT_HEADS.items():
        args = p.parse_args(["--task", task, "--heads", *reversed(heads)])
        assert runner.selected_heads(args) == {task: list(reversed(heads))}
    for args in (["--heads", "mlp"], ["--task", "classification", "--heads", "cnn8"],
                 ["--task", "segmentation", "--heads", "linear", "linear"],
                 ["--task", "classification", "--heads", "mlp", "--classification-heads", "logreg"]):
        with pytest.raises(ValueError):
            runner.selected_heads(p.parse_args(args))
    assert runner.selected_heads(p.parse_args(["--task", "both", "--classification-heads", "mlp",
                                               "--segmentation-heads", "cnn8"])) == {
        "classification": ["mlp"], "segmentation": ["cnn8"]}


def test_existing_head_architectures_and_original_hyperparameters():
    config = HEAD_CONFIGS["mlp"]
    head = mlp_model(config)
    assert isinstance(head, ClassificationHead)
    assert [(l.in_features, l.out_features) for l in head.net if isinstance(l, torch.nn.Linear)] == [
        (384, 512), (512, 256), (256, 128), (128, 10)]
    assert [l.p for l in head.net if isinstance(l, torch.nn.Dropout)] == [.3]*3
    assert sum(isinstance(l, torch.nn.BatchNorm1d) for l in head.net) == 3
    assert (config["lr"], config["weight_decay"], config["batch_size"], config["epochs"], config["patience"]) == (
        1e-3, 1e-4, 256, 100, 15)
    assert isinstance(make_decoder("linear"), SegmentationHead)
    cnn = make_decoder("cnn8")
    assert isinstance(cnn, DeepCNN)
    assert sum(isinstance(l, torch.nn.Conv2d) and l.kernel_size == (3, 3) for l in cnn.modules()) == 8


def test_logreg_matches_original_six_C_validation_selection(tmp_path):
    rng = np.random.default_rng(42)
    train, val = rng.normal(size=(12, 384)).astype(np.float32), rng.normal(size=(6, 384)).astype(np.float32)
    y, yv = np.arange(12) % 2, np.arange(6) % 2
    fit_logreg(tmp_path, train, y, val, yv, HEAD_CONFIGS["logreg"], 42)
    saved = joblib.load(tmp_path / "model.joblib")
    scaler = StandardScaler().fit(train)
    from sklearn.metrics import f1_score
    best, expected = -1, None
    for C in [.001, .01, .1, 1, 10, 100]:
        baseline = LogisticRegression(C=C, solver="lbfgs", max_iter=3000, random_state=42)
        baseline.fit(scaler.transform(train), y)
        score = f1_score(yv, baseline.predict(scaler.transform(val)), average="macro")
        if score > best:
            best, expected = score, baseline
    np.testing.assert_array_equal(saved["scaler"].mean_, scaler.mean_)
    np.testing.assert_array_equal(saved["classifier"].coef_, expected.coef_)
    assert saved["classifier"].C == expected.C


def test_mlp_matches_archived_original_when_available(tmp_path):
    # Independent regression oracle from the original experiment, when retained.
    source = runner.ROOT / "backups/cleanup_v3_20261021_175045_981536/scripts/train_mlp_head.py"
    if not source.is_file():
        pytest.skip("Optional original archived trainer is not distributed in Git")
    spec = importlib.util.spec_from_file_location("original_mlp", source)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    rng = np.random.default_rng(81)
    X, V = rng.normal(size=(8, 384)).astype(np.float32), rng.normal(size=(4, 384)).astype(np.float32)
    y, v = (np.arange(8) % 2).astype(np.int64), (np.arange(4) % 2).astype(np.int64)
    config = copy.deepcopy(HEAD_CONFIGS["mlp"])
    config["epochs"] = 2
    adapted = tmp_path / "adapted"
    adapted.mkdir()
    fit_mlp(adapted, X, y, V, v, config, 42, 0, torch.device("cpu"))
    def load(*_):
        scaler = StandardScaler().fit(X)
        return scaler.transform(X).astype(np.float32), y, scaler.transform(V).astype(np.float32), v, scaler, (8, 4, 1)
    args = SimpleNamespace(split="A", v2_dir=tmp_path, v3_dir=tmp_path / "original",
                           seed=42, dropout=.3, batch_size=256, lr=.001, weight_decay=.0001, epochs=2, patience=15)
    with patch.object(original, "load_split_data", side_effect=load), patch("torch.cuda.is_available", return_value=False):
        original.train(args)
    baseline_dir = args.v3_dir / "baselines/mlp_split_A"
    old = torch.load(baseline_dir / "best_model.pt", weights_only=True)
    new = torch.load(adapted / "best_model.pt", weights_only=True)
    assert old["epoch"] == new["epoch"]
    for key in old["state_dict"]:
        torch.testing.assert_close(old["state_dict"][key], new["state_dict"][key], rtol=0, atol=0)
    assert (baseline_dir / "history.csv").read_text() == (adapted / "history.csv").read_text()


def test_group_manifest_alignment_and_test_exclusion(fixtures):
    f = fixtures
    records = classification_records(f.manifest, f.a, f.images, workers=0)
    assert len(records) == 12
    assert all(r["split"] != "test" for r in records)
    assert classification_records(f.manifest, f.a, f.images, workers=3) == records
    changed = copy.deepcopy(f.rows)
    changed[8]["group_id"] = changed[0]["group_id"]
    write_csv(f.a, changed)
    with pytest.raises(ValueError, match="Group leakage"):
        classification_records(f.manifest, f.a, f.images, workers=0)
    changed[8]["group_id"] = "unique"
    changed[0]["label"] = "9"
    write_csv(f.a, changed)
    with pytest.raises(ValueError, match="label mismatch"):
        classification_records(f.manifest, f.a, f.images, workers=0)


def test_cache_identity_alignment_and_corruption(fixtures):
    f = fixtures
    records = classification_records(f.manifest, f.a, f.images, workers=0)
    contract = feature_contract(records, f.manifest, f.a, f.images, {"sha": "A"}, True)
    cache = f.root / "cache"
    meta = extract_features(cache, contract, None, torch.device("cpu"), workers=0, smoke=True)
    assert meta["complete"]
    assert [x.shape for x in arrays(cache, contract)] == [(8, 384), (8,), (4, 384), (4,)]
    wrong = dict(contract, encoder={"sha": "B"})
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_features(cache, wrong)
    file = cache / meta["chunks"][0]["file"]
    with np.load(file) as data:
        values = dict(data)
    values["paths"] = values["paths"][::-1]
    np.savez(file, **values)
    with pytest.raises(ValueError, match="Corrupt/misaligned"):
        validate_features(cache, contract)


def test_staged_both_tasks_all_four_heads_and_immutable_results(fixtures):
    f = fixtures
    base = arguments(f)
    with patch("torch.hub.load", side_effect=AssertionError("NO NETWORK OR PRETRAINED MODEL")), \
         patch.object(runner, "REFERENCE", f.root / "nonexistent_release.pth"):
        for stage in ("preflight", "extract", "train", "evaluate", "report"):
            runner.main(base + ["--stage", stage])
    root = f.root / "out/NONSCIENTIFIC_SMOKE/dinov2_vits14_FAKE"
    classification = sorted(root.glob("classification/split_*/*/seed42/validation_metrics.json"))
    segmentation = sorted(root.glob("segmentation/NONSCIENTIFIC_SMOKE/runs/A3/*/*/validation_metrics.json"))
    assert len(classification) == 4 and len(segmentation) == 2
    for path in classification+segmentation:
        result = json.loads(path.read_text())
        assert result["test_evaluated"] is False
        assert result["nonscientific_smoke_test"] is True
    for path in segmentation:
        metrics = json.loads(path.read_text())
        assert all(k in metrics["validation_at_0_50"] for k in ("ma_iou", "miou", "ma_precision", "ma_recall"))
        cache = json.loads((path.parent / "cache_metadata.json").read_text())
        assert cache["contract"]["feature_shape"] == [1, 384, 48, 64]
        assert {r["split"] for r in cache["contract"]["records"]} == {"train", "val"}
    reports = list(root.glob("reports/*/report.json"))
    assert len(reports) == 1
    assert json.loads(reports[0].read_text())["complete"]
    timestamps = {p: p.stat().st_mtime_ns for p in root.rglob("best_model.pt")}
    runner.main(base + ["--stage", "train", "--resume"])
    assert timestamps == {p: p.stat().st_mtime_ns for p in timestamps}


def test_checkpoint_specific_isolation_and_identical_head_settings(fixtures):
    f = fixtures
    base = arguments(f) + ["--task", "classification", "--splits", "A"]
    roots = []
    for number in (1, 2):
        checkpoint = f.root / f"teacher{number}.pth"
        torch.save({"w": torch.tensor(float(number))}, checkpoint)
        args = runner.parser().parse_args(base+["--encoder-checkpoint", str(checkpoint)])
        _, _, root = runner.identity_and_root(args)
        roots.append(root)
        runner.main(base + ["--encoder-checkpoint", str(checkpoint), "--stage", "all"])
    assert roots[0] != roots[1]
    for head in ("logreg", "mlp"):
        configs = [json.loads((r / f"classification/split_A/{head}/seed42/config.json").read_text()) for r in roots]
        assert configs[0]["head_config"] == configs[1]["head_config"]
        assert configs[0]["cache_contract_sha256"] != configs[1]["cache_contract_sha256"]


def test_verification_gate_is_required_and_does_not_claim_smoke_equivalence(fixtures):
    f = fixtures
    args = runner.parser().parse_args(arguments(f))
    checkpoint, identity, root = runner.identity_and_root(args)
    root.mkdir(parents=True)
    with pytest.raises(ValueError, match="preflight"):
        runner.verify_loading(root, checkpoint, args, identity)
    result = runner.verify_loading(root, checkpoint, args, identity, run=True)
    assert result["loading_equivalence_verified"] is False
    with patch.object(runner, "verification_contract", return_value={"changed": "code"}):
        with pytest.raises(ValueError, match="stale/incompatible"):
            runner.verify_loading(root, checkpoint, args, identity)


def test_mlp_epoch_resume_preserves_dropout_shuffle_and_optimizer(tmp_path):
    from src.downstream import probes
    rng = np.random.default_rng(12)
    X, V = rng.normal(size=(8, 384)).astype(np.float32), rng.normal(size=(4, 384)).astype(np.float32)
    y, v = (np.arange(8) % 2).astype(np.int64), (np.arange(4) % 2).astype(np.int64)
    config = copy.deepcopy(HEAD_CONFIGS["mlp"])
    config["epochs"] = 3
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    full.mkdir()
    resumed.mkdir()
    fit_mlp(full, X, y, V, v, config, 42, 0, torch.device("cpu"))
    real_evaluate, calls = probes.evaluate_mlp, 0
    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:  # Interrupt epoch two; epoch one was atomically saved.
            raise RuntimeError("synthetic interruption")
        return real_evaluate(*args, **kwargs)
    with patch.object(probes, "evaluate_mlp", side_effect=interrupted), pytest.raises(RuntimeError):
        fit_mlp(resumed, X, y, V, v, config, 42, 0, torch.device("cpu"))
    fit_mlp(resumed, X, y, V, v, config, 42, 0, torch.device("cpu"), resume=True)
    assert (full / "history.csv").read_bytes() == (resumed / "history.csv").read_bytes()
    a = torch.load(full / "last_state.pt", weights_only=False)
    b = torch.load(resumed / "last_state.pt", weights_only=False)
    for name in a["model"]:
        torch.testing.assert_close(a["model"][name], b["model"][name], atol=0, rtol=0)


def test_segmentation_delegates_identical_protocol_across_checkpoints(fixtures):
    from scripts import run_aachen_ablation
    args = runner.parser().parse_args(arguments(fixtures))
    args.smoke_test = False
    commands = []
    with patch.object(run_aachen_ablation, "main", side_effect=commands.append):
        for name in ("original", "adapted"):
            runner.run_segmentation(args, fixtures.root / name, fixtures.root / f"{name}.pth", ["linear", "cnn8"])
    def protocol(command):
        command = list(command)
        for option in ("--output-root", "--encoder-checkpoint"):
            command[command.index(option)+1] = "ENCODER_SPECIFIC"
        return command
    assert protocol(commands[0]) == protocol(commands[1])
    assert commands[0][commands[0].index("--variants")+1] == "A3"
    assert "--pinned-encoder" in commands[0]


@pytest.mark.parametrize("wrapper", [None, "state_dict", "encoder_state_dict"])
def test_pinned_factory_loads_backbone_exports_without_release_download(tmp_path, wrapper):
    from src.ssl.upstream import activate, checkout_path
    if not checkout_path().is_dir():
        pytest.skip("Optional pinned Meta checkout is not available")
    activate()
    from src.downstream.pinned_encoder import get_pinned_dinov2
    state = torch.nn.Linear(3, 384).state_dict()
    checkpoint = tmp_path / "teacher_backbone.pth"
    torch.save({wrapper: state} if wrapper else state, checkpoint)
    with patch("dinov2.hub.backbones.dinov2_vits14", return_value=torch.nn.Linear(3, 384)) as factory:
        model = get_pinned_dinov2(checkpoint)
    factory.assert_called_once_with(pretrained=False)
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    for key in state:
        torch.testing.assert_close(model.state_dict()[key], state[key], rtol=0, atol=0)
