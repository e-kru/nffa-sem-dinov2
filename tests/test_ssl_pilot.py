"""Integration tests use Meta's external code; no downloaded weights needed.

The reference test executes the original SSLMetaArch.forward_backward on CPU,
substituting only CUDA transfer, FSDP resharding, and xFormers tensor packing.
It compares the original loss outputs AND all student gradients.
"""

import csv
import importlib
from pathlib import Path
import sys
import types
from unittest import mock

import pytest
import torch
from torch import nn

pytest.importorskip("omegaconf")
from src.ssl.upstream import PIN, ROOT, activate, checkout_path

if not checkout_path().is_dir():
    pytest.skip("Install the pinned external checkout for SSL integration tests", allow_module_level=True)
activate()
from src.ssl.adapter import SingleGPUAdapter, load_config, schedules, train_step
from src.ssl.data import augmentation_and_collate, train_records


@pytest.fixture
def cfg():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    config = load_config(ROOT / "configs/sem_ssl_pilot.yaml")
    config.crops.global_crops_size = 56
    config.crops.local_crops_size = 28
    config.crops.local_crops_number = 2
    config.train.batch_size_per_gpu = 2
    config.student.drop_path_rate = 0
    config.dino.head_n_prototypes = 32
    config.dino.head_hidden_dim = 64
    config.dino.head_bottleneck_dim = 16
    return config


@pytest.fixture
def model(cfg):
    return SingleGPUAdapter(cfg, None, allow_random_init=True)


def batch(cfg):
    from dinov2.data import collate_data_and_cast
    # Distinct masks and images make reversal / masked index mistakes visible.
    class FixedMask:
        def __call__(self, count):
            result = torch.zeros(16, dtype=torch.bool)
            result[:count] = True
            return result.reshape(4, 4).numpy()
    samples = [({"global_crops": [torch.randn(3, 56, 56) for _ in range(2)],
                 "local_crops": [torch.randn(3, 28, 28) for _ in range(2)]}, ()) for _ in range(2)]
    return collate_data_and_cast(samples, (0.25, 0.5), 0.5, torch.float32,
                                 n_tokens=16, mask_generator=FixedMask())


def test_original_components_and_teacher_invariants(model):
    from dinov2.loss import DINOLoss, iBOTPatchLoss, KoLeoLoss
    from dinov2.layers import DINOHead
    assert type(model.dino_loss) is DINOLoss
    assert type(model.ibot_patch_loss) is iBOTPatchLoss
    assert type(model.koleo_loss) is KoLeoLoss
    assert type(model.student.dino_head) is DINOHead
    assert model.student.backbone.__class__.__module__ == "dinov2.models.vision_transformer"
    model.eval().train()
    assert model.student.training and not model.teacher.training
    assert all(not p.requires_grad for p in model.teacher.parameters())
    assert all(torch.equal(v, model.teacher.state_dict()[k]) for k, v in model.student.state_dict().items())


def test_ema_matches_every_parameter_and_updates_after_optimizer(model, cfg):
    before = {k: p.clone() for k, p in model.teacher.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameter_groups())
    # Constant positive LR ensures this check exercises real optimization.
    sched = {"lr": [1e-3], "last_layer_lr": [1e-3], "weight_decay": [0.04],
             "momentum": [0.9], "teacher_temp": [0.04]}
    row = train_step(model, optimizer, batch(cfg), sched, 0)
    assert row["total_loss"] > 0
    changed = False
    for name, target in model.teacher.named_parameters():
        student = dict(model.student.named_parameters())[name]
        torch.testing.assert_close(target, before[name] * 0.9 + student * 0.1)
        changed |= not torch.equal(student, before[name])
        assert target.grad is None
    assert changed


def test_delayed_centers_survive_flush_and_state_dict(model, cfg):
    model(batch(cfg), 0.04)[0].backward()
    assert not model.dino_loss.updated and not model.ibot_patch_loss.updated
    model.dino_loss.apply_center_update()
    model.ibot_patch_loss.apply_center_update()
    restored = SingleGPUAdapter(cfg, None, allow_random_init=True)
    restored.load_state_dict(model.state_dict())
    next_batch = batch(cfg)
    actual, _ = model(next_batch, 0.04)
    expected, _ = restored(next_batch, 0.04)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(model.dino_loss.center, restored.dino_loss.center)
    torch.testing.assert_close(model.ibot_patch_loss.center, restored.ibot_patch_loss.center)


def test_original_forward_backward_loss_and_gradient_parity(model, cfg):
    # Test-only stand-in for fmha packing: DINOHead is token-wise and has no BN.
    class Packed:
        @classmethod
        def from_tensor_list(cls, tensors):
            result = cls()
            result.lengths = [x.shape[1] for x in tensors]
            return result, torch.cat(tensors, dim=1)

        def split(self, value):
            return list(value.split(self.lengths, dim=1))

    xformers = types.ModuleType("xformers")
    ops = types.ModuleType("xformers.ops")
    ops.fmha = types.SimpleNamespace(BlockDiagonalMask=Packed)
    xformers.ops = ops
    with mock.patch.dict(sys.modules, {"xformers": xformers, "xformers.ops": ops}):
        upstream_module = importlib.import_module("dinov2.train.ssl_meta_arch")

    reference_cfg = cfg.copy()
    reference_cfg.compute_precision.grad_scaler = False
    reference_cfg.crops.global_crops_size = 518
    reference = upstream_module.SSLMetaArch(reference_cfg)
    reference.student.load_state_dict(model.student.state_dict())
    reference.teacher.load_state_dict(model.teacher.state_dict())
    reference.need_to_synchronize_fsdp_streams = False
    reference.train()

    class SeparateCropCalls(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, images, masks=None, **kwargs):
            if isinstance(images, list):
                return [self.inner(x, masks=m, **kwargs) for x, m in zip(images, masks)]
            return self.inner(images, masks=masks, **kwargs)

    original_backbone = reference.student.backbone
    reference.student["backbone"] = SeparateCropCalls(original_backbone)
    images = batch(cfg)
    with mock.patch.object(torch.Tensor, "cuda", lambda self, **kw: self), \
            mock.patch.object(upstream_module, "reshard_fsdp_model", lambda _: None):
        expected = reference.forward_backward(images, teacher_temp=0.04)
    total, actual = model(images, 0.04)
    total.backward()
    for name in ("dino_local_crops_loss", "dino_global_crops_loss"):
        torch.testing.assert_close(actual[name], expected[name], atol=1e-6, rtol=1e-5)
    # Meta logs half of the iBOT/KoLeo contribution used for backpropagation.
    for name in ("ibot_loss", "koleo_loss"):
        torch.testing.assert_close(actual[name], expected[name] * 2, atol=1e-6, rtol=1e-5)
    for adapted, original in ((model.student.backbone, original_backbone),
                              (model.student.dino_head, reference.student.dino_head)):
        reference_params = dict(original.named_parameters())
        for name, param in adapted.named_parameters():
            expected_grad = reference_params[name].grad
            assert (param.grad is None) == (expected_grad is None), name
            if param.grad is not None:
                torch.testing.assert_close(param.grad, expected_grad, atol=3e-5, rtol=3e-4, msg=name)


def test_schedules_reuse_meta_cosines_and_last_layer_freeze(cfg):
    sched = schedules(cfg)
    assert sched["lr"].__class__.__module__ == "dinov2.utils.utils"
    assert sched["lr"][0] == 0
    assert sched["last_layer_lr"][99] == 0
    assert sched["last_layer_lr"][100] > 0
    assert sched["momentum"][0] == cfg.teacher.momentum_teacher
    assert sched["teacher_temp"][0] == cfg.teacher.warmup_teacher_temp
    assert sched["teacher_temp"][100] == cfg.teacher.teacher_temp


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_manifest_inclusion_and_both_evaluation_holdouts(tmp_path):
    manifest, primary, secondary = [tmp_path / p for p in ("manifest.csv", "B.csv", "A.csv")]
    write_csv(manifest, [{"image_id": p, "included": "0" if p == "excluded" else "1",
                          "crop_mode": "crop_v2"} for p in ("train", "linked", "other", "val", "test", "excluded")])
    b = [{"image_id": p, "group_id": g, "split": s} for p, g, s in (
        ("train", "b0", "train"), ("linked", "b0", "train"), ("other", "b1", "train"),
        ("val", "b2", "val"), ("test", "b3", "test"), ("excluded", "b4", "train"))]
    write_csv(primary, b)
    a = [{**r, "group_id": f"a{i}", "split": "test" if r["image_id"] == "linked" else r["split"]}
         for i, r in enumerate(b)]
    write_csv(secondary, a)
    assert {r["image_id"] for r in train_records(manifest, primary)} == {"train", "linked", "other"}
    assert [r["image_id"] for r in train_records(manifest, primary, [secondary])] == ["other"]
    b[1]["split"] = "val"
    write_csv(primary, b)
    with pytest.raises(ValueError, match="Group crosses"):
        train_records(manifest, primary)


def test_dependency_revision_and_modifications_rejected(tmp_path):
    with mock.patch("src.ssl.upstream.git_output", return_value="wrong"):
        with pytest.raises(RuntimeError, match="pinned"):
            activate(tmp_path)
    with mock.patch("src.ssl.upstream.git_output", side_effect=[PIN["commit"], " M dinov2/loss/dino_clstoken_loss.py"]):
        with pytest.raises(RuntimeError, match="clean"):
            activate(tmp_path)


def test_unmasked_pilot_batch_rejected(model, cfg):
    images = batch(cfg)
    images["mask_indices_list"] = torch.empty(0, dtype=torch.int64)
    with pytest.raises(ValueError, match="No masked patches"):
        model(images, 0.04)
