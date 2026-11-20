"""Single-device orchestration of Meta's original DINOv2 components.

Loss mathematics live exclusively in the external dependency. This module
handles crop routing, loss weights, optimization and parameter EMA. See
docs/sem_ssl_pilot.md for every execution/configuration deviation.
"""

from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn

from src.ssl.upstream import activate


def load_config(path: Path):
    root = activate()
    from omegaconf import OmegaConf
    cfg = OmegaConf.merge(
        OmegaConf.load(root / "dinov2/configs/ssl_default_config.yaml"), OmegaConf.load(path)
    )
    # Narrow support makes semantic differences explicit and rejects silent no-ops.
    cellstyle = cfg.get('recipe') == 'sem_cellstyle_vitl14_v1'
    if (cfg.student.arch != ("vit_large" if cellstyle else "vit_small") or cfg.student.patch_size != 14
            or cfg.student.num_register_tokens != 0 or cfg.student.block_chunks != 0
            or cfg.student.ffn_layer != "mlp" or cfg.student.in_chans != 3
            or cfg.student.channel_adaptive or cfg.student.pretrained_weights
            or cfg.MODEL.WEIGHTS or cfg.train.cell_augmentation):
        raise ValueError("Pilot requires the released RGB ViT-S/14, no registers, unchunked MLP")
    if cfg.train.centering != "centering" or cfg.ibot.separate_head:
        raise ValueError("Pilot supports Meta centering and a shared DINO/iBOT head")
    if min(cfg.dino.loss_weight, cfg.dino.koleo_loss_weight, cfg.ibot.loss_weight) <= 0:
        raise ValueError("All three original objectives must be active")
    if cfg.train.batch_size_per_gpu < 2 or cfg.crops.local_crops_number < 1:
        raise ValueError("KoLeo requires >=2 images per crop; local crops must be present")
    if cfg.crops.global_crops_size % 14 or cfg.crops.local_crops_size % 14:
        raise ValueError("Crop sizes must be multiples of patch size 14")
    if cellstyle:
        if cfg.optim.scaling_rule != 'absolute':
            raise ValueError('Cellstyle learning rates must be absolute')
        cfg.optim.lr = cfg.optim.backbone_lr
        return cfg
    if cfg.optim.scaling_rule != "sqrt_wrt_1024":
        raise ValueError("Use the upstream sqrt_wrt_1024 LR scaling rule")
    cfg.optim.lr = cfg.optim.base_lr * (cfg.train.batch_size_per_gpu / 1024) ** 0.5
    return cfg


class SingleGPUAdapter(nn.Module):
    def __init__(self, cfg, checkpoint: Path | None, *, allow_random_init=False):
        super().__init__()
        activate()
        from dinov2.hub import backbones
        from dinov2.layers import DINOHead
        from dinov2.loss import DINOLoss, iBOTPatchLoss, KoLeoLoss

        self.cfg = cfg
        # The 518 positional grid is part of the released state dict. Meta's
        # forward interpolates it at 224/98 without rewriting the checkpoint.
        factory = getattr(backbones, {'vit_small': 'dinov2_vits14', 'vit_large': 'dinov2_vitl14'}[cfg.student.arch])
        backbone = factory(
            pretrained=False, drop_path_rate=cfg.student.drop_path_rate,
            drop_path_uniform=cfg.student.drop_path_uniform,
            qkv_bias=cfg.student.qkv_bias, proj_bias=cfg.student.proj_bias,
            ffn_bias=cfg.student.ffn_bias, interpolate_offset=cfg.student.interpolate_offset,
            interpolate_antialias=cfg.student.interpolate_antialias,
        )
        if checkpoint is None and not allow_random_init:
            raise ValueError("An explicit released pretrained backbone checkpoint is required")
        self.pretrained_verified = False
        if checkpoint is not None:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            backbone.load_state_dict(state, strict=True)
            self.pretrained_verified = all(torch.equal(v, state[k]) for k, v in backbone.state_dict().items())
            if not self.pretrained_verified:
                raise RuntimeError("Pretrained tensor equality check failed")
        def make_head():
            return DINOHead(
                in_dim=backbone.embed_dim, out_dim=cfg.dino.head_n_prototypes,
                hidden_dim=cfg.dino.head_hidden_dim, bottleneck_dim=cfg.dino.head_bottleneck_dim,
                nlayers=cfg.dino.head_nlayers,
            )
        head = make_head()
        self.student = nn.ModuleDict({"backbone": backbone, "dino_head": head})
        # weight_norm creates non-leaf weight views, so build the head afresh.
        self.teacher = nn.ModuleDict({"backbone": deepcopy(backbone), "dino_head": make_head()})
        self.teacher.load_state_dict(self.student.state_dict(), strict=True)
        self.teacher.requires_grad_(False)
        # Optional ablation: only transformer blocks, not embeddings/tokens/norm/head.
        frozen = cfg.student.get("freeze_first_blocks", 0)
        if type(frozen) is not int or not 0 <= frozen <= len(backbone.blocks):
            raise ValueError("freeze_first_blocks must be an integer within the backbone depth")
        for block in self.student.backbone.blocks[:frozen]:
            block.requires_grad_(False)
        self.dino_loss = DINOLoss(cfg.dino.head_n_prototypes)
        self.ibot_patch_loss = iBOTPatchLoss(cfg.dino.head_n_prototypes)
        self.koleo_loss = KoLeoLoss()
        self.train()

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    def forward(self, batch, teacher_temp, *, precision="fp32", diagnostics=None,
                non_blocking=True, transfer_events=None, center_sink=None):
        device = next(self.student.parameters()).device
        if transfer_events is not None:
            transfer_events[0].record()
        images = {k: v.to(device, non_blocking=non_blocking) if isinstance(v, torch.Tensor) else v
                  for k, v in batch.items()}
        if transfer_events is not None:
            transfer_events[1].record()
        global_crops, local_crops = images["collated_global_crops"], images["collated_local_crops"]
        masks, indices = images["collated_masks"], images["mask_indices_list"]
        n_masked = indices.numel()
        if n_masked == 0:
            raise ValueError("No masked patches; cannot verify iBOT in this pilot batch")
        n_local = self.cfg.crops.local_crops_number
        if global_crops.shape[0] % 2 or local_crops.shape[0] != global_crops.shape[0] // 2 * n_local:
            raise ValueError("Expected two view-major global crops and configured local crops")
        if precision not in {"fp32", "bf16"} or (precision == "bf16" and device.type != "cuda"):
            raise ValueError("Use fp32 for CPU checks, bf16 or fp32 for CUDA")
        amp = torch.autocast("cuda", dtype=torch.bfloat16) if precision == "bf16" else nullcontext()
        with amp:
            with torch.no_grad():
                teacher_out = self.teacher.backbone(global_crops, is_training=True)
                # Exactly Meta's global A/B swap; patch targets stay in original order.
                cls_a, cls_b = teacher_out["x_norm_clstoken"].chunk(2)
                teacher_cls = torch.cat((cls_b, cls_a))
                teacher_patches = teacher_out["x_norm_patchtokens"].flatten(0, 1)[indices]
                teacher_logits = self.teacher.dino_head(torch.cat((teacher_cls, teacher_patches)))
                teacher_cls_logits, teacher_patch_logits = teacher_logits.split((len(teacher_cls), n_masked))
            # Single tensors invoke Meta's own non-nested attention path. The
            # same original backbone is used for masked globals and unmasked locals.
            global_out = self.student.backbone(global_crops, masks=masks, is_training=True)
            local_out = self.student.backbone(local_crops, is_training=True)
            student_cls = global_out["x_norm_clstoken"]
            student_patches = global_out["x_norm_patchtokens"].flatten(0, 1)[indices]
            lengths = (len(local_crops), len(global_crops), n_masked)
            logits = self.student.dino_head(torch.cat((local_out["x_norm_clstoken"], student_cls, student_patches)))
            local_logits, global_logits, patch_logits = logits.split(lengths)

        # FP32 loss inputs retain upstream loss functions, centers and delayed
        # center updates; AMP is limited to the backbone and projection head.
        with torch.no_grad():
            teacher_probs = self.dino_loss.softmax_center_teacher(teacher_cls_logits.float(), teacher_temp).view(2, -1, logits.shape[-1])
            if center_sink is None:
                self.dino_loss.update_center(teacher_cls_logits.float())
            patch_targets = teacher_patch_logits.float().unsqueeze(0)
            teacher_patch_probs = self.ibot_patch_loss.softmax_center_teacher(patch_targets, teacher_temp).squeeze(0)
            if center_sink is None:
                self.ibot_patch_loss.update_center(patch_targets)
            else:
                center_sink.add(teacher_cls_logits.float(), teacher_patch_logits.float())

        denominator = 2 + 2 * n_local
        losses = {
            "dino_local_crops_loss": self.dino_loss(local_logits.float().chunk(n_local), teacher_probs) / denominator,
            "dino_global_crops_loss": self.dino_loss([global_logits.float()], [teacher_probs.flatten(0, 1)]) * 2 / denominator,
            "koleo_loss": self.cfg.dino.koleo_loss_weight * sum(self.koleo_loss(x.float()) for x in student_cls.chunk(2)),
            "ibot_loss": self.ibot_patch_loss.forward_masked(
                patch_logits.float(), teacher_patch_probs, student_masks_flat=masks,
                n_masked_patches=n_masked, masks_weight=images["masks_weight"],
            ),
        }
        total = (self.cfg.dino.loss_weight * (losses["dino_local_crops_loss"] + losses["dino_global_crops_loss"])
                 + losses["koleo_loss"] + self.cfg.ibot.loss_weight * losses["ibot_loss"])
        if diagnostics is not None:
            with torch.no_grad():
                features = torch.nn.functional.normalize(student_cls.float(), dim=-1)
                count = len(features)
                diagnostics["cls_feature_std"] = features.std(dim=0, unbiased=False).mean()
                diagnostics["cls_pair_cosine"] = ((features @ features.T).sum() - count) / (count * (count - 1))
                diagnostics["teacher_entropy"] = -(teacher_probs * teacher_probs.clamp_min(1e-30).log()).sum(-1).mean()
                probabilities = (global_logits.float() / self.dino_loss.student_temp).softmax(-1)
                diagnostics['student_entropy'] = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(-1).mean()
                diagnostics['feature_variance'] = student_cls.float().var(dim=0, unbiased=False).mean()
        return total, losses

    @torch.no_grad()
    def update_teacher(self, momentum):
        student = dict(self.student.named_parameters())
        teacher = dict(self.teacher.named_parameters())
        if student.keys() != teacher.keys():
            raise RuntimeError("Student/teacher parameter mismatch")
        for name, target in teacher.items():
            # Even an EMA of equal FP32 tensors can round differently. Frozen
            # parameters retain the exact released values in BOTH networks.
            if student[name].requires_grad:
                target.mul_(momentum).add_(student[name], alpha=1 - momentum)

    def parameter_groups(self):
        from dinov2.utils.param_groups import get_params_groups_with_decay, fuse_params_groups
        groups = []
        for module in self.student.values():
            groups.extend(fuse_params_groups(get_params_groups_with_decay(
                module, lr_decay_rate=self.cfg.optim.layerwise_decay,
                patch_embed_lr_mult=self.cfg.optim.patch_embed_lr_mult,
            )))
        return groups


def schedules(cfg):
    from dinov2.utils.utils import CosineScheduler
    length = cfg.train.OFFICIAL_EPOCH_LENGTH
    total = cfg.optim.epochs * length
    common = dict(base_value=cfg.optim.lr, final_value=cfg.optim.min_lr, total_iters=total,
                  warmup_iters=cfg.optim.warmup_epochs * length, start_warmup_value=0)
    lr = CosineScheduler(**common)
    last = CosineScheduler(**common)
    last.schedule[:cfg.optim.freeze_last_layer_epochs * length] = 0
    temp_steps = cfg.teacher.warmup_teacher_temp_epochs * length
    return {
        "lr": lr, "last_layer_lr": last,
        "weight_decay": CosineScheduler(cfg.optim.weight_decay, cfg.optim.weight_decay_end, total),
        "momentum": CosineScheduler(cfg.teacher.momentum_teacher, cfg.teacher.final_momentum_teacher, total),
        "teacher_temp": CosineScheduler(cfg.teacher.teacher_temp, cfg.teacher.teacher_temp, temp_steps,
                                        warmup_iters=temp_steps, start_warmup_value=cfg.teacher.warmup_teacher_temp),
    }


def train_step(model, optimizer, batch, step_schedules, iteration, *, precision="fp32", diagnostics=False,
               defer_metrics=False, non_blocking=True, transfer_events=None):
    # Benchmark-only mode: keep scalars on device and validate them at a CUDA
    # measurement boundary. Normal training retains its immediate finite checks.
    if defer_metrics and not diagnostics:
        raise ValueError("Deferred finite checks require diagnostics=True")
    values = {k: float(v[iteration]) for k, v in step_schedules.items()}
    for group in optimizer.param_groups:
        group["weight_decay"] = values["weight_decay"] * group["wd_multiplier"]
        group["lr"] = values["last_layer_lr" if group["is_last_layer"] else "lr"] * group["lr_multiplier"]
    optimizer.zero_grad(set_to_none=True)
    monitoring = {} if diagnostics else None
    kwargs = {"diagnostics": monitoring} if diagnostics else {}
    if not non_blocking:
        kwargs['non_blocking'] = False
    if transfer_events is not None:
        kwargs['transfer_events'] = transfer_events
    total, losses = model(batch, values["teacher_temp"], precision=precision, **kwargs)
    if not defer_metrics and not torch.isfinite(total):
        raise FloatingPointError("Nonfinite loss before backward")
    total.backward()
    # Official clipping is per submodule, not over the combined model.
    norms = []
    for name, module in model.student.items():
        norm = torch.nn.utils.clip_grad_norm_(
            module.parameters(), model.cfg.optim.clip_grad or float("inf"), error_if_nonfinite=not defer_metrics
        )
        if not defer_metrics and not torch.isfinite(norm):
            raise FloatingPointError("Nonfinite gradient")
        if diagnostics:
            monitoring[f"gradient_norm_{name}"] = norm.detach()
            norms.append(norm.detach())
    optimizer.step()
    model.update_teacher(values["momentum"])
    if diagnostics:
        monitoring["gradient_norm"] = torch.stack(norms).norm()
        tensors = {"total_loss": total.detach(), **{k: v.detach() for k, v in losses.items()}, **monitoring}
        if defer_metrics:
            return {"iteration": iteration, **values, **tensors}
        scalars = torch.stack(list(tensors.values())).float().cpu().tolist()
        return {"iteration": iteration, **values, **dict(zip(tensors, scalars))}
    return {"iteration": iteration, **values, "total_loss": float(total.detach()),
            **{k: float(v.detach()) for k, v in losses.items()}}
