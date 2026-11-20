"""Accumulation orchestration around the existing adapter and original Meta losses."""
import functools
import time

import torch
from torch.utils.checkpoint import checkpoint

from .regularization import drift_by_block


def checkpoint_student(model):
    """Wrap bound forwards only; keep Meta parameter/state_dict names unchanged."""
    for block in model.student.backbone.blocks:
        original = block.forward
        @functools.wraps(original)
        def forward(*args, _original=original, **kwargs):
            return checkpoint(_original, *args, use_reentrant=False, preserve_rng_state=True, **kwargs)
        block.forward = forward


class Centers:
    """Constant-size FP32 sufficient statistics; never retain microbatch logits."""
    def __init__(self):
        self.sums, self.counts = {}, {}

    @torch.no_grad()
    def add(self, cls, patches):
        for key, logits in (('dino', cls), ('ibot', patches)):
            value = logits.detach().float().sum(0, keepdim=True)
            self.sums[key] = value if key not in self.sums else self.sums[key] + value
            self.counts[key] = self.counts.get(key, 0) + len(logits)

    @torch.no_grad()
    def apply(self, model):
        # Original Meta methods perform the EMA; the one-row inputs encode the
        # exact concatenated-logit mean (patch-count weighted, including masks).
        model.dino_loss.update_center(self.sums['dino'] / self.counts['dino'])
        model.ibot_patch_loss.update_center((self.sums['ibot'] / self.counts['ibot']).unsqueeze(0))
        model.dino_loss.apply_center_update()
        model.ibot_patch_loss.apply_center_update()


def parameter_groups(model):
    from dinov2.utils.param_groups import get_params_groups_with_decay, fuse_params_groups
    groups = []
    for name, module in model.student.items():
        entries = get_params_groups_with_decay(module, lr_decay_rate=model.cfg.optim.layerwise_decay,
                                              patch_embed_lr_mult=model.cfg.optim.patch_embed_lr_mult)
        if name == 'backbone':
            # Absolute peak applies to the deepest transformer block and norm.
            # Meta counts the output norm as one additional layer; shift the
            # block/embedding exponents by one, retaining adjacent-block decay.
            decay = model.cfg.optim.layerwise_decay
            for entry in entries:
                entry['lr_multiplier'] = min(1., entry['lr_multiplier'] / decay)
        for group in fuse_params_groups(entries):
            group['module'] = name
            groups.append(group)
    return groups


def accumulated_step(model, optimizer, batches, sched, iteration, *, precision='bf16', scalar_metrics=True, gradient_probe=None):
    """One update. Each microbatch keeps the old teacher, temperature and centers."""
    cfg = model.cfg
    count = cfg.runtime.accumulation_steps
    regularizer = getattr(model, 'regularization', None)
    values = {key: float(value[iteration]) for key, value in sched.items()}
    frozen_last = iteration < cfg.experiment.freeze_last_layer_steps
    for group in optimizer.param_groups:
        group['weight_decay'] = values['weight_decay'] * group['wd_multiplier']
        base = values['backbone_lr' if group['module'] == 'backbone' else 'head_lr']
        group['lr'] = (0. if group['is_last_layer'] and frozen_last else base) * group['lr_multiplier']
    optimizer.zero_grad(set_to_none=True)
    # Make restored/saved and uninterrupted execution identical at this boundary.
    model.dino_loss.apply_center_update()
    model.ibot_patch_loss.apply_center_update()
    centers, metrics, indices, draws, waiting = Centers(), {}, [], [], 0.
    for _ in range(count):
        begun = time.perf_counter()
        batch = next(batches)
        waiting += time.perf_counter() - begun
        diagnostic = {}
        total, losses = model(batch, values['teacher_temp'], precision=precision,
                              diagnostics=diagnostic, center_sink=centers, non_blocking=cfg.loader.non_blocking)
        if not torch.isfinite(total.detach()):
            raise FloatingPointError('Nonfinite loss; accumulated update not committed')
        (total / count).backward()
        for key, value in dict(total_loss=total, **losses, **diagnostic).items():
            detached = value.detach() / count
            metrics[key] = detached if key not in metrics else metrics[key] + detached
        if regularizer is not None and regularizer.settings['feature_distillation']:
            # SSL graph has already been released; only then build this graph.
            with regularizer.feature_loss(model.student.backbone,batch,precision=precision,
                                          non_blocking=cfg.loader.non_blocking) as (raw,similarities):
                if not torch.isfinite(raw.detach()):
                    raise FloatingPointError('Nonfinite feature regularization')
                if gradient_probe is not None:
                    gradient_probe.feature(raw,model.student.backbone,count)
                else:
                    (regularizer.settings['lambda_feat']*raw/count).backward()
                for key,value in dict(feature_raw_loss=raw,**similarities).items():
                    value=value.detach()/count
                    metrics[key]=metrics[key]+value if key in metrics else value
        indices.extend(batch.get('record_indices', [])); draws.extend(batch.get('draw_ids', []))
    calibration = {}
    if gradient_probe is not None:
        if regularizer is None:
            raise ValueError('Gradient calibration requires the original reference')
        calibration = gradient_probe.finish(regularizer,model.student.backbone)
    if regularizer is not None:
        metrics['ssl_loss']=metrics['total_loss']
        apply = gradient_probe is None or gradient_probe.apply_regularizers
        if regularizer.settings['l2_sp']:
            coefficient=regularizer.settings['lambda_sp'] if apply else 0.
            raw=regularizer.apply_sp(model.student.backbone,coefficient)
            if not torch.isfinite(raw):
                raise FloatingPointError('Nonfinite L2-SP loss')
            metrics['sp_raw_loss']=raw
            metrics['sp_weighted_loss']=coefficient*raw
            metrics['total_loss']=metrics['total_loss']+metrics['sp_weighted_loss']
        if regularizer.settings['feature_distillation']:
            coefficient=regularizer.settings['lambda_feat'] if apply else 0.
            metrics['feature_weighted_loss']=coefficient*metrics['feature_raw_loss']
            metrics['total_loss']=metrics['total_loss']+metrics['feature_weighted_loss']
    if regularizer is not None and not torch.isfinite(metrics['total_loss']):
        raise FloatingPointError('Nonfinite combined SSL/regularization objective before optimizer')
    # Match Meta: freeze output-layer WEIGHTS through zero LR. Gradients still
    # participate in head clipping and AdamW moments, including during warmup.
    for name, module in model.student.items():
        metrics['gradient_norm_' + name] = torch.nn.utils.clip_grad_norm_(
            module.parameters(), cfg.optim.clip_grad, error_if_nonfinite=True).detach()
    optimizer.step()
    centers.apply(model)
    model.update_teacher(values['momentum'])
    if regularizer is not None:
        metrics.update({'parameter_drift_'+key:value for key,value in
                        drift_by_block(
                            regularizer.pairs(model.student.backbone)).items()})
    if scalar_metrics:
        numbers = torch.stack(list(metrics.values())).float().cpu().tolist()
        metrics = dict(zip(metrics, numbers))
    return dict(step=iteration+1, **values, **metrics, **calibration, dataloader_wait_seconds=waiting,
                record_indices=indices, draw_ids=draws,
                original_presentations=(iteration+1)*cfg.runtime.effective_batch,
                group_lrs=[group['lr'] for group in optimizer.param_groups])
