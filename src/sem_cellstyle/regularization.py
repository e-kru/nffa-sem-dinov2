"""Optional original-backbone anchors, independent of Meta's EMA teacher/losses."""
from contextlib import contextmanager, nullcontext
from copy import deepcopy
import math

import torch
from torch import nn
from torch.nn import functional as F


DEFAULTS = dict(l2_sp=False, feature_distillation=False, lambda_sp=None, lambda_feat=None,
                lambda_cls=1., lambda_patch=1., calibration_receipt=None, calibration_sha256=None)


def options(cfg, *, resolved=False):
    values = {**DEFAULTS, **dict(cfg.get('regularization', {}))}
    if set(values)-set(DEFAULTS):
        raise ValueError('Unknown regularization options: '+str(set(values)-set(DEFAULTS)))
    for name in ('l2_sp', 'feature_distillation'):
        if type(values[name]) is not bool:
            raise ValueError(name+' must be boolean')
    for name in ('lambda_sp', 'lambda_feat', 'lambda_cls', 'lambda_patch'):
        value = values[name]
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int,float))
                                  or not math.isfinite(value) or value < 0):
            raise ValueError(name+' must be finite and nonnegative')
    for flag, coefficient in (('l2_sp','lambda_sp'),('feature_distillation','lambda_feat')):
        if values[flag] and resolved and (values[coefficient] is None or values[coefficient] <= 0):
            raise ValueError('Calibrate and freeze a positive '+coefficient+' before initializing an enabled arm')
    if values['feature_distillation'] and (values['lambda_cls'] is None or values['lambda_patch'] is None
                                           or values['lambda_cls']+values['lambda_patch'] <= 0):
        raise ValueError('Feature distillation needs a positive CLS or patch weight')
    return values


@torch.no_grad()
def drift_by_block(pairs):
    """Relative L2 norm of parameter displacement, grouped by ViT block."""
    squares, originals = {}, {}
    for name, value, original in pairs:
        key = 'block_'+name.split('.')[1].zfill(2) if name.startswith('blocks.') else 'embeddings_and_norm'
        delta = (value.detach().float()-original.detach().float()).square().sum()
        norm = original.detach().float().square().sum()
        squares[key] = squares.get(key,0.)+delta
        originals[key] = originals.get(key,0.)+norm
    return {key: (value / originals[key].clamp_min(1e-30)).sqrt() for key,value in squares.items()}


class BackboneRegularization(nn.Module):
    """One immutable set of weights; a full encoder exists ONLY for distillation."""
    def __init__(self, backbone, cfg):
        super().__init__()
        self.settings = options(cfg, resolved=True)
        self.names = tuple(name for name,_ in backbone.named_parameters())
        self.parameter_count = sum(p.numel() for p in backbone.parameters())
        if self.settings['feature_distillation']:
            # Called before activation-checkpoint wrapping; deepcopy consumes no RNG.
            self.reference = deepcopy(backbone).requires_grad_(False).eval()
        else:
            self.reference = None
            for i,(_,p) in enumerate(backbone.named_parameters()):
                self.register_buffer(f'anchor_{i}',p.detach().clone())
        self.train()

    def train(self, mode=True):
        super().train(mode)
        if self.reference is not None:
            self.reference.eval()
        return self

    def originals(self):
        if self.reference is not None:
            return dict(self.reference.named_parameters())
        return {name:getattr(self,f'anchor_{i}') for i,name in enumerate(self.names)}

    def pairs(self, backbone):
        original = self.originals()
        parameters = dict(backbone.named_parameters())
        if parameters.keys() != original.keys():
            raise ValueError('Original/adapted backbone parameter identities differ')
        return ((name,value,original[name]) for name,value in parameters.items())

    @torch.no_grad()
    def sp_loss(self, backbone):
        return sum((value.float()-original.float()).square().sum()
                   for _,value,original in self.pairs(backbone))/(2*self.parameter_count)

    @torch.no_grad()
    def apply_sp(self, backbone, coefficient):
        """Exact analytic gradient, streamed per tensor to avoid a second ViT graph.

        d[sum ||theta-theta0||²/(2*N)]/dtheta = (theta-theta0)/N.
        Called once per optimizer update, outside the accumulation loop.
        """
        loss = None
        for _,p,original in self.pairs(backbone):
            delta = p.detach().float()-original.float()
            term = delta.square().sum()
            loss = term if loss is None else loss+term
            if coefficient:
                if p.grad is None:
                    p.grad = delta.mul(coefficient/self.parameter_count)
                else:
                    p.grad.add_(delta,alpha=coefficient/self.parameter_count)
        return loss/(2*self.parameter_count)

    @contextmanager
    def feature_loss(self, backbone, batch, *, precision, non_blocking=True):
        """Same already augmented global tensors, both unmasked, aligned by token.

        Eval mode only for this extra branch removes DropPath noise from the
        anchoring target. Keep it through backward/checkpoint recomputation;
        SSL's original train mode is restored even on failure. No model RNG used.
        """
        if self.reference is None:
            raise ValueError('Feature reference is disabled')
        device = next(backbone.parameters()).device
        views = batch['collated_global_crops'].to(device,non_blocking=non_blocking)
        training = backbone.training
        backbone.eval()
        try:
            amp = torch.autocast('cuda',dtype=torch.bfloat16) if precision=='bf16' else nullcontext()
            with amp:
                with torch.no_grad():
                    target = self.reference(views,is_training=True)
                adapted = backbone(views,is_training=True) # deliberately NO masks
            cls = F.cosine_similarity(adapted['x_norm_clstoken'].float(),target['x_norm_clstoken'].float(),dim=-1).clamp(-1,1).mean()
            patches = F.cosine_similarity(adapted['x_norm_patchtokens'].float(),target['x_norm_patchtokens'].float(),dim=-1).clamp(-1,1).mean()
            raw = self.settings['lambda_cls']*(1-cls)+self.settings['lambda_patch']*(1-patches)
            yield raw,dict(feature_cls_loss=1-cls,feature_patch_loss=1-patches,
                           feature_cls_similarity=cls,feature_patch_similarity=patches)
        finally:
            backbone.train(training)

    @torch.no_grad()
    def verify_original(self, state):
        original = self.originals()
        if any(not torch.equal(value.detach().cpu(),state[name]) for name,value in original.items()):
            raise ValueError('Immutable original-backbone reference differs from initialization')
        if self.reference is not None and (self.reference.training or any(p.requires_grad for p in self.reference.parameters())):
            raise ValueError('Feature reference must stay frozen in eval mode')


def attach(model):
    settings = options(model.cfg,resolved=True)
    if settings['l2_sp'] or settings['feature_distillation']:
        model.regularization = BackboneRegularization(model.student.backbone,model.cfg)


class GradientProbe:
    """Disposable calibration only: separate accumulated raw backbone gradients.

    With apply_regularizers=False the optimizer trajectory is exactly SSL-only.
    Each feature gradient buffer costs one backbone gradient, never an SSL graph.
    """
    def __init__(self, apply_regularizers=False):
        self.apply_regularizers = apply_regularizers
        self.feature_gradients = None

    def feature(self, raw, backbone, accumulation):
        params = tuple(backbone.parameters())
        gradients = torch.autograd.grad(raw/accumulation,params,allow_unused=True)
        if self.feature_gradients is None:
            self.feature_gradients = [None if g is None else g.detach() for g in gradients]
        else:
            for i,gradient in enumerate(gradients):
                if gradient is not None:
                    if self.feature_gradients[i] is None:
                        self.feature_gradients[i]=gradient.detach()
                    else:
                        self.feature_gradients[i].add_(gradient.detach())

    @torch.no_grad()
    def finish(self, regularizer, backbone):
        parameters = tuple(backbone.parameters())
        zero = next(backbone.parameters()).new_zeros(())
        ssl2 = sum((p.grad.float().square().sum() for p in parameters if p.grad is not None),zero)
        feat2 = sum((g.float().square().sum() for g in self.feature_gradients if g is not None),zero)
        sp2 = sum(((p.float()-reference.float()).square().sum() for _,p,reference in regularizer.pairs(backbone)),zero)/regularizer.parameter_count**2
        numbers=torch.stack((ssl2.sqrt(),sp2.sqrt(),feat2.sqrt())).cpu().tolist()
        if self.apply_regularizers:
            coefficient = regularizer.settings['lambda_feat']
            for p,g in zip(parameters,self.feature_gradients):
                if g is not None:
                    if p.grad is None: p.grad=g.mul(coefficient)
                    else: p.grad.add_(g,alpha=coefficient)
        return dict(calibration_ssl_gradient_norm=numbers[0],calibration_sp_gradient_norm=numbers[1],
                    calibration_feature_gradient_norm=numbers[2])
