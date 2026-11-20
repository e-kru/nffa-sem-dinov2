"""Small controlled-study helpers: four toggles, one training-only calibration."""
import gc
import json
import math
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import torch

from src.ssl.checkpoints import atomic_json
from src.ssl.data import sha256
from src.ssl.experiment import digest
from src.ssl.upstream import ROOT
from .recipe import configuration, contract, resolve, schedules
from .regularization import options, GradientProbe

ARMS = {'A':(False,False),'B':(True,False),'C':(False,True),'D':(True,True)}
COEFFICIENTS = ('lambda_sp','lambda_feat','lambda_cls','lambda_patch')


def arm_templates():
    result={arm:configuration(ROOT/f'configs/sem_ssl/sem_cellstyle_regularization_{arm}.yaml') for arm in ARMS}
    canonical=None
    shared_settings=None
    for arm,cfg in result.items():
        item=OmegaConf.to_container(cfg,resolve=True)
        settings=item.pop('regularization'); item['runtime'].pop('output')
        if (settings['l2_sp'],settings['feature_distillation']) != ARMS[arm]:
            raise ValueError('Incorrect regularization toggles in arm '+arm)
        same={k:v for k,v in settings.items() if k not in ('l2_sp','feature_distillation')}
        if shared_settings is None: shared_settings=same
        elif shared_settings!=same: raise ValueError('Coefficients/settings differ between controlled arm templates')
        if canonical is None: canonical=item
        elif item!=canonical: raise ValueError('Non-regularization hyperparameters differ between arm templates')
    return result


def arm_config(base, arm, output, coefficients=None, receipt=None):
    cfg=OmegaConf.create(OmegaConf.to_container(base,resolve=True))
    cfg.regularization=OmegaConf.to_container(arm_templates()[arm].regularization)
    for key in ('lambda_cls','lambda_patch'):
        cfg.regularization[key]=options(base)[key]
    cfg.runtime.output=str(Path(output).resolve())
    if coefficients is not None:
        for key in COEFFICIENTS: cfg.regularization[key]=coefficients[key]
    if receipt is not None:
        cfg.regularization.calibration_receipt=str(Path(receipt).resolve())
        cfg.regularization.calibration_sha256=sha256(Path(receipt))
    return cfg


def common_identity(cfg, data, initialization, *, include_batch=True):
    value=contract(cfg,data,initialization,'cuda')
    c=value['configuration']; c.pop('regularization',None); c['runtime'].pop('output')
    value['feature_token_weights']={key:options(cfg)[key] for key in ('lambda_cls','lambda_patch')}
    if not include_batch:
        c['runtime'].pop('physical_batch',None); c['runtime'].pop('accumulation_steps',None)
        c['train'].pop('batch_size_per_gpu',None)
    return digest(value)


def require_calibration(cfg, data, initialization):
    settings=options(cfg,resolved=True)
    if not settings['l2_sp'] and not settings['feature_distillation'] and not settings['calibration_receipt']:
        return # existing unregularized pipeline remains usable as before
    if not settings['calibration_receipt'] or not settings['calibration_sha256']:
        raise ValueError('Regularization study needs its frozen shared training-only calibration receipt')
    path=Path(settings['calibration_receipt'])
    if sha256(path)!=settings['calibration_sha256']:
        raise ValueError('Calibration receipt changed')
    receipt=json.loads(path.read_text())
    if receipt.get('status')!='frozen' or receipt.get('common_identity')!=common_identity(cfg,data,initialization):
        raise ValueError('Calibration is not frozen or differs in dataset/batch/initialization/recipe/source')
    if any(settings[key]!=receipt['coefficients'][key] for key in COEFFICIENTS):
        raise ValueError('Coefficients differ from the shared frozen calibration')
    return receipt


def select_coefficients(rows, *, target=.10, minimum_median=.01, max_combined=.25, lambda_cls=1., lambda_patch=1., feature_noise_floor=0.):
    """Cap each weighted raw-gradient/SSL-gradient ratio, never tune by val scores."""
    if not 0 < minimum_median <= target or 2*target > max_combined or not rows:
        raise ValueError('Invalid or empty calibration protocol')
    ratios={name:[] for name in ('sp','feature')}
    for row in rows:
        ssl=row['calibration_ssl_gradient_norm']
        if not math.isfinite(ssl) or ssl<=0: raise ValueError('Invalid SSL gradient norm in calibration')
        for name in ratios:
            raw=row[f'calibration_{name}_gradient_norm']
            if name=='feature' and feature_noise_floor>0 and raw<=10*feature_noise_floor:
                raise ValueError('Feature gradient does not exceed original-model numerical noise by 10x')
            value=raw/ssl
            if not math.isfinite(value) or value<=1e-30:
                raise ValueError('No measurable '+name+' gradient; increase the shared calibration drift window')
            ratios[name].append(value)
    coefficients=dict(lambda_cls=lambda_cls,lambda_patch=lambda_patch)
    diagnostics={}
    for name,values in ratios.items():
        coefficient=target/max(values)
        weighted=[coefficient*v for v in values]
        if float(np.median(weighted)) < minimum_median:
            raise ValueError('Calibration gradient window is unstable/nontriviality check failed: '+name)
        coefficients['lambda_sp' if name=='sp' else 'lambda_feat']=coefficient
        diagnostics[name]=dict(max_weighted_gradient_ratio=max(weighted),median_weighted_gradient_ratio=float(np.median(weighted)))
    return coefficients,diagnostics


def check_combined(rows, coefficients, *, minimum_median=.01, maximum=.25, feature_noise_floor=0.):
    ratios={name:[coefficients['lambda_sp' if name=='sp' else 'lambda_feat']*r[f'calibration_{name}_gradient_norm']/r['calibration_ssl_gradient_norm']
                  for r in rows] for name in ('sp','feature')}
    if not rows or any(not math.isfinite(v) for values in ratios.values() for v in values):
        raise ValueError('Combined-arm validation has invalid gradient ratios')
    if any(row['calibration_feature_gradient_norm']<=10*feature_noise_floor for row in rows):
        raise ValueError('Combined-arm feature gradient is at the original-model numerical noise floor')
    # Triangle bound is deliberately conservative; cancellation cannot hide dominance.
    largest=max(sum(pair) for pair in zip(ratios['sp'],ratios['feature']))
    if largest>maximum or any(float(np.median(v))<minimum_median for v in ratios.values()):
        raise ValueError('Fixed coefficients fail joint non-dominance/nontriviality check; no arm is launched')
    return dict(maximum_combined_gradient_triangle_bound=largest,
                median_sp_gradient_ratio=float(np.median(ratios['sp'])),
                median_feature_gradient_ratio=float(np.median(ratios['feature'])))


def calibrate(base, dataset, collate, data, initialization, output, *, warmup=20, probes=10,
              device='cuda', target=.10, minimum=.01, maximum=.25):
    """Discard BOTH trajectories. Probe at the predefined end-of-LR-warmup offset.

    At theta0 L2-SP is identically zero, and early BF16 features may quantize
    tiny warmup updates away. A short peak-LR diagnostic trajectory creates a
    measurable displacement without pretending to have run five data passes.
    Every scientific arm still starts at optimizer/schedule index zero.
    """
    from .runner import initialize,loader
    from .training import accumulated_step
    if warmup<1 or probes<2:
        raise ValueError('Calibration needs positive drift warmup and >=2 gradient probes')
    output=Path(output)
    if output.exists(): raise FileExistsError(output)
    token_weights={key:options(base)[key] for key in ('lambda_cls','lambda_patch')}
    unit=dict(lambda_sp=1.,lambda_feat=1.,**token_weights)
    cfg=arm_config(base,'D',Path(base.runtime.output),unit)
    offset=cfg.experiment.warmup_steps
    total=warmup+probes
    if offset+total>cfg.experiment.schedule_total_steps:
        raise ValueError('Calibration window outside the original full schedule')
    def trajectory(coefficients, apply):
        for key,value in coefficients.items(): cfg.regularization[key]=value
        model,optimizer=initialize(cfg,initialization,device)
        stream=iter(loader(cfg,dataset,collate,device,0,total)); sched=schedules(cfg)
        rows=[]; noise=0.
        for index in range(total):
            row=accumulated_step(model,optimizer,stream,sched,offset+index,
                                 precision='bf16' if str(device).startswith('cuda') else 'fp32',
                                 gradient_probe=GradientProbe(apply_regularizers=apply))
            if index==0: noise=row['calibration_feature_gradient_norm']
            if index>=warmup:
                keys=['calibration_ssl_gradient_norm','calibration_sp_gradient_norm','calibration_feature_gradient_norm',
                      'ssl_loss','sp_raw_loss','feature_raw_loss','feature_cls_similarity','feature_patch_similarity']
                rows.append({key:row[key] for key in keys}|dict(calibration_update=index+1,schedule_index=offset+index,
                             consumed_originals=(index+1)*cfg.runtime.effective_batch,draw_ids=row['draw_ids']))
        del model,optimizer,stream
        gc.collect()
        if str(device).startswith('cuda'): torch.cuda.empty_cache()
        return rows,noise
    raw,noise=trajectory(unit,False)
    coefficients,diagnostics=select_coefficients(raw,target=target,minimum_median=minimum,max_combined=maximum,feature_noise_floor=noise,**token_weights)
    validation,validation_noise=trajectory(coefficients,True)
    combined=check_combined(validation,coefficients,minimum_median=minimum,maximum=maximum,feature_noise_floor=validation_noise)
    result=dict(schema='sem-cellstyle-regularization-calibration-v1',status='frozen',
                common_identity=common_identity(base,data,initialization),coefficients=coefficients,
                training_data_only=True,validation_or_test_accessed=False,weights_promoted=False,
                warmup_updates=warmup,gradient_probe_updates=probes,schedule_offset=offset,
                schedule_offset_policy='end_of_original_lr_warmup; disposable diagnostic only',
                original_model_feature_gradient_noise_floor=noise,combined_original_model_noise_floor=validation_noise,
                feature_signal_minimum_multiple=10, target_per_regularizer_gradient_ratio=target,minimum_median_gradient_ratio=minimum,
                maximum_combined_gradient_bound=maximum,selection_diagnostics=diagnostics,
                combined_validation=combined,unregularized_probe_rows=raw,combined_probe_rows=validation)
    atomic_json(output,result)
    return result
