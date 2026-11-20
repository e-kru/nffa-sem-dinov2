"""Both validation references, student parameter drift and frozen-Teacher diagnostics."""
import csv
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from omegaconf import OmegaConf

from src.ssl.checkpoints import atomic_json
from src.ssl.data import sha256
from src.ssl.experiment import digest
from src.ssl.reporting import result_bundle
from .recipe import configuration
from .regularization import drift_by_block
from .regularization_study import ARMS,COEFFICIENTS

KEYS=[f'nffa_{s}_{h}' for s in ('A','B') for h in ('logreg','mlp')]+['aachen_linear','aachen_cnn8']


def scalar_metrics(metrics):
    """Comparable fractions, never mix fixed and validation-selected thresholds."""
    values={}
    for key,entry in metrics.items():
        if entry is None: continue
        if key.startswith('nffa_'): values[key+'.macro_f1']=entry['macro_f1']
        else:
            for protocol in ('fixed','selected'):
                for metric in ('ma_iou','miou','ma_precision','ma_recall'):
                    values[f'{key}.{protocol}.{metric}']=entry[protocol][metric]
    return values


def differences(value, original, unregularized):
    return dict(value=value,delta_vs_original=None if value is None or original is None else value-original,
                delta_vs_A=None if value is None or unregularized is None else value-unregularized)


def collect(root):
    calibration=json.loads((root/'calibration.json').read_text())
    reference_protocols,rows,missing={},[],[]
    reference_splits=None
    for arm in ARMS:
        cfg=configuration(root/f'configs/{arm}.yaml')
        splits=list(cfg.evaluation.get('classification_splits',['A','B']))
        if reference_splits is not None and splits!=reference_splits:
            raise ValueError('NFFA validation splits differ across arms')
        reference_splits=splits
        if any(cfg.regularization[key]!=calibration['coefficients'][key] for key in COEFFICIENTS):
            raise ValueError('Arm coefficients differ from the frozen shared calibration')
        run=root/'runs'/arm
        metadata=json.loads((run/'run.json').read_text()) if (run/'run.json').exists() else None
        if metadata is None: missing.append('No training run '+arm)
        history={}
        if (run/'metrics.jsonl').exists():
            history={r['step']:r for line in (run/'metrics.jsonl').read_text().splitlines() if line.strip() for r in [json.loads(line)]}
        for point in cfg.budget.checkpoints_at_passes:
            step=cfg.experiment.pass_to_step[str(point)]
            folder=run/'checkpoints'/f'step_{step:06d}'
            row=dict(arm=arm,presentation_passes=point,step=step,metrics=dict.fromkeys(KEYS),scalar_metrics={},
                     training=history.get(step),student_drift_by_block=None,teacher_training_view_similarity=None)
            if not (folder/'complete.json').exists():
                missing.append(f'{arm}: pass {point} checkpoint missing');rows.append(row);continue
            snapshot=json.loads((folder/'complete.json').read_text())
            expected=digest(metadata['contract'])
            if snapshot['contract_sha256']!=expected:
                raise ValueError('Training contract changed across checkpoints in '+arm)
            sha=snapshot['files']['teacher_backbone.pth']
            base=root/'evaluation'/('original_large' if point==0 else arm)/f'dinov2_vitl14_{sha[:16]}'
            fingerprints=set()
            for task in ('classification','segmentation'):
                metrics,protocols,gaps,weights=result_bundle(base,sha,task,classification_splits=splits,
                                                             head_profile=cfg.evaluation.get('head_profile','legacy'))
                row['metrics'].update(metrics);fingerprints.update(weights);missing.extend(gaps)
                for key,value in protocols.items():
                    previous=reference_protocols.setdefault(key,value)
                    if previous!=value: raise ValueError('Validation protocol changed across arms/checkpoints: '+key)
            if len(fingerprints)>1: raise ValueError('Classification/segmentation backbones differ')
            row['scalar_metrics']=scalar_metrics(row['metrics'])
            # mmap loads only the accessed Student tensors, not AdamW/head arrays.
            if metadata and (folder/'training.pth').exists():
                state=torch.load(folder/'training.pth',map_location='cpu',weights_only=False,mmap=True)
                initialization=Path(metadata['config']['initialization']['checkpoint'])
                original=torch.load(initialization,map_location='cpu',weights_only=True,mmap=True)
                names=metadata['backbone_parameter_names']
                pairs=((name,state['model']['student.backbone.'+name],original[name]) for name in names)
                row['student_drift_by_block']={key:float(value) for key,value in drift_by_block(pairs).items()}
                del state,original
            rows.append(row)
    original=next((r['scalar_metrics'] for r in rows if r['arm']=='A' and r['presentation_passes']==0),{})
    a_by_pass={r['presentation_passes']:r['scalar_metrics'] for r in rows if r['arm']=='A'}
    for row in rows:
        row['comparison']={key:differences(row['scalar_metrics'].get(key),original.get(key),
                           a_by_pass.get(row['presentation_passes'],{}).get(key)) for key in set(original)|set(row['scalar_metrics'])}
    diagnostics=json.loads((root/'diagnostics.json').read_text()) if (root/'diagnostics.json').exists() else None
    if diagnostics:
        for row in rows:
            row['teacher_training_view_similarity']=next((r for r in diagnostics['results']
                if r['arm']==row['arm'] and r['presentation_passes']==row['presentation_passes']),None)
    runtime={}
    for arm in ARMS:
        path=root/'runs'/arm/'metrics.jsonl'
        logs=[json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
        runtime[arm]=dict(optimizer_update_wall_seconds=sum(r['wall_seconds'] for r in logs),
                         peak_allocated_vram_bytes=max((r['peak_vram_bytes'] for r in logs),default=None),
                         timing_excludes_checkpoint_and_downstream_io=True)
    return dict(schema='sem-cellstyle-regularization-comparison-v1',partition='val',test_evaluated=False,
                coefficients=calibration['coefficients'],calibration=calibration,rows=rows,missing=missing,
                all_predeclared_milestones_available=not missing,runtime=runtime,
                primary_reference='Original Meta DINOv2 ViT-L/14 at pass 0',secondary_reference='Unregularized run A at matching pass',
                historical_vits_results_modified=False,seed_count=1,classification_splits=splits,
                head_profile=cfg.evaluation.get('head_profile','legacy'),
                recommendation='Compare validation deltas against BOTH references; report task tradeoffs and confirm promising outcomes with more seeds. No SSL-loss-only selection.')


def write_report(root):
    result=collect(root)
    folder=root/'report'/digest(result)[:16]
    if folder.exists(): return folder/'comparison.md'
    folder.mkdir(parents=True)
    atomic_json(folder/'comparison.json',result)
    columns=['arm','passes','step','metric','value','delta_vs_original','delta_vs_A']
    with (folder/'comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        for row in result['rows']:
            for metric,entry in sorted(row['comparison'].items()):
                writer.writerow(dict(arm=row['arm'],passes=row['presentation_passes'],step=row['step'],metric=metric,**entry))
    lines=['# L2-SP / Feature Distillation: Large validation comparison','',
           'Fractions; deltas are absolute differences. Original Large and matching unregularized A are separate references.',
           'Threshold protocols fixed 0.5 and validation-selected are never pooled. Test not evaluated. One seed.','',
           'NFFA validation splits: '+', '.join(result['classification_splits'])+'.','',
           'Downstream head profile: '+result.get('head_profile','legacy')+'.','',
           'Frozen coefficients: `'+json.dumps(result['coefficients'],sort_keys=True)+'`.','',
           '| Arm | Pass | Metric | Value | Δ original L | Δ A, same pass |',
           '| --- | ---: | --- | ---: | ---: | ---: |']
    for row in result['rows']:
        for metric,item in sorted(row['comparison'].items()):
            if not (metric.endswith('macro_f1') or metric.endswith('ma_iou')):continue
            values=['—' if item[key] is None else f'{item[key]:.6f}' for key in ('value','delta_vs_original','delta_vs_A')]
            lines.append(f"| {row['arm']} | {row['presentation_passes']} | {metric} | "+' | '.join(values)+' |')
    lines+=['','## Training and original-feature diagnostics','',
            '| Arm | Pass | SSL loss | SP raw | Feature weighted | Mean block drift | Teacher CLS cosine | Teacher patch cosine |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in result['rows']:
        history=row['training'] or {}; diagnostic=row['teacher_training_view_similarity'] or {}
        blocks=[v for key,v in (row['student_drift_by_block'] or {}).items() if key.startswith('block_')]
        values=[history.get('ssl_loss',history.get('total_loss')),history.get('sp_raw_loss'),
                history.get('feature_weighted_loss'),sum(blocks)/len(blocks) if blocks else None,
                diagnostic.get('cls_similarity'),diagnostic.get('patch_similarity')]
        rendered=['—' if v is None else f'{v:.6g}' for v in values]
        lines.append(f"| {row['arm']} | {row['presentation_passes']} | "+' | '.join(rendered)+' |')
    lines+=['','## Runtime and memory','', '| Arm | Sum of update wall seconds | Peak allocated GiB |','| --- | ---: | ---: |']
    for arm,entry in result['runtime'].items():
        memory=entry['peak_allocated_vram_bytes']
        lines.append(f"| {arm} | {entry['optimizer_update_wall_seconds']:.2f} | {memory/2**30 if memory is not None else '—'} |")
    lines+=['','Checkpoint/downstream I/O is outside these update timings. Actual check throughput/reserved VRAM is in `checks/*/benchmark.json`.',
            '',result['recommendation'],'','Future milestones and missing evaluation results remain explicit:',
            '',*[f'- {item}' for item in result['missing']],
            '', 'Student drift is measured against original weights; training feature similarities refer to Student,',
            'while the separate training-image diagnostic compares exported frozen Teacher features.']
    (folder/'comparison.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,metric in zip(axes.flat,('nffa_A_logreg.macro_f1','nffa_B_mlp.macro_f1','aachen_linear.fixed.ma_iou','aachen_cnn8.selected.ma_iou')):
        for arm in ARMS:
            rows=[r for r in result['rows'] if r['arm']==arm]
            ax.plot([r['presentation_passes'] for r in rows],[r['scalar_metrics'].get(metric,float('nan')) for r in rows],'o-',label=arm)
        ax.set(xlabel='Original-presentation pass',ylabel=metric);ax.grid(alpha=.3);ax.legend()
    fig.tight_layout();fig.savefig(folder/'validation_vs_pass.png');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(16,4))
    for arm in ARMS:
        rows=[r for r in result['rows'] if r['arm']==arm]
        for axis,field in zip(axes[1:],('cls_similarity','patch_similarity')):
            axis.plot([r['presentation_passes'] for r in rows],[(r['teacher_training_view_similarity'] or {}).get(field,float('nan')) for r in rows],'o-',label=arm)
            axis.set(xlabel='Pass',ylabel='Frozen Teacher '+field)
        blocks=sorted({key for r in rows for key in (r['student_drift_by_block'] or {}) if key.startswith('block_')})
        color='C'+str(list(ARMS).index(arm))
        for block in blocks:
            axes[0].plot([r['presentation_passes'] for r in rows],[(r['student_drift_by_block'] or {}).get(block,float('nan')) for r in rows],alpha=.25,color=color)
        means=[sum((r['student_drift_by_block'] or {}).get(block,float('nan')) for block in blocks)/len(blocks) if blocks else float('nan') for r in rows]
        axes[0].plot([r['presentation_passes'] for r in rows],means,color=color,label=arm+' mean',linewidth=2)
    axes[0].set(xlabel='Pass',ylabel='Student drift: thin=blocks, bold=block mean')
    axes[0].legend()
    for ax in axes: ax.grid(alpha=.3)
    for ax in axes[1:]: ax.legend()
    fig.tight_layout();fig.savefig(folder/'drift_and_feature_similarity.png');plt.close(fig)
    return folder/'comparison.md'


def feature_diagnostics(root,base,dataset,collate,initialization,*,device='cuda'):
    """Separate post-training diagnostic; training views only, no changes to heads.

    One fixed temporary in-memory batch is shared across frozen exports, NOT a
    cache of augmented training views. This costs no extra forwards in arm A/B's
    training step and never influences coefficient calibration/model selection.
    """
    from torch.utils.data import DataLoader
    from src.ssl.stream import DrawDataset,DrawCollate
    from src.ssl.data import seed_worker
    from src.ssl.loading import loader_kwargs
    from .data import PresentationBatches
    from src.downstream.pinned_encoder import get_pinned_dinov2
    destination=root/'diagnostics.json'
    if destination.exists(): raise FileExistsError(destination)
    cfg=configuration(root/'configs/A.yaml')
    batch=next(iter(DataLoader(DrawDataset(dataset,42),batch_sampler=PresentationBatches(len(dataset),cfg.runtime.physical_batch,42,0,1),
        collate_fn=DrawCollate(collate,42),worker_init_fn=seed_worker,generator=torch.Generator().manual_seed(42),**loader_kwargs(cfg,device))))
    views=batch['collated_global_crops'].to(device)
    reference=get_pinned_dinov2(initialization,'dinov2_vitl14').to(device).eval()
    def forward(model):
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=str(device).startswith('cuda')):
            return model(views,is_training=True)
    target=forward(reference);results=[]
    for arm in ARMS:
        for point in cfg.budget.checkpoints_at_passes:
            step=cfg.experiment.pass_to_step[str(point)]
            folder=root/'runs'/arm/'checkpoints'/f'step_{step:06d}'
            if not (folder/'complete.json').exists(): continue
            meta=json.loads((folder/'complete.json').read_text());path=folder/'teacher_backbone.pth'
            if sha256(path)!=meta['files']['teacher_backbone.pth']: raise ValueError('Changed diagnostic export')
            model=get_pinned_dinov2(path,'dinov2_vitl14').to(device).eval()
            features=forward(model)
            with torch.inference_mode():
                similarities={name:float(F.cosine_similarity(features[key].float(),target[key].float(),dim=-1).clamp(-1,1).mean())
                    for name,key in (('cls_similarity','x_norm_clstoken'),('patch_similarity','x_norm_patchtokens'))}
            results.append(dict(arm=arm,presentation_passes=point,step=step,teacher_sha256=meta['files']['teacher_backbone.pth'],**similarities))
            del features,model
    atomic_json(destination,dict(training_data_only=True,validation_or_test_images_used=False,
        drawn_originals=cfg.runtime.physical_batch,draw_ids=batch['draw_ids'],record_indices=batch['record_indices'],results=results))
