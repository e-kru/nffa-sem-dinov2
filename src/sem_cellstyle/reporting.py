"""Large-to-Large validation comparison; historical Small stays a separate link."""
import json

from src.ssl.checkpoints import validate_snapshot
from src.ssl.experiment import digest
from src.ssl.reporting import result_bundle, render_report


KEYS = [f'nffa_{split}_{head}' for split in ('A','B') for head in ('logreg','mlp')]+['aachen_linear','aachen_cnn8']


def collect(cfg, run, evaluation_root):
    metadata = json.loads((run/'run.json').read_text())
    budget = metadata['config']['experiment']
    splits = list(cfg.evaluation.get('classification_splits', ['A','B']))
    selected_keys = [f'nffa_{split}_{head}' for split in splits for head in ('logreg','mlp')]+['aachen_linear','aachen_cnn8']
    rows, missing, protocols, ssl_contracts = [], [], {}, set()
    for point in cfg.budget.checkpoints_at_passes:
        step = budget['pass_to_step'][str(point)]
        folder = run/'checkpoints'/f'step_{step:06d}'
        full = point in cfg.evaluation.full_validation_passes
        expected = selected_keys if full else [*[f'nffa_{s}_logreg' for s in splits], 'aachen_linear']
        row = dict(presentation_passes=point,step=step,teacher_sha256=None,
                   milestone_available=folder.is_dir(),evaluation_scope='full' if full else 'quick',
                   metrics=dict.fromkeys(KEYS),deliberately_omitted=[key for key in KEYS if key not in expected])
        if not folder.is_dir():
            missing.append(f'Predeclared pass {point}: checkpoint not available')
            rows.append(row); continue
        snapshot=validate_snapshot(folder); ssl_contracts.add(snapshot['contract_sha256'])
        sha=snapshot['files']['teacher_backbone.pth']; row['teacher_sha256']=sha
        fingerprints=set()
        for task in ('classification','segmentation'):
            results, received, gaps, weights=result_bundle(evaluation_root/f'dinov2_vitl14_{sha[:16]}',sha,task,
                                                           classification_splits=splits,
                                                           head_profile=cfg.evaluation.get('head_profile','legacy'))
            fingerprints.update(weights)
            for key in expected:
                if key in results:
                    row['metrics'][key]=results[key]
                    if results[key] is None:
                        missing.append(f'pass {point}: {key}')
                if key in received:
                    previous=protocols.setdefault(key,received[key])
                    if previous!=received[key]:
                        raise ValueError('Downstream protocol changed across Large checkpoints: '+key)
        if len(fingerprints)>1:
            raise ValueError('Classification/segmentation backbone tensors differ')
        rows.append(row)
    if len(ssl_contracts)>1:
        raise ValueError('SSL contract changed across milestones')
    return dict(partition='val',test_evaluated=False,rows=rows,missing=missing,complete=not missing,
                protocol_sha256=protocols,primary_reference='original_meta_dinov2_vitl14_no_registers_pass_0',
                historical_small_reference='outputs/sem_ssl_reports/nffa_5000step_workers_v1/comparison.md',
                historical_small_scores_merged=False,seed_count=1,classification_splits=splits,
                head_profile=cfg.evaluation.get('head_profile','legacy'),
                recommendation='Review downstream validation and feature diagnostics against original Large. No selection from SSL loss alone.',
                causal_claim='Combined practical adaptation recipe; individual causal contributions are not identified.')


def write(cfg,run,evaluation_root,destination):
    result=collect(cfg,run,evaluation_root)
    folder=destination/digest(result)[:16]
    if not folder.exists():
        render_report(result,run,folder)
        with (folder/'comparison.md').open('a') as stream:
            stream.write('\nPrimary reference: original ViT-L/14 at pass 0. Historical ViT-S is separate.\n')
            stream.write('\nNFFA validation splits: '+', '.join(result['classification_splits'])+'.\n')
            stream.write('\nDownstream head profile: '+result['head_profile']+'.\n')
            stream.write('\nPass mapping: '+json.dumps({r['presentation_passes']:r['step'] for r in result['rows']})+'\n')
            stream.write('\nPasses 1/5 intentionally omit MLP and CNN8. Future checkpoints remain missing.\n')
            stream.write('\n'+result['recommendation']+'\n\n'+result['causal_claim']+'\n')
    return folder/'comparison.md'
