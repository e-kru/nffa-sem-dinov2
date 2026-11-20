"""Regularization math and controlled-grid CPU fixtures; no real Large/GPU run."""
import copy
import json
from pathlib import Path
from unittest import mock

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from test_sem_cellstyle import fixture
from src.sem_cellstyle.regularization import BackboneRegularization, GradientProbe, options
from src.sem_cellstyle.runner import initialize,loader,execute
from src.sem_cellstyle.recipe import schedules,configuration
from src.sem_cellstyle.training import accumulated_step
from src.sem_cellstyle.regularization_study import (ARMS,arm_templates,arm_config,common_identity,
    select_coefficients,check_combined,require_calibration,calibrate)
from src.ssl.checkpoints import find_latest, rng_state,restore_rng
from src.ssl.data import sha256


def enable(cfg, sp=True, feature=True, **kwargs):
    cfg=copy.deepcopy(cfg)
    cfg.regularization=dict(l2_sp=sp,feature_distillation=feature,lambda_sp=2.,lambda_feat=.3,
                           lambda_cls=.7,lambda_patch=1.2,**kwargs)
    return cfg


def test_templates_differ_only_in_toggles_and_output():
    templates=arm_templates()
    assert len(templates)==4
    for arm,cfg in templates.items():
        settings=options(cfg)
        assert (settings['l2_sp'],settings['feature_distillation'])==ARMS[arm]
        assert settings['lambda_sp'] is settings['lambda_feat'] is None
        assert cfg.optim.backbone_lr==1e-5 and cfg.optim.head_lr==1e-4
    with pytest.raises(ValueError,match='Calibrate'):
        options(templates['D'],resolved=True)


def test_sp_formula_excludes_head_and_exact_gradient_once(fixture):
    cfg,_,_,_,init=fixture
    cfg=enable(cfg,feature=False)
    model,_=initialize(cfg,init,'cpu'); reg=model.regularization
    original={name:p.clone() for name,p in model.student.backbone.named_parameters()}
    with torch.no_grad():
        for p in model.student.backbone.parameters(): p.add_(.01)
        for p in model.student.dino_head.parameters(): p.add_(3.)
    n=sum(p.numel() for p in model.student.backbone.parameters())
    expected=sum((p-original[name]).square().sum() for name,p in model.student.backbone.named_parameters())/(2*n)
    expected_grads=torch.autograd.grad(2*expected,tuple(model.student.backbone.parameters()))
    actual=reg.apply_sp(model.student.backbone,2.)
    torch.testing.assert_close(actual,expected.detach())
    for p,gradient in zip(model.student.backbone.parameters(),expected_grads):
        torch.testing.assert_close(p.grad,gradient)
    assert all(p.grad is None for p in model.student.dino_head.parameters())
    assert reg.reference is None
    assert reg.parameter_count==n
    assert all(torch.equal(value,original[name]) for name,value in reg.originals().items())


def test_feature_views_unmasked_gradients_reference_eval_and_rng(fixture):
    cfg,data,collate,_,init=fixture
    cfg=enable(cfg,sp=False)
    model,_=initialize(cfg,init,'cpu'); reg=model.regularization
    with torch.no_grad(): model.student.backbone.blocks[1].attn.qkv.weight.add_(.04*torch.randn_like(model.student.backbone.blocks[1].attn.qkv.weight))
    batch=next(iter(loader(cfg,data,collate,'cpu',0,1)))
    assert batch['collated_masks'].any()
    ref_before={key:value.clone() for key,value in reg.reference.state_dict().items()}
    before=rng_state()
    with mock.patch.object(model.student.backbone,'forward',wraps=model.student.backbone.forward) as student, \
         mock.patch.object(reg.reference,'forward',wraps=reg.reference.forward) as reference:
        with reg.feature_loss(model.student.backbone,batch,precision='fp32') as (loss,metrics):
            assert not model.student.backbone.training and not reg.reference.training
            loss.backward()
        assert student.call_count==reference.call_count==1
        assert student.call_args.kwargs.get('masks') is None
        assert reference.call_args.kwargs.get('masks') is None
        assert torch.equal(student.call_args.args[0],reference.call_args.args[0])
    assert model.student.backbone.training
    assert torch.equal(before['torch'],rng_state()['torch'])
    assert metrics['feature_cls_similarity']<1 and metrics['feature_patch_similarity']<1
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.student.backbone.parameters())
    assert all(p.grad is None and not p.requires_grad for p in reg.reference.parameters())
    assert all(torch.equal(value,reg.reference.state_dict()[key]) for key,value in ref_before.items())
    model.train()
    assert not reg.reference.training


def test_identical_features_and_one_shared_anchor_for_combined(fixture):
    cfg,data,collate,_,init=fixture
    model,opt=initialize(enable(cfg),init,'cpu')
    reg=model.regularization
    assert not [name for name,_ in reg.named_buffers() if name.startswith('anchor_')]
    assert all(not p.requires_grad for p in reg.parameters())
    assert not any(id(p) in {id(q) for q in reg.parameters()} for g in opt.param_groups for p in g['params'])
    batch=next(iter(loader(cfg,data,collate,'cpu',0,1)))
    with reg.feature_loss(model.student.backbone,batch,precision='fp32') as (loss,_):
        assert abs(loss.item())<1e-6


def test_disabled_is_bitwise_existing_path_and_has_no_reference(fixture):
    cfg,data,collate,_,init=fixture
    a,ao=initialize(cfg,init,'cpu')
    before=rng_state()
    for index,batch in enumerate(loader(cfg,data,collate,'cpu',0,2)):
        pass # no model RNG touched by draw-scoped transforms
    restore_rng(before)
    stream=iter(loader(cfg,data,collate,'cpu',0,3))
    rows_a=[accumulated_step(a,ao,stream,schedules(cfg),i,precision='fp32') for i in range(3)]
    disabled=enable(cfg,sp=False,feature=False)
    with mock.patch('src.sem_cellstyle.regularization.BackboneRegularization',side_effect=AssertionError('extra reference')):
        b,bo=initialize(disabled,init,'cpu')
        assert not hasattr(b,'regularization')
        stream=iter(loader(disabled,data,collate,'cpu',0,3))
        rows_b=[accumulated_step(b,bo,stream,schedules(disabled),i,precision='fp32') for i in range(3)]
    assert all(torch.equal(v,b.state_dict()[key]) for key,v in a.state_dict().items())
    assert [row['total_loss'] for row in rows_a]==[row['total_loss'] for row in rows_b]
    for key,state in ao.state_dict()['state'].items():
        for name,value in state.items(): assert torch.equal(value,bo.state_dict()['state'][key][name])


def test_accumulation_sp_once_and_teacher_center_unchanged_by_feature_branch(fixture):
    cfg,data,collate,_,init=fixture
    cfg=enable(cfg)
    model,opt=initialize(cfg,init,'cpu')
    reference={key:value.clone() for key,value in model.regularization.state_dict().items()}
    with mock.patch.object(model.regularization,'apply_sp',wraps=model.regularization.apply_sp) as sp, \
         mock.patch.object(model,'update_teacher',wraps=model.update_teacher) as ema, \
         mock.patch.object(model.dino_loss,'update_center',wraps=model.dino_loss.update_center) as center, \
         mock.patch.object(opt,'step',wraps=opt.step) as step:
        row=accumulated_step(model,opt,iter(loader(cfg,data,collate,'cpu',2,3)),schedules(cfg),2,precision='fp32')
    assert sp.call_count==ema.call_count==center.call_count==step.call_count==1
    assert row['total_loss']==pytest.approx(row['ssl_loss']+row['sp_weighted_loss']+row['feature_weighted_loss'])
    assert all(torch.equal(value,model.regularization.state_dict()[key]) for key,value in reference.items())
    assert 'parameter_drift_block_00' in row and 'parameter_drift_block_01' in row


@pytest.mark.parametrize('sp,feature',[(True,False),(False,True),(True,True)])
def test_regularized_checkpoint_resume_exact_and_original_reference(fixture,tmp_path,sp,feature):
    cfg,data,collate,identity,init=fixture
    cfg=enable(cfg,sp,feature)
    full=tmp_path/'full'; resume=tmp_path/'resume'
    a,ao=execute(cfg,data,collate,identity,init,full,device='cpu')
    execute(cfg,data,collate,identity,init,resume,device='cpu',interrupt_after=2)
    b,bo=execute(cfg,data,collate,identity,init,resume,device='cpu',resume=True)
    assert all(torch.equal(value,b.state_dict()[key]) for key,value in a.state_dict().items())
    for key,state in ao.state_dict()['state'].items():
        for name,value in state.items(): assert torch.equal(value,bo.state_dict()['state'][key][name])
    b.regularization.verify_original(torch.load(init,weights_only=True))
    changed=copy.deepcopy(cfg);changed.regularization.lambda_sp=3.
    with pytest.raises(ValueError,match='differs'):
        execute(changed,data,collate,identity,init,resume,device='cpu',resume=True)


def test_probe_is_unregularized_trajectory_but_measures_nonzero_gradients(fixture):
    cfg,data,collate,_,init=fixture
    a,ao=initialize(cfg,init,'cpu')
    stream=iter(loader(cfg,data,collate,'cpu',0,4))
    for i in range(4): accumulated_step(a,ao,stream,schedules(cfg),i,precision='fp32')
    probe_cfg=enable(cfg)
    b,bo=initialize(probe_cfg,init,'cpu')
    stream=iter(loader(cfg,data,collate,'cpu',0,4))
    for i in range(4): row=accumulated_step(b,bo,stream,schedules(cfg),i,precision='fp32',gradient_probe=GradientProbe(False))
    for key,value in a.state_dict().items(): assert torch.equal(value,b.state_dict()[key]),key
    assert row['calibration_ssl_gradient_norm']>0 and row['calibration_sp_gradient_norm']>0
    assert row['calibration_feature_gradient_norm']>0


def test_gradient_ratio_selection_and_joint_guard():
    rows=[dict(calibration_ssl_gradient_norm=2.,calibration_sp_gradient_norm=.002,
               calibration_feature_gradient_norm=.04) for _ in range(3)]
    values,diagnostics=select_coefficients(rows)
    assert values['lambda_sp']==pytest.approx(100.) and values['lambda_feat']==pytest.approx(5.)
    assert check_combined(rows,values)['maximum_combined_gradient_triangle_bound']==pytest.approx(.2)
    rows[0]['calibration_feature_gradient_norm']=10.
    with pytest.raises(ValueError,match='non-dominance'):check_combined(rows,values)
    rows[0]['calibration_feature_gradient_norm']=0.
    with pytest.raises(ValueError,match='No measurable'):select_coefficients(rows)


def test_calibration_receipt_same_coefficients_B_D_C_D_and_batch_binding(fixture,tmp_path):
    cfg,_,_,data,init=fixture
    coefficients=dict(lambda_sp=100.,lambda_feat=5.,lambda_cls=1.,lambda_patch=1.)
    path=tmp_path/'calibration.json'
    path.write_text(json.dumps(dict(status='frozen',common_identity=common_identity(cfg,data,init),coefficients=coefficients)))
    for arm in ARMS:
        arm_cfg=arm_config(cfg,arm,tmp_path/arm,coefficients,path)
        require_calibration(arm_cfg,data,init)
    arm_cfg.runtime.physical_batch=32
    with pytest.raises(ValueError,match='differs'):require_calibration(arm_cfg,data,init)


def test_training_only_calibration_fixture_and_frozen_receipt(fixture,tmp_path):
    cfg,data,collate,identity,init=fixture
    output=tmp_path/'calibration.json'
    result=calibrate(cfg,data,collate,identity,init,output,warmup=2,probes=2,
                     device='cpu',target=.01,minimum=1e-8,maximum=1.)
    assert result['status']=='frozen' and result['training_data_only']
    assert not result['validation_or_test_accessed'] and not result['weights_promoted']
    assert result['schedule_offset']==cfg.experiment.warmup_steps
    assert len(result['unregularized_probe_rows'])==len(result['combined_probe_rows'])==2
    assert result['coefficients']['lambda_sp']>0 and result['coefficients']['lambda_feat']>0
    require_calibration(arm_config(cfg,'D',tmp_path/'D',result['coefficients'],output),identity,init)
    with pytest.raises(FileExistsError):
        calibrate(cfg,data,collate,identity,init,output,warmup=2,probes=2,device='cpu')


def test_joint_calibration_gradient_application_matches_normal_update(fixture):
    cfg,data,collate,_,init=fixture
    cfg=enable(cfg)
    a,ao=initialize(cfg,init,'cpu')
    rng=rng_state()
    stream=iter(loader(cfg,data,collate,'cpu',0,4))
    for i in range(4): accumulated_step(a,ao,stream,schedules(cfg),i,precision='fp32')
    b,bo=initialize(cfg,init,'cpu');restore_rng(rng)
    stream=iter(loader(cfg,data,collate,'cpu',0,4))
    for i in range(4): accumulated_step(b,bo,stream,schedules(cfg),i,precision='fp32',gradient_probe=GradientProbe(True))
    for key,value in a.state_dict().items():
        torch.testing.assert_close(value,b.state_dict()[key],rtol=2e-5,atol=2e-7)


def test_full_evaluation_uses_all_heads_at_early_milestones(tmp_path,monkeypatch):
    from scripts.evaluate_sem_cellstyle import commands
    from src.sem_cellstyle.recipe import resolve
    cfg=resolve(configuration(),200000,32)
    run=tmp_path/'run';run.mkdir();(run/'run.json').write_text(json.dumps({'config':OmegaConf.to_container(cfg)}))
    monkeypatch.setattr('src.ssl.checkpoints.validate_snapshot',lambda path:{})
    rows=list(commands(cfg,run,tmp_path/'eval',[1,5],full=True))
    assert len(rows)==6
    for _,task,command in rows:
        if task=='report': continue
        selected=command[command.index('--heads')+1:command.index('--workers')]
        assert selected==(['logreg','mlp'] if task=='classification' else ['linear','cnn8'])


def test_report_deltas_and_threshold_protocols():
    from src.sem_cellstyle.regularization_report import scalar_metrics,differences
    points=dict(ma_iou=.4,miou=.6,ma_precision=.7,ma_recall=.8)
    metrics=scalar_metrics(dict(nffa_A_logreg=dict(macro_f1=.9),aachen_linear=dict(fixed=points,selected=points|{'ma_iou':.45})))
    assert metrics['aachen_linear.fixed.ma_iou']==.4 and metrics['aachen_linear.selected.ma_iou']==.45
    assert differences(.85,.9,.8)['delta_vs_original']==pytest.approx(-.05)
    assert differences(.85,.9,.8)['delta_vs_A']==pytest.approx(.05)
    assert differences(None,.9,.8)['delta_vs_A'] is None


def test_grid_orchestration_freezes_common_recipe_and_skips_completed_training(fixture,tmp_path,monkeypatch):
    from types import SimpleNamespace
    from scripts import run_sem_cellstyle_regularization_grid as grid
    from src.sem_cellstyle.benchmark import benchmark_identity,recommendation
    from src.sem_cellstyle.recipe import resolve,contract
    from src.ssl.experiment import digest
    _,_,_,_,init=fixture
    base=configuration();base.initialization.checkpoint=str(init);base.initialization.sha256=sha256(init)
    base.runtime.allow_cuda_resume_update_drift=True
    dataset=range(200000);data=dict(provider='NONSCIENTIFIC_FAKE_ORCHESTRATION',count=len(dataset))
    args=SimpleNamespace(output=tmp_path/'grid',stage='run',calibration_warmup_updates=20,calibration_probes=10,
                         benchmark_warmup_updates=10,benchmark_measured_updates=30,verified_data_root=None)
    calls=[]
    monkeypatch.setattr(grid,'validate_snapshot',lambda path: dict(step=int(path.name.split('_')[-1]),files={'teacher_backbone.pth':'fixture'}))
    monkeypatch.setattr('src.sem_cellstyle.regularization_report.write_report',lambda root: root/'report/fixture.md')
    def runner(command,log):
        calls.append(command)
        module=command[2]
        file=Path(command[command.index('--config')+1])
        cfg=configuration(file)
        if module=='scripts.benchmark_sem_cellstyle_a100':
            output=Path(command[command.index('--output')+1]);output.mkdir(parents=True)
            i=command.index('--batch-sizes')+1;batches=[]
            while i<len(command) and not command[i].startswith('--'):
                batches.append(int(command[i]));i+=1
            rows=[dict(physical_batch=b,status='ok',finite=True,peak_reserved_bytes=b,total_vram_bytes=80,
                       original_images_per_second=b,resume_verified=True,export_verified=True,
                       resume_update_bitwise_exact=False,cuda_update_drift_accepted=True) for b in batches]
            result=dict(results=rows,recommendation=recommendation(rows),benchmark_identity=benchmark_identity(resolve(cfg,len(dataset),batches[0]),data,init))
            (output/'benchmark.json').write_text(json.dumps(result))
        elif module=='scripts.run_sem_cellstyle_regularization_grid':
            stage=command[command.index('--stage')+1];root=Path(command[command.index('--output')+1])
            if stage=='calibrate':
                batch=int(command[command.index('--physical-batch')+1])
                item=dict(status='frozen',common_identity=common_identity(resolve(cfg,len(dataset),batch),data,init),
                          coefficients=dict(lambda_sp=100.,lambda_feat=5.,lambda_cls=1.,lambda_patch=1.))
                (root/'calibration.json').write_text(json.dumps(item))
            else:(root/'diagnostics.json').write_text(json.dumps(dict(results=[])))
        elif module=='scripts.train_sem_cellstyle':
            stage=command[command.index('--stage')+1]
            if stage=='train':
                run=Path(cfg.runtime.output);run.mkdir(parents=True)
                (run/'run.json').write_text(json.dumps(dict(config=OmegaConf.to_container(cfg),contract=contract(cfg,data,init,'cuda'))))
                (run/'status.json').write_text(json.dumps(dict(status='stage_complete',step=cfg.experiment.run_until_step)))
                for step in cfg.experiment.milestones:
                    if step<=cfg.experiment.run_until_step:(run/f'checkpoints/step_{step:06d}').mkdir(parents=True)
        elif module=='scripts.evaluate_sem_cellstyle':
            output=Path(command[command.index('--output')+1]);(output/'dinov2_vitl14_fixture').mkdir(parents=True,exist_ok=True)
        else:raise AssertionError(module)
    grid.run_study(args,base,dataset,None,data,init,runner)
    training=[command for command in calls if command[2]=='scripts.train_sem_cellstyle' and command[command.index('--stage')+1]=='train']
    assert [Path(c[c.index('--config')+1]).stem for c in training]==list(ARMS)
    first_train=calls.index(training[0])
    assert len([c for c in calls[:first_train] if c[2]=='scripts.benchmark_sem_cellstyle_a100'])==5
    for arm in ARMS:
        cfg=configuration(args.output/f'configs/{arm}.yaml')
        assert cfg.runtime.physical_batch==32 and cfg.runtime.effective_batch==128 and cfg.runtime.accumulation_steps==4
        assert cfg.regularization.lambda_sp==100 and cfg.regularization.lambda_feat==5
        assert cfg.runtime.allow_cuda_resume_update_drift is True
    calls.clear();grid.run_study(args,base,dataset,None,data,init,runner)
    assert not [c for c in calls if c[2]=='scripts.train_sem_cellstyle']
    changed=copy.deepcopy(base);changed.optim.head_lr=2e-4
    with pytest.raises(ValueError,match='changed'):
        grid.run_study(args,changed,dataset,None,data,init,runner)


def test_grid_plan_uses_no_dataset_or_gpu(capsys,monkeypatch):
    from scripts.run_sem_cellstyle_regularization_grid import main
    monkeypatch.setattr('src.sem_cellstyle.data.provider',lambda cfg: (_ for _ in ()).throw(AssertionError('dataset accessed')))
    monkeypatch.setattr('src.sem_cellstyle.runner.require_a100',lambda: (_ for _ in ()).throw(AssertionError('GPU accessed')))
    main(['--stage','plan'])
    value=json.loads(capsys.readouterr().out)
    assert not value['approved_manifest_configured'] and not value['gpu_checked'] and not value['training_started']


def test_feature_calibration_rejects_numerical_noise():
    row=dict(calibration_ssl_gradient_norm=1.,calibration_sp_gradient_norm=.01,calibration_feature_gradient_norm=1e-7)
    with pytest.raises(ValueError,match='numerical noise'):
        select_coefficients([row,row],feature_noise_floor=5e-8)


def test_relative_token_weights_are_shared_and_bound(fixture,tmp_path):
    cfg,_,_,data,init=fixture
    original=common_identity(cfg,data,init)
    cfg.regularization=dict(l2_sp=False,feature_distillation=False,lambda_cls=.5,lambda_patch=1.5)
    assert common_identity(cfg,data,init)!=original
    for arm in ARMS:
        result=arm_config(cfg,arm,tmp_path/arm)
        assert result.regularization.lambda_cls==.5 and result.regularization.lambda_patch==1.5


def test_joint_report_serializes_both_deltas_and_plots(tmp_path,monkeypatch):
    import csv
    from src.sem_cellstyle import regularization_report as report
    rows=[]
    for arm in ARMS:
        for point in (0,1,5,10):
            scores={'nffa_A_logreg.macro_f1':.8,'nffa_B_mlp.macro_f1':.81,
                    'aachen_linear.fixed.ma_iou':.4,'aachen_cnn8.selected.ma_iou':.45}
            rows.append(dict(arm=arm,presentation_passes=point,step=point*10,scalar_metrics=scores,
                comparison={key:report.differences(value,.85,.78) for key,value in scores.items()},
                training=dict(ssl_loss=2.,sp_raw_loss=.01,feature_weighted_loss=.02),
                student_drift_by_block={'block_00':.01,'block_01':.02},
                teacher_training_view_similarity=dict(cls_similarity=.99,patch_similarity=.98)))
    result=dict(rows=rows,classification_splits=['A','B'],coefficients=dict(lambda_sp=100,lambda_feat=5,lambda_cls=1,lambda_patch=1),
                runtime={arm:dict(optimizer_update_wall_seconds=10.,peak_allocated_vram_bytes=2**30) for arm in ARMS},
                missing=['future milestone 100'],recommendation='SYNTHETIC TEST ONLY')
    monkeypatch.setattr(report,'collect',lambda root:result)
    output=report.write_report(tmp_path)
    assert output.is_file()
    with output.with_suffix('.csv').open() as stream: written=list(csv.DictReader(stream))
    first=next(row for row in written if row['metric']=='nffa_A_logreg.macro_f1')
    assert float(first['delta_vs_original'])==pytest.approx(-.05)
    assert float(first['delta_vs_A'])==pytest.approx(.02)
    assert all((output.parent/name).stat().st_size>0 for name in ('comparison.json','validation_vs_pass.png','drift_and_feature_similarity.png'))
    assert report.write_report(tmp_path)==output
