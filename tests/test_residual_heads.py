"""Residual paths, profile isolation and actual CPU head training/resume."""
import copy
import json
from unittest import mock

import numpy as np
import pytest
import torch
from torch import nn

from src.downstream.heads import ClassificationHead, DeepCNN, SpatialResidual
from src.downstream.residual_heads import ResidualMLP, ResidualMLPBlock
from src.downstream.probes import HEAD_CONFIGS, head_configurations, mlp_model, fit_mlp, evaluate_probe
from test_encoder_evaluation import fixtures


@pytest.mark.parametrize('dim', [384,1024])
def test_new_profile_has_three_residual_hidden_blocks_and_ten_percent_dropout(dim):
    config=head_configurations('residual_v1')['mlp']
    model=mlp_model(config,dim)
    assert isinstance(model,ResidualMLP)
    assert [(b.main[0].in_features,b.main[0].out_features) for b in model.blocks]==[
        (dim,512),(512,256),(256,128)]
    assert all(isinstance(b.shortcut,nn.Linear) and b.shortcut.bias is None for b in model.blocks)
    assert [b.main[-1].p for b in model.blocks]==[.1,.1,.1]
    x=torch.randn(4,dim,requires_grad=True)
    y=model(x); assert y.shape==(4,10) and torch.isfinite(y).all()
    nn.functional.cross_entropy(y,torch.tensor([0,1,2,3])).backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
    assert all(b.shortcut.weight.grad is not None and torch.isfinite(b.shortcut.weight.grad).all() for b in model.blocks)
    model.eval()
    assert torch.equal(model(x),model(x))


@pytest.mark.parametrize('widths',[(8,8),(8,5)])
def test_zero_main_branch_proves_real_shortcut_and_input_gradient(widths):
    a,b=widths
    block=ResidualMLPBlock(a,b).eval()
    with torch.no_grad():
        block.main[0].weight.zero_(); block.main[0].bias.zero_()
    x=torch.randn(3,a,requires_grad=True)
    expected=block.shortcut(x)
    actual=block(x)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    gradient=torch.autograd.grad(actual.sum(),x)[0]
    reference=torch.autograd.grad(expected.sum(),x)[0]
    torch.testing.assert_close(gradient,reference,rtol=0,atol=0)
    assert isinstance(block.shortcut,nn.Identity) == (a==b)


def test_cnn8_already_has_four_residual_blocks_and_ten_percent_dropout():
    cnn=DeepCNN(1024,8)
    assert len(cnn.blocks)==4 and all(isinstance(b,SpatialResidual) for b in cnn.blocks)
    assert sum(isinstance(m,nn.Conv2d) and m.kernel_size==(3,3) for m in cnn.modules())==8
    assert [m.p for m in cnn.modules() if isinstance(m,nn.Dropout2d)]==[.1]*4
    x=torch.randn(2,128,7,9,requires_grad=True)
    output=cnn.blocks(x)
    torch.testing.assert_close(output,x,rtol=0,atol=0)
    output.sum().backward()
    torch.testing.assert_close(x.grad,torch.ones_like(x),rtol=0,atol=0)
    assert cnn(torch.randn(2,1024,7,9)).shape==(2,2,7,9)


def test_legacy_profile_and_other_training_hyperparameters_remain_available():
    legacy=head_configurations('legacy'); current=head_configurations('residual_v1')
    assert legacy==HEAD_CONFIGS and legacy['mlp']['dropout']==.3
    assert isinstance(mlp_model(legacy['mlp']),ClassificationHead)
    assert legacy['logreg']==current['logreg']
    for key in ('hidden_dims','num_classes','lr','weight_decay','batch_size','epochs','patience','loss','selection','scaler'):
        assert legacy['mlp'][key]==current['mlp'][key]
    current['mlp']['hidden_dims'][0]=1
    assert HEAD_CONFIGS['mlp']['hidden_dims']==[512,256,128]
    with pytest.raises(ValueError): head_configurations('unknown')


def test_residual_head_training_checkpoint_and_exact_epoch_resume(tmp_path):
    torch.set_num_threads(1)
    config=head_configurations('residual_v1')['mlp']
    config.update(hidden_dims=[16,12,8],batch_size=4,epochs=3)
    rng=np.random.default_rng(42)
    train,val=rng.normal(size=(12,24)).astype('float32'),rng.normal(size=(6,24)).astype('float32')
    y,yv=np.arange(12)%3,np.arange(6)%3
    full,resumed=tmp_path/'full',tmp_path/'resumed'; full.mkdir();resumed.mkdir()
    device=torch.device('cpu')
    fit_mlp(full,train,y,val,yv,config,42,0,device)
    from src.downstream import probes
    original_save=probes.save_csv
    def interrupt(path,rows):
        original_save(path,rows)
        if len(rows)==1: raise InterruptedError('CPU fixture interruption after durable checkpoint')
    with mock.patch.object(probes,'save_csv',side_effect=interrupt):
        with pytest.raises(InterruptedError): fit_mlp(resumed,train,y,val,yv,config,42,0,device)
    fit_mlp(resumed,train,y,val,yv,config,42,0,device,resume=True)
    assert (full/'history.csv').read_bytes()==(resumed/'history.csv').read_bytes()
    a=torch.load(full/'last_state.pt',weights_only=False)
    b=torch.load(resumed/'last_state.pt',weights_only=False)
    for name,value in a['model'].items(): torch.testing.assert_close(value,b['model'][name],rtol=0,atol=0)
    assert b['head_config']['architecture_version']=='residual_mlp_v1'
    result=evaluate_probe(resumed,'mlp',val,yv,config,device)
    assert 0<=result['macro_f1']<=1
    changed=copy.deepcopy(config); changed['dropout']=.2
    with pytest.raises(ValueError,match='Resume head configuration'):
        fit_mlp(resumed,train,y,val,yv,changed,42,0,device,resume=True)
    with pytest.raises(ValueError,match='Evaluation head configuration'):
        evaluate_probe(resumed,'mlp',val,yv,changed,device)


def test_pilot_and_generated_evaluation_commands_select_versioned_profile(tmp_path,monkeypatch):
    from src.sem_cellstyle.recipe import configuration,resolve
    from scripts.evaluate_sem_cellstyle import commands
    from omegaconf import OmegaConf
    cfg=configuration('configs/sem_ssl/sem_cellstyle_nffa_vitl14_pilot.yaml')
    assert cfg.evaluation.head_profile=='residual_v1'
    assert 'residual_heads_v1' in cfg.runtime.output
    resolved=resolve(cfg,12801,32)
    run=tmp_path/'run';run.mkdir()
    (run/'run.json').write_text(json.dumps(dict(config=OmegaConf.to_container(resolved))))
    monkeypatch.setattr('src.ssl.checkpoints.validate_snapshot',lambda path:{})
    for _,_,command in commands(cfg,run,tmp_path/'eval',[0,1,5,10],full=True):
        assert command[command.index('--head-profile')+1]=='residual_v1'


def test_shared_runner_records_new_head_and_rejects_legacy_resume(fixtures):
    from scripts import evaluate_sem_encoder as runner
    from test_encoder_evaluation import arguments
    args=arguments(fixtures)+['--task','classification','--splits','A','--heads','mlp',
                               '--head-profile','residual_v1','--stage','all']
    runner.main(args)
    output=fixtures.root/'out/NONSCIENTIFIC_SMOKE/dinov2_vits14_FAKE/classification/split_A/mlp/seed42'
    config=json.loads((output/'config.json').read_text())
    assert config['head_profile']=='residual_v1' and config['head_config']['dropout']==.1
    assert config['head_config']['architecture_version']=='residual_mlp_v1'
    runner.main(args+['--stage','all','--resume'])
    with pytest.raises(SystemExit): runner.main(args+['--head-profile','legacy','--resume'])
