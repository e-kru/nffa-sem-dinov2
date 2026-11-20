"""Small CPU Meta fixtures. They are not measurements of the real Large model."""
import copy
import csv
import json
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image
import pytest
import torch
from omegaconf import OmegaConf

from src.sem_cellstyle.data import SEMViews, PresentationBatches, CorpusDataset, collate_for, provider
from src.sem_cellstyle.recipe import configuration, budgets, resolve, schedules, contract
from src.sem_cellstyle.runner import execute, initialize
from src.sem_cellstyle.training import Centers, accumulated_step, checkpoint_student
from src.ssl.checkpoints import find_latest, rng_state, restore_rng
from src.ssl.data import sha256
from src.ssl.stream import DrawDataset, DrawCollate, data_rng
from src.ssl.upstream import activate

activate()


def tiny(**kwargs):
    from dinov2.models.vision_transformer import DinoVisionTransformer
    return DinoVisionTransformer(img_size=56, patch_size=14, embed_dim=24, depth=2, num_heads=3,
                                 block_chunks=0, init_values=1, drop_path_rate=.1)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    import dinov2.hub.backbones
    monkeypatch.setattr(dinov2.hub.backbones, 'dinov2_vitl14', tiny)
    cfg = resolve(configuration(), 200000, 32)
    cfg.runtime.physical_batch = 2
    cfg.train.batch_size_per_gpu, cfg.runtime.accumulation_steps, cfg.runtime.effective_batch = 2, 2, 4
    cfg.train.num_workers = 0
    cfg.crops.global_crops_size, cfg.crops.local_crops_size, cfg.crops.local_crops_number = 56, 28, 2
    cfg.dino.head_n_prototypes, cfg.dino.head_hidden_dim, cfg.dino.head_bottleneck_dim = 16, 32, 8
    cfg.ibot.mask_ratio_min_max = [.3,.5]
    cfg.experiment.schedule_total_steps, cfg.experiment.run_until_step = 6, 4
    cfg.experiment.warmup_steps, cfg.experiment.teacher_temp_warmup_steps = 2, 4
    cfg.experiment.freeze_last_layer_steps = 1
    cfg.experiment.milestones = [0,2,4,6]
    cfg.experiment.pass_to_step = {'0':0,'1':1,'5':2,'10':4,'25':4,'50':5,'100':6}
    cfg.checkpointing.recovery_every_optimizer_updates = 1
    initialization = tmp_path / 'original.pth'
    torch.manual_seed(7); torch.save(tiny().state_dict(), initialization)
    records = []
    for i in range(7):
        path = tmp_path / f'image{i}.png'
        Image.fromarray(np.random.default_rng(i).integers(0,256,(77,101),dtype=np.uint8)).save(path)
        records.append(dict(path=str(path), source='fixture', image_id=str(i), original_id=str(i), specimen_id=str(i), split='train'))
    data = CorpusDataset(records, {'fixture': {'orientation_invariant':False}}, SEMViews(cfg))
    identity = dict(count=7, provider='NONSCIENTIFIC_SYNTHETIC_FIXTURE', records=records)
    return cfg, data, collate_for(cfg), identity, initialization


def test_real_recipe_and_example_ceiling_budget():
    cfg = configuration()
    assert cfg.student.arch == 'vit_large' and cfg.student.num_register_tokens == 0
    assert cfg.runtime.physical_batch is None
    example = budgets(cfg,200000)
    assert example['schedule_total_steps'] == 156250 and example['run_until_step'] == 15625
    assert example['milestones'] == [0,1563,7813,15625,39063,78125,156250]
    assert (example['warmup_steps'],example['freeze_last_layer_steps'],example['teacher_temp_warmup_steps']) == (7813,1563,31250)
    with pytest.raises(ValueError): resolve(cfg,200000,None)
    for batch,k in ((16,8),(32,4)):
        resolved = resolve(cfg,200000,batch)
        assert resolved.runtime.accumulation_steps == k
        assert resolved.optim.lr == 1e-5
        schedule = schedules(resolved)
        assert len(schedule['backbone_lr'].schedule) == 156250
        assert max(schedule['backbone_lr'].schedule) == pytest.approx(1e-5)
        assert max(schedule['head_lr'].schedule) == pytest.approx(1e-4)
        assert schedule['teacher_temp'][0] == .04 and schedule['teacher_temp'][31249] == .07
        resolved.experiment.run_until_step = 50000
        np.testing.assert_array_equal(schedules(resolved)['backbone_lr'].schedule, schedule['backbone_lr'].schedule)


def test_unready_corpus_fails_closed():
    with pytest.raises(ValueError,match='Dataset not ready'): provider(configuration())


def test_manifest_provider_uses_receipt_and_never_hashes_images(fixture,tmp_path):
    cfg, data, _, _, _ = fixture
    manifest = tmp_path/'manifest.csv'
    with manifest.open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(data.records[0])); writer.writeheader(); writer.writerows(data.records)
    receipt = dict(schema='verified_sem_manifest_v1', status='verified', verified_train_originals=7,
        manifest_sha256=sha256(manifest), corpus_version='CPU fixture', content_snapshot='immutable fixture only', curation_record='fixture only',
        leakage_review_record='fixture only', sources={'fixture':{}}, exclusions=dict(
            nffa_A_B_val_test_and_groups=True,all_aachen=True,nasa_benchmarks_if_evaluated=True))
    receipt_path = tmp_path/'receipt.json'; receipt_path.write_text(json.dumps(receipt))
    cfg.dataset.manifest, cfg.dataset.receipt = str(manifest), str(receipt_path)
    with mock.patch('src.sem_cellstyle.data.sha256', wraps=sha256) as hashing:
        ds, _, identity = provider(cfg)
        assert identity['count'] == len(ds) == 7
        assert {call.args[0] for call in hashing.call_args_list} == {manifest,receipt_path}
    assert ds[0][0]['global_crops'][0].shape == (3,56,56)
    receipt['verified_train_originals'] = 8; receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='does not match'): provider(cfg)
    receipt['verified_train_originals'] = 7; receipt['exclusions']['all_aachen'] = False
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='held-out'): provider(cfg)


def test_square_geometry_gray_replication_and_orientation_gate():
    views = SEMViews(configuration())
    for _ in range(100):
        x0,y0,x1,y1 = views.square_box((1024,768),(.5,1))
        assert 543 <= x1-x0 == y1-y0 <= 768
        assert 0 <= x0 < x1 <= 1024 and 0 <= y0 < y1 <= 768
    image = Image.fromarray(np.arange(80*100,dtype=np.uint8).reshape(80,100))
    with mock.patch('src.sem_cellstyle.data.ImageOps.mirror') as mirror, mock.patch.object(Image.Image,'rotate') as rotate:
        result = views(image,False)
        assert not mirror.called and not rotate.called
    assert len(result['global_crops']) == 2 and len(result['local_crops']) == 8
    value = result['global_crops'][0]
    denorm = value*torch.tensor([.229,.224,.225])[:,None,None]+torch.tensor([.485,.456,.406])[:,None,None]
    torch.testing.assert_close(denorm[0],denorm[1]); torch.testing.assert_close(denorm[1],denorm[2])
    with data_rng(42): a = views(image)
    with data_rng(42): b = views(image)
    assert torch.equal(a['global_crops'][0],b['global_crops'][0])
    assert not torch.equal(a['global_crops'][0],a['global_crops'][1])


def test_permutation_keeps_epoch_tails_and_seeks():
    batches = list(PresentationBatches(7,2,42,0,10))
    draws = [item for batch in batches for item in batch]
    assert sorted(i for i,_ in draws[:7]) == list(range(7))
    assert sorted(i for i,_ in draws[7:14]) == list(range(7))
    assert [d for _,d in draws] == list(range(20))
    assert batches[3:] == list(PresentationBatches(7,2,42,3,10))


def test_original_components_teacher_copy_and_native_checkpointing(fixture):
    from dinov2.loss import DINOLoss, iBOTPatchLoss, KoLeoLoss
    cfg,_,_,_,init = fixture
    model,opt = initialize(cfg,init,'cpu')
    assert type(model.dino_loss) is DINOLoss and type(model.ibot_patch_loss) is iBOTPatchLoss and type(model.koleo_loss) is KoLeoLoss
    assert model.dino_loss.student_temp == .1 and model.dino_loss.center_momentum == .9
    for name,p in model.student.named_parameters():
        target = dict(model.teacher.named_parameters())[name]
        assert p.requires_grad and not target.requires_grad and torch.equal(p,target) and p.data_ptr()!=target.data_ptr()
    assert not model.teacher.training and all(p.dtype==torch.float32 for p in model.parameters())
    cfg.runtime.student_activation_checkpointing=False
    reference,_=initialize(cfg,init,'cpu')
    x=torch.randn(2,3,56,56)
    state=rng_state(); a=model.student.backbone(x); a.sum().backward()
    restore_rng(state); b=reference.student.backbone(x); b.sum().backward()
    assert torch.equal(a,b)
    for (name,p),(_,q) in zip(model.student.backbone.named_parameters(),reference.student.backbone.named_parameters()):
        if p.grad is not None: torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
    assert all('_checkpoint' not in name for name in model.state_dict())


def test_centers_constant_across_microbatches_and_one_ema_optimizer(fixture):
    cfg,data,collate,identity,init=fixture
    model,opt=initialize(cfg,init,'cpu')
    from src.sem_cellstyle.runner import loader
    stream=iter(loader(cfg,data,collate,'cpu',2,3))
    old_dino=model.dino_loss.center.clone(); old_ibot=model.ibot_patch_loss.center.clone()
    seen=[]
    original=Centers.add
    def capture(self,cls,patches):
        assert torch.equal(model.dino_loss.center,old_dino) and torch.equal(model.ibot_patch_loss.center,old_ibot)
        seen.append((cls.detach().clone(),patches.detach().clone())); original(self,cls,patches)
    with mock.patch.object(Centers,'add',capture), mock.patch.object(opt,'step',wraps=opt.step) as step, \
         mock.patch.object(model,'update_teacher',wraps=model.update_teacher) as ema, \
         mock.patch.object(model.dino_loss,'update_center',wraps=model.dino_loss.update_center) as dino, \
         mock.patch.object(model.ibot_patch_loss,'update_center',wraps=model.ibot_patch_loss.update_center) as ibot:
        row=accumulated_step(model,opt,stream,schedules(cfg),2,precision='fp32')
        assert step.call_count==ema.call_count==dino.call_count==ibot.call_count==1
    torch.testing.assert_close(model.dino_loss.center,torch.cat([x[0] for x in seen]).mean(0,keepdim=True)*.1)
    torch.testing.assert_close(model.ibot_patch_loss.center,torch.cat([x[1] for x in seen]).mean(0,keepdim=True)[None]*.1)
    assert np.isfinite(row['total_loss']) and len(row['draw_ids'])==4
    assert row['student_entropy']>0 and row['teacher_entropy']>0 and row['feature_variance']>0
    assert any(group['module']=='dino_head' for group in opt.param_groups)


def test_last_head_layer_is_frozen_during_first_pass(fixture):
    cfg,data,collate,_,init=fixture
    from src.sem_cellstyle.runner import loader
    model,opt=initialize(cfg,init,'cpu')
    before={name:p.clone() for name,p in model.student.dino_head.last_layer.named_parameters()}
    accumulated_step(model,opt,iter(loader(cfg,data,collate,'cpu',0,1)),schedules(cfg),0,precision='fp32')
    for name,p in model.student.dino_head.last_layer.named_parameters():
        assert p.grad is not None and torch.equal(p,before[name])
        assert 'exp_avg' in opt.state[p]
    assert all(group['lr']==0 for group in opt.param_groups if group['is_last_layer'])


def test_resume_full_state_and_export_features_match(fixture,tmp_path):
    cfg,data,collate,identity,init=fixture
    full=tmp_path/'full'; resumed=tmp_path/'resumed'
    a,ao=execute(cfg,data,collate,identity,init,full,device='cpu')
    execute(cfg,data,collate,identity,init,resumed,device='cpu',interrupt_after=2)
    b,bo=execute(cfg,data,collate,identity,init,resumed,device='cpu',resume=True)
    assert all(torch.equal(value,b.state_dict()[key]) for key,value in a.state_dict().items())
    for key,state in ao.state_dict()['state'].items():
        for name,value in state.items(): assert torch.equal(value,bo.state_dict()['state'][key][name])
    folder,payload,_=find_latest(resumed)
    assert payload['data_cursor']['image_presentations']==16
    assert payload['data_cursor']['completed_microbatches']==8
    assert json.loads((folder/'feature_verification.json').read_text())['live_teacher_feature_max_abs_error']==0
    assert len(list((resumed/'recovery').glob('step_*')))==2
    assert sha256(full/'checkpoints/step_000000/teacher_backbone.pth')==sha256(init)
    changed=OmegaConf.create(OmegaConf.to_container(cfg)); changed.runtime.physical_batch=16
    with pytest.raises(ValueError,match='differs'): execute(changed,data,collate,identity,init,resumed,device='cpu',resume=True)


def test_contract_allows_stop_extension_only(fixture):
    cfg,_,_,identity,init=fixture
    first=contract(cfg,identity,init,'cpu')
    cfg.experiment.run_until_step=6
    assert contract(cfg,identity,init,'cpu')==first
    cfg.experiment.schedule_total_steps=7
    assert contract(cfg,identity,init,'cpu')!=first


def test_large_heads_and_cache_dimensions(tmp_path):
    from src.downstream.probes import HEAD_CONFIGS, mlp_model
    from src.aachen_ablation.engine import make_decoder
    from src.downstream.benchmark import feature_contract
    from src.aachen_ablation.cache import cache_contract
    from src.aachen_ablation.geometry import VARIANTS
    mlp=mlp_model(HEAD_CONFIGS['mlp'],1024).eval()
    assert mlp(torch.randn(2,1024)).shape==(2,10)
    for head in ('linear','cnn8'):
        assert make_decoder(head,1024)(torch.randn(2,1024,4,4)).shape==(2,2,4,4)
    manifest=tmp_path/'manifest'; manifest.write_text('fixture')
    identity=dict(model='dinov2_vitl14',feature_dim=1024)
    assert feature_contract([],manifest,manifest,tmp_path,identity,False)['feature_shape']==[1024]
    assert cache_contract(manifest,[],VARIANTS['A3'],'float32',None,False,'dinov2_vitl14')['feature_shape']==[1,1024,48,64]


def test_recommendation_reserve_stability_and_smallest_near_best():
    from src.sem_cellstyle.benchmark import recommendation
    rows=[dict(physical_batch=16,status='ok',finite=True,peak_reserved_bytes=60,total_vram_bytes=80,original_images_per_second=40),
          dict(physical_batch=32,status='ok',finite=True,peak_reserved_bytes=70,total_vram_bytes=80,original_images_per_second=41)]
    assert recommendation(rows)['physical_batch']==16
    rows[1]['original_images_per_second']=50
    assert recommendation(rows)['physical_batch']==32
    rows[1]['peak_reserved_bytes']=75
    assert recommendation(rows)['physical_batch']==16
    rows[0]['status']='oom'
    assert recommendation(rows)['physical_batch'] is None


def test_nonfinite_update_stops_before_optimizer(fixture):
    cfg,data,collate,_,init=fixture
    from src.sem_cellstyle.runner import loader
    model,opt=initialize(cfg,init,'cpu')
    with mock.patch.object(model,'forward',return_value=(torch.tensor(float('nan'),requires_grad=True),{})), \
         mock.patch.object(opt,'step') as step:
        with pytest.raises(FloatingPointError):
            accumulated_step(model,opt,iter(loader(cfg,data,collate,'cpu',0,1)),schedules(cfg),0,precision='fp32')
        step.assert_not_called()


def test_benchmark_subprocess_orchestration_and_serialization(tmp_path,monkeypatch):
    import scripts.benchmark_sem_cellstyle_a100 as cli
    import src.sem_cellstyle.data as data_module
    import src.sem_cellstyle.recipe as recipe_module
    import src.sem_cellstyle.benchmark as benchmark_module
    monkeypatch.setattr(data_module,'provider',lambda cfg: (list(range(200)),None,{'count':200}))
    monkeypatch.setattr(recipe_module,'require_initialization',lambda cfg: tmp_path/'original')
    monkeypatch.setattr(benchmark_module,'benchmark_identity',lambda *args: 'fixture_identity')
    seen=[]
    def subprocess(command,check):
        batch=int(command[command.index('--candidate')+1]); seen.append(command)
        output=Path(command[command.index('--output')+1])
        row=dict(physical_batch=batch,status='ok',finite=True,peak_reserved_bytes=batch,
                 total_vram_bytes=80,original_images_per_second=40.,resume_verified=True,export_verified=True)
        (output/f'candidate_{batch}.json').write_text(json.dumps(row))
        return mock.Mock(returncode=0)
    from types import SimpleNamespace
    monkeypatch.setattr(cli,'subprocess',SimpleNamespace(run=subprocess))
    out=tmp_path/'bench'
    cli.main(['--output',str(out),'--num-workers','0','--warmup-updates','1','--measured-updates','2'])
    result=json.loads((out/'benchmark.json').read_text())
    assert len(seen)==2 and ['--candidate','16'] == seen[0][seen[0].index('--candidate'):seen[0].index('--candidate')+2]
    assert result['recommendation']['physical_batch']==16
    assert not result['scientific_training_started'] and not result['weights_promoted']
    assert all((out/name).is_file() for name in ('benchmark.csv','benchmark.json','throughput_vs_batch.png','memory_vs_batch.png'))
    with (out/'benchmark.csv').open() as stream:
        assert len(list(csv.DictReader(stream)))==2
    with pytest.raises(FileExistsError):
        cli.main(['--output',str(out)])


def test_benchmark_candidate_oom_is_serialized(tmp_path,monkeypatch):
    import scripts.benchmark_sem_cellstyle_a100 as cli
    import src.sem_cellstyle.data as data_module
    import src.sem_cellstyle.recipe as recipe_module
    import src.sem_cellstyle.benchmark as benchmark_module
    monkeypatch.setattr(data_module,'provider',lambda cfg: ([],None,{}))
    monkeypatch.setattr(recipe_module,'require_initialization',lambda cfg: tmp_path/'original')
    def oom(*args): raise torch.cuda.OutOfMemoryError('synthetic OOM')
    monkeypatch.setattr(benchmark_module,'measure',oom)
    cli.main(['--output',str(tmp_path),'--candidate','32'])
    assert json.loads((tmp_path/'candidate_32.json').read_text())['status']=='oom'


def test_evaluation_commands_use_large_original_heads_and_validation_workers(tmp_path,monkeypatch):
    from scripts.evaluate_sem_cellstyle import commands
    import src.ssl.checkpoints as checkpoints
    cfg=configuration(); resolved=resolve(cfg,200000,32)
    run=tmp_path/'run'; run.mkdir()
    (run/'run.json').write_text(json.dumps({'config':OmegaConf.to_container(resolved)}))
    monkeypatch.setattr(checkpoints,'validate_snapshot',lambda path: {})
    rows=list(commands(cfg,run,tmp_path/'eval',[0,1]))
    assert len(rows)==6
    classification=rows[0][2]; segmentation=rows[1][2]; quick=rows[3][2]
    assert classification[classification.index('--encoder-name')+1]=='dinov2_vitl14'
    assert classification[classification.index('--workers')+1]=='8'
    assert segmentation[segmentation.index('--workers')+1]=='16'
    assert classification[classification.index('--heads')+1:classification.index('--workers')]==['logreg','mlp']
    assert quick[quick.index('--heads')+1:quick.index('--workers')]==['logreg']
    assert '--smoke-test' not in classification and '--execute' not in classification


def test_missing_large_results_are_explicit_and_small_reference_separate(tmp_path):
    from src.sem_cellstyle.reporting import collect
    cfg=configuration(); resolved=resolve(cfg,200000,32)
    run=tmp_path/'run';run.mkdir()
    (run/'run.json').write_text(json.dumps({'config':OmegaConf.to_container(resolved)}))
    result=collect(cfg,run,tmp_path/'eval')
    assert len(result['rows'])==7 and not result['complete']
    assert result['rows'][1]['evaluation_scope']=='quick'
    assert 'aachen_cnn8' in result['rows'][1]['deliberately_omitted']
    assert result['rows'][0]['metrics']['nffa_A_logreg'] is None
    assert not result['historical_small_scores_merged'] and not result['test_evaluated']


def test_absolute_layerwise_groups_and_optimizer_master_fp32(fixture):
    cfg,data,collate,_,init=fixture
    from src.sem_cellstyle.runner import loader
    model,opt=initialize(cfg,init,'cpu')
    by_parameter={id(p):group for group in opt.param_groups for p in group['params']}
    rates={name:by_parameter[id(p)]['lr_multiplier'] for name,p in model.student.backbone.named_parameters()}
    assert rates['blocks.1.attn.qkv.weight']==pytest.approx(1.)
    assert rates['blocks.0.attn.qkv.weight']==pytest.approx(.9)
    assert rates['patch_embed.proj.weight']==pytest.approx(.9**2*.2)
    accumulated_step(model,opt,iter(loader(cfg,data,collate,'cpu',2,3)),schedules(cfg),2,precision='fp32')
    for group in opt.param_groups:
        expected=cfg.optim.backbone_lr if group['module']=='backbone' else cfg.optim.head_lr
        assert group['lr']<=expected
    for state in opt.state.values():
        assert state['exp_avg'].dtype==state['exp_avg_sq'].dtype==torch.float32


def test_benchmark_rejects_changed_physical_recipe_or_unverified_result(fixture):
    from src.sem_cellstyle.benchmark import benchmark_identity,validate_choice
    cfg,_,_,data,init=fixture
    cfg.runtime.physical_batch=16;cfg.train.batch_size_per_gpu=16;cfg.runtime.accumulation_steps=8
    result=dict(benchmark_identity=benchmark_identity(cfg,data,init),results=[dict(physical_batch=16,status='ok',finite=True,
        peak_reserved_bytes=60,total_vram_bytes=80,resume_verified=True,export_verified=True)])
    validate_choice(result,cfg,data,init)
    result['results'][0].update(resume_update_bitwise_exact=False, cuda_update_drift_accepted=True)
    with pytest.raises(ValueError,match='without study approval'):validate_choice(result,cfg,data,init)
    cfg.runtime.allow_cuda_resume_update_drift=True
    result['benchmark_identity']=benchmark_identity(cfg,data,init)
    validate_choice(result,cfg,data,init)
    result['results'][0]['cuda_update_drift_accepted']=False
    with pytest.raises(ValueError,match='without study approval'):validate_choice(result,cfg,data,init)
    cfg.runtime.physical_batch=32
    with pytest.raises(ValueError,match='has not passed'):validate_choice(result,cfg,data,init)
    cfg.runtime.physical_batch=16
    cfg.optim.head_lr=2e-4
    with pytest.raises(ValueError,match='differs'):validate_choice(result,cfg,data,init)


def test_benchmark_resume_reuses_completed_candidates(tmp_path,monkeypatch):
    import scripts.benchmark_sem_cellstyle_a100 as cli
    import src.sem_cellstyle.data as data_module
    import src.sem_cellstyle.recipe as recipe_module
    import src.sem_cellstyle.benchmark as benchmark_module
    from types import SimpleNamespace
    monkeypatch.setattr(data_module,'provider',lambda cfg: (list(range(200)),None,{'count':200}))
    monkeypatch.setattr(recipe_module,'require_initialization',lambda cfg: tmp_path/'original')
    monkeypatch.setattr(benchmark_module,'benchmark_identity',lambda *args:'fixture')
    output=tmp_path/'bench';output.mkdir()
    (output/'benchmark_plan.json').write_text(json.dumps(dict(benchmark_identity='fixture',batch_sizes=[16,32],
        warmup_updates=1,measured_updates=2,num_workers=0)))
    row=dict(physical_batch=16,status='oom',finite=False)
    (output/'candidate_16.json').write_text(json.dumps(row))
    calls=[]
    def launch(command,check):
        calls.append(command)
        (output/'candidate_32.json').write_text(json.dumps(dict(physical_batch=32,status='oom',finite=False)))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(cli,'subprocess',SimpleNamespace(run=launch))
    cli.main(['--output',str(output),'--warmup-updates','1','--measured-updates','2','--num-workers','0','--resume'])
    assert len(calls)==1 and calls[0][calls[0].index('--candidate')+1]=='32'
    assert json.loads((output/'benchmark.json').read_text())['recommendation']['physical_batch'] is None
