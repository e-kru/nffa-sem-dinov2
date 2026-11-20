import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import ConcatDataset, Subset

from scripts import run_sem_ssl_v0_300epochs as run
from scripts.audit_sem_v0_decode import audit
from scripts.build_sem_posttraining_manifest import THRESHOLDS
from scripts.select_sem_v0_decode_verified import REASON, curate
from src.sem_cellstyle.recipe import schedules
from src.sem_corpus_v0.epochs import EpochLayout, full_schedule, train_to, verified_milestone
from src.sem_corpus_v0.preflight import (V0AugADataset, check_decode, load_training_image,
                                         overlap, validation_scope, verified_decode_audit)
from src.ssl.data import sha256
from src.ssl.loading import load_image

from test_sem_cellstyle import fixture


def test_300_epoch_schedule_uses_full_inventory_and_augs_a_proportions():
    cfg,layout = run.configuration(Path('/tmp/v0'),Path('training_manifest.csv'),{'identity':{'count':172301}})
    assert layout.updates==2693 and layout.tail==13
    assert cfg.experiment.schedule_total_steps==807900
    assert list(cfg.experiment.milestones)==[0,2693,13465,26930,67325,134650,269300,538600,807900]
    assert (cfg.experiment.warmup_steps,cfg.experiment.freeze_last_layer_steps,
            cfg.experiment.teacher_temp_warmup_steps)==(40395,8079,161580)
    assert layout.cursor(807900)['image_presentations']==51690300
    assert [len(x) for x in layout.batches(42,2692,2694)]==[7,6,32,32]
    schedule = schedules(cfg)
    assert max(schedule['backbone_lr'].schedule)==pytest.approx(1e-5)
    assert max(schedule['head_lr'].schedule)==pytest.approx(1e-4)
    assert schedule['teacher_temp'][0]==.04 and schedule['teacher_temp'][161579]==.07
    assert (cfg.crops.global_crops_size,cfg.crops.local_crops_size,cfg.crops.local_crops_number)==(224,98,8)


def test_v0_evaluation_uses_verified_nffa_paths_with_frozen_ssl_configuration():
    from scripts.evaluate_sem_encoder import parser

    manifest = Path('/tmp/v0/frozen/training_manifest.csv')
    cfg, _ = run.configuration(Path('/tmp/v0'), manifest, {'identity':{'count':172300}})
    for task, heads, workers in [('classification', ['logreg','mlp'],8), ('segmentation',['linear','cnn8'],16)]:
        command = run.pilot.evaluation_command(cfg, Path('teacher_backbone.pth'), Path('evaluation'),task)
        args = parser().parse_args(command[3:])
        assert args.task==task and args.heads==heads and args.workers==workers
        assert args.classification_manifest==Path('outputs/dataset_v3/manifest.csv')
        assert args.split_a==Path('outputs/dataset_v3/splits/split_A_exact.csv') and args.splits==['A']
        assert args.image_root.name=='nffa_sem_100' and args.verified_data_root and args.resume
    assert cfg.dataset.manifest==str(manifest) and 'primary_split' not in cfg.dataset


@pytest.mark.parametrize('originals',[63,64,65,66,67,97,129])
def test_epoch_tails_include_each_original_once_and_can_be_resumed(originals):
    layout=EpochLayout(originals,32)
    batches=list(layout.batches(42,0,2*layout.updates))
    assert all(2<=len(b)<=32 for b in batches)
    for epoch in range(2):
        first=epoch*2*layout.updates
        pairs=[pair for batch in batches[first:first+2*layout.updates] for pair in batch]
        assert sorted(i for i,_ in pairs)==list(range(originals))
        assert [draw for _,draw in pairs]==list(range(epoch*originals,(epoch+1)*originals))
    assert list(layout.batches(42,1,2*layout.updates))==batches[2:]


def test_duplicate_group_checks_include_similar_images_with_distinct_ids():
    rows=[dict(image_id='train',split='train',group_id='exact0'),
          dict(image_id='val',split='val',group_id='exact1')]
    assert overlap(rows,{'train'})['val']==dict(direct=0,by_group=0)
    assert overlap(rows,{'train'},{'train':'similar','val':'similar'})['val']==dict(direct=0,by_group=1)


def test_jlu_preserves_contrast_before_aug_a(tmp_path):
    path = tmp_path/'jlu.tif'
    Image.fromarray(np.array([[257, 1028], [32768, 65535]], dtype=np.uint16)).save(path)
    row = dict(path=path, source='jlu_cathodes', width=2, height=2)
    assert load_image(path).convert('L').getextrema() == (255, 255)
    rgb = np.asarray(load_training_image(row))
    assert rgb.tolist() == [[[1]*3, [4]*3], [[128]*3, [255]*3]]
    assert check_decode({'jlu_cathodes':[row]}) == dict(tested=1, failures=[], lost_contrast_examples=[])
    row['source'] = 'other'
    assert np.array_equal(np.asarray(load_training_image(row)), np.asarray(load_image(path)))


def test_pixel_decode_audit_rejects_truncated_png_with_valid_header(tmp_path):
    valid, broken = tmp_path/'valid.png', tmp_path/'broken.png'
    pixels = np.arange(128**2, dtype=np.uint16).reshape(128,128)
    Image.fromarray(pixels).save(valid)
    broken.write_bytes(valid.read_bytes()[:64])
    with Image.open(broken) as image:
        assert image.size == (128,128)
    manifest = tmp_path/'training_manifest.csv'
    records = [dict(image_id=p.stem,path=str(p),source='jlu_cathodes',width=128,height=128,
                    included='1',split='train',exclusion_reason='',
                    **{f'min_side_ge_{n}':int(n==128) for n in THRESHOLDS}) for p in (valid,broken)]
    with manifest.open('w',newline='') as stream:
        writer = csv.DictWriter(stream, records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    receipt = tmp_path/'full_decode_audit.json'
    result = audit(manifest, receipt, workers=2)
    assert result['checked']==2 and not result['transient_failures']
    assert [row['path'] for row in result['failures']]==[str(broken)]
    summary = {'selection_policies':{'full_decode_v1':{
        'audit_sha256':sha256(receipt), 'parent_training_manifest_sha256':sha256(manifest),
        'checked':2, 'excluded':1}}}
    assert verified_decode_audit(tmp_path,summary,{valid})['active']==1
    with pytest.raises(ValueError,match='Full pixel decode evidence'):
        verified_decode_audit(tmp_path,summary,{broken})
    data = V0AugADataset(records, {}, None)
    with pytest.raises(OSError, match='broken.png'):
        data[1]
    parent, revised = tmp_path/'parent', tmp_path/'revised'
    parent.mkdir()
    for filename in ('manifest.csv','training_manifest.csv'):
        (parent/filename).write_bytes(manifest.read_bytes())
    original_sha = sha256(manifest)
    parent_summary = dict(
        manifest_sha256=original_sha, training_manifest_sha256=original_sha,
        included=2, excluded=0, total_rows=2, exclusion_reasons={}, selection_policies={}, curation_history=[],
        sources={'jlu_cathodes': dict(included=2, excluded=0,
                                       **{f'included_ge_{n}':2*int(n==128) for n in THRESHOLDS})})
    for name, payload in [('manifest_summary',parent_summary),
                          ('receipt.draft',dict(training_manifest_sha256=original_sha)),
                          ('execution_policy',dict(manifest_sha256=original_sha))]:
        (parent/f'{name}.json').write_text(json.dumps(payload))
    (parent/'README.md').touch()
    curate(parent,revised,receipt)
    assert sha256(parent/'training_manifest.csv') == original_sha
    with (revised/'manifest.csv').open(newline='') as stream:
        updated = list(csv.DictReader(stream))
    assert [(row['included'],row['exclusion_reason']) for row in updated] == [('1',''),('0',REASON)]
    with (revised/'training_manifest.csv').open(newline='') as stream:
        assert list(csv.DictReader(stream)) == updated[:1]
    revised_summary = json.loads((revised/'manifest_summary.json').read_text())
    assert verified_decode_audit(revised,revised_summary,{valid})['active']==1


def test_reviewed_export_accepts_similarity_warnings_but_blocks_known_holdouts():
    rows = [dict(image_id='train', split='train', group_id='0'),
            dict(image_id='val', split='val', group_id='1')]
    exact = overlap(rows, {'train'})
    inclusive = overlap(rows, {'train', 'val'})
    leakage = dict(nffa_A_exact=exact,
                   nffa_A_known_similarity=overlap(rows, {'train'}, {'train':'same', 'val':'same'}),
                   nffa_B=inclusive, aachen=exact)
    blockers, warnings, splits = validation_scope(leakage, 'verified-policy-sha', None)
    assert not blockers and len(warnings) == 4 and splits == ['A']
    assert validation_scope(leakage, None, None)[0]
    assert validation_scope(dict(leakage, aachen=inclusive), 'verified-policy-sha', None)[0]
    assert validation_scope(dict(leakage, nffa_A_exact=inclusive), 'verified-policy-sha', None)[0]


def test_v0_epoch_checkpoint_resume_preserves_full_update_state(fixture,tmp_path):
    cfg,data,collate,_,initialization=fixture
    data=ConcatDataset([data,Subset(data,[0,1,2,3])])
    cfg.runtime.physical_batch=cfg.train.batch_size_per_gpu=3
    cfg.runtime.effective_batch=6
    cfg.experiment.milestones=[0,2,4]
    layout=EpochLayout(len(data),3)
    expected={'recipe':'test exact epochs'}
    uninterrupted, resumed=tmp_path/'continuous',tmp_path/'interrupted'
    train_to(cfg,data,collate,layout,initialization,expected,uninterrupted,4,device='cpu')
    train_to(cfg,data,collate,layout,initialization,expected,resumed,2,device='cpu')
    with pytest.raises(InterruptedError):
        train_to(cfg,data,collate,layout,initialization,expected,resumed,4,device='cpu',interrupt_after=3)
    train_to(cfg,data,collate,layout,initialization,expected,resumed,4,device='cpu')
    a,b=(p/'checkpoints/step_000004' for p in (uninterrupted,resumed))
    for folder in (a,b):
        verified_milestone(folder,expected,layout,sha256(initialization))
    before,after=(torch.load(p/'training.pth',map_location='cpu',weights_only=False) for p in (a,b))
    assert before['schedule']==after['schedule']
    assert before['data_cursor']==after['data_cursor']==layout.cursor(4)
    assert all(torch.equal(v,after['model'][k]) for k,v in before['model'].items())
    for key,value in before['optimizer']['state'].items():
        assert all(torch.equal(v,after['optimizer']['state'][key][k]) for k,v in value.items())
    rows=[json.loads(line) for line in (resumed/'metrics.jsonl').read_text().splitlines()]
    assert [row['original_presentations'] for row in rows]==[6,11,17,22]


def test_blocked_preflight_stops_before_any_gpu_work(tmp_path,monkeypatch):
    report=dict(identity={'count':172321},blockers=['known overlap'],ready=False)
    monkeypatch.setattr(run,'preflight',lambda *args:(Path('training_manifest.csv'),[],report))
    monkeypatch.setattr(run,'sha256',lambda _:run.PIN['backbone_sha256'])
    monkeypatch.setattr(run,'disk_plan',lambda *args:dict(required_free_bytes=1,observed_free_bytes=2))
    monkeypatch.setattr(run,'require_a100',lambda:pytest.fail('GPU started before the leakage gate'))
    with pytest.raises(SystemExit,match='known overlap'):
        run.main(['--output',str(tmp_path)])
    assert json.loads((tmp_path/'status.json').read_text())['status']=='preflight_blocked'
    assert not (tmp_path/'training').exists()


def test_orchestration_resumes_failed_evaluation_and_continues(tmp_path,monkeypatch):
    class Configuration:
        class experiment:
            epoch_to_step={'0':0,'1':2,'300':600}
    events=[]
    monkeypatch.setattr(run,'MILESTONE_EPOCHS',(0,1,300))
    monkeypatch.setattr(run.torch.cuda,'empty_cache',lambda:None)
    def train(cfg,data,collate,layout,init,expected,root,target):
        events.append(('train',target))
        folder=root/'checkpoints'/f'step_{target:06d}'
        folder.mkdir(parents=True)
        (folder/'complete.json').touch()
    def evaluate(cfg,path,output,reference=None):
        step=int(path.parent.name.rsplit('_',1)[1])
        events.append(('evaluate',step))
        if step==2 and events.count(('evaluate',2))==1:
            raise InterruptedError
        return dict(metrics={'test_fixture':1.},protocols={'same':'protocol'},bundle={})
    monkeypatch.setattr(run,'train_to',train)
    monkeypatch.setattr(run,'evaluate',evaluate)
    monkeypatch.setattr(run,'verified_milestone',lambda *args:{'verified':True})
    monkeypatch.setattr(run,'training_summary',lambda *args:{})
    monkeypatch.setattr(run,'large_reference',lambda *args:{'available':False})
    monkeypatch.setattr(run,'publish',lambda *args:None)
    args=(tmp_path,Configuration(),None,None,EpochLayout(11,3),None,None)
    with pytest.raises(InterruptedError):
        run.run_milestones(*args)
    run.run_milestones(*args)
    assert events==[('train',0),('evaluate',0),('train',2),('evaluate',2),
                    ('evaluate',2),('train',600),('evaluate',600)]
