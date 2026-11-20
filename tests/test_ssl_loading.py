"""Lossless cache and loader tuning must preserve every sampled Meta view/mask."""
import copy
import json
import pickle
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader

pytest.importorskip('omegaconf')
from src.ssl.upstream import ROOT, activate, checkout_path
if not checkout_path().is_dir():
    pytest.skip('Pinned Meta checkout unavailable', allow_module_level=True)
activate()
from src.ssl.data import SEMTrainDataset, augmentation_and_collate, sha256
from src.ssl.experiment import configuration
from src.ssl.loading import (DecodedCache, LoaderOptions, build_cache, cache_identity, configure_dataset,
                             estimate_cache, load_image, loader_kwargs, options)
from src.ssl.stream import CursorBatches, DrawCollate, DrawDataset
from scripts.benchmark_sem_ssl_loader import cases, compare


@pytest.fixture
def fixture(tmp_path):
    torch.set_num_threads(1)
    cfg=configuration(ROOT/'configs/sem_ssl/nffa_200step.yaml')
    cfg.train.batch_size_per_gpu=2;cfg.train.num_workers=0
    records=[]
    for i in range(4):
        path=tmp_path/f'{i}.png'
        pixels=np.random.default_rng(i).integers(0,180,(128,160,3),dtype=np.uint8)
        pixels[108:]=255
        Image.fromarray(pixels if i%2 else pixels[:,:,0]).save(path)
        records.append(dict(image_id=path.name,group_id=str(i),sha256=sha256(path),bytes=path.stat().st_size))
    identity=dict(provider='nffa_v3',root=str(tmp_path),manifests=[],records=records,count=4)
    return cfg,identity,tmp_path


def test_lossless_cache_matches_rgb_metadata_crop_and_is_spawn_small(fixture):
    cfg,identity,root=fixture
    result=build_cache(root/'cache',identity,max_bytes=1024**2,workers=2)
    cache=DecodedCache(root/'cache',identity,verify_blob=True)
    assert result['bytes']>0
    for row in identity['records']:
        expected=load_image(root/row['image_id'])
        actual=cache.image(row['image_id'])
        assert actual.mode=='RGB' and actual.size==expected.size
        np.testing.assert_array_equal(actual,expected)
    serialized=pickle.dumps(cache)
    assert len(serialized)<result['bytes']/4  # No decoded corpus serialized per worker.
    restored=pickle.loads(serialized)
    assert restored._mapping is None
    np.testing.assert_array_equal(restored.image('0.png'),cache.image('0.png'))


def test_budget_refuses_before_creating_cache_and_no_overwrite(fixture):
    _,identity,root=fixture
    with pytest.raises(MemoryError):
        build_cache(root/'limited',identity,max_bytes=1)
    assert not (root/'limited').exists()
    assert not list(root.glob('.pending_*'))
    build_cache(root/'cache',identity,max_bytes=1024**2)
    with pytest.raises(FileExistsError):
        build_cache(root/'cache',identity,max_bytes=1024**2)


def test_cache_rejects_data_order_preprocessing_and_corruption(fixture):
    _,identity,root=fixture
    build_cache(root/'cache',identity,max_bytes=1024**2)
    changed=copy.deepcopy(identity);changed['records'].reverse()
    with pytest.raises(ValueError,match='identity'):
        DecodedCache(root/'cache',changed)
    with mock.patch('src.ssl.loading.preprocessing_sha',return_value='changed'):
        with pytest.raises(ValueError,match='identity'):
            DecodedCache(root/'cache',identity)
    blob=root/'cache/pixels.bin'
    content=blob.read_bytes();blob.write_bytes(bytes([content[0]^1])+content[1:])
    with pytest.raises(ValueError,match='checksum'):
        DecodedCache(root/'cache',identity,verify_blob=True)
    blob.write_bytes(b'broken')
    with pytest.raises(ValueError,match='Truncated'):
        DecodedCache(root/'cache',identity)


def test_cache_identity_binds_full_source_hashes(fixture):
    _,identity,_=fixture
    altered=copy.deepcopy(identity);altered['records'][0]['sha256']='different'
    assert cache_identity(identity)!=cache_identity(altered)


def batches(cfg,identity,profile=False,cache=None):
    cfg.loader=dict(profile=profile,decoded_cache=str(cache) if cache else None)
    transform,collate=augmentation_and_collate(cfg)
    dataset=configure_dataset(cfg,SEMTrainDataset(Path(identity['root']),identity['records'],transform),identity)
    return list(DataLoader(DrawDataset(dataset,42),batch_sampler=CursorBatches(4,2,42,0,3),
        collate_fn=DrawCollate(collate,42),num_workers=0,generator=torch.Generator().manual_seed(42)))


def test_original_views_and_masks_bitwise_equal_with_cache_and_profiling(fixture):
    cfg,identity,root=fixture
    build_cache(root/'cache',identity,max_bytes=1024**2)
    raw=batches(cfg,identity)
    profiled=batches(cfg,identity,profile=True)
    cached=batches(cfg,identity,profile=True,cache=root/'cache')
    for before,diagnostic,after in zip(raw,profiled,cached):
        for key,value in before.items():
            if isinstance(value,torch.Tensor):
                assert torch.equal(value,diagnostic[key]),key
                assert torch.equal(value,after[key]),key
            else:
                assert value==after[key]
        for name in ('crop_global_seconds','crop_local_seconds','color_seconds','blur_solarize_seconds',
                     'normalize_seconds','collate_seconds'):
            assert diagnostic['loader_profile'][name]>=0
        assert 'cache_fetch_seconds' in after['loader_profile']
        assert 'metadata_seconds' not in after['loader_profile']
    # Same source can be sampled repeatedly; augmented views are freshly random.
    assert not torch.equal(cached[0]['collated_global_crops'],cached[2]['collated_global_crops'])


def test_profiling_read_decode_path_is_pixel_identical(fixture):
    _,identity,root=fixture
    for record in identity['records']:
        profile={}
        np.testing.assert_array_equal(load_image(root/record['image_id']),load_image(root/record['image_id'],profile))
        assert set(profile)=={'read_seconds','decode_rgb_seconds','metadata_seconds'}


def test_loader_fallback_and_options(fixture):
    cfg,_,_=fixture
    assert options(cfg)==LoaderOptions()
    assert loader_kwargs(cfg,'cpu')==dict(num_workers=0,pin_memory=False)
    cfg.train.num_workers=8
    kwargs=loader_kwargs(cfg,'cuda')
    assert kwargs['multiprocessing_context']=='spawn' and kwargs['prefetch_factor']==2
    assert not kwargs['persistent_workers'] and kwargs['pin_memory']
    cfg.loader=dict(persistent_workers=True,prefetch_factor=4,pin_memory=False,non_blocking=False)
    kwargs=loader_kwargs(cfg,'cuda')
    assert kwargs['persistent_workers'] and kwargs['prefetch_factor']==4 and not kwargs['pin_memory']
    assert not options(cfg).non_blocking
    cfg.loader.prefetch_factor=0
    with pytest.raises(ValueError):options(cfg)


def test_full_loader_ablation_and_comparison_do_not_invent_gpu_improvement():
    plan=cases([8,12,16],cache=Path('/tmp/cache'),full=True)
    assert {r['workers'] for r in plan}=={8,12,16}
    assert any(not r['pin'] for r in plan) and any(not r['non_blocking'] for r in plan)
    assert any(r['prefetch']==4 for r in plan) and any(r['persistent'] for r in plan)
    rows=[dict(case='raw_w8',batch_size=64,status='ok',images_per_second=100,pss_bytes=1000),
          dict(case='decoded_w8',batch_size=64,status='ok',images_per_second=120,pss_bytes=1300),
          dict(case='raw_w16',batch_size=64,status='error')]
    compare(rows)
    assert rows[1]['improvement_percent']==pytest.approx(20)
    assert rows[1]['additional_pss_bytes']==300
    assert rows[2]['improvement_percent'] is None


def test_failed_build_cleans_up_only_its_partial_directory(fixture):
    _,identity,root=fixture
    unrelated=root/'preserve';unrelated.mkdir();(unrelated/'data').write_text('keep')
    with mock.patch('src.ssl.loading.load_image',side_effect=RuntimeError('decode failed')):
        with pytest.raises(RuntimeError):
            build_cache(root/'cache',identity,max_bytes=1024**2)
    assert (unrelated/'data').read_text()=='keep'
    assert not list(root.glob('.pending_*'))


def test_shared_filesystem_cache_is_rejected(fixture):
    _,identity,root=fixture
    with mock.patch('src.ssl.loading.filesystem',return_value='nfs4'):
        with pytest.raises(ValueError,match='node-local'):
            build_cache(root/'network-cache',identity,max_bytes=1024**2)
    assert not (root/'network-cache').exists()


def test_cli_can_explicitly_disable_cache_and_instrumentation(fixture):
    from scripts.benchmark_sem_ssl_a100 import parser
    from src.ssl.loading import apply_arguments
    cfg,_,_=fixture
    cfg.loader=dict(decoded_cache='/old/cache',profile=True,persistent_workers=True)
    args=parser().parse_args(['--decoded-cache','none','--no-profile-loader','--no-persistent-workers'])
    result=apply_arguments(cfg,args)
    assert result.decoded_cache is None and not result.profile and not result.persistent_workers


def test_transfer_event_hook_and_blocking_option_leave_original_step_usable(fixture):
    from src.ssl.adapter import SingleGPUAdapter,train_step
    cfg,identity,_=fixture
    cfg.crops.global_crops_size,cfg.crops.local_crops_size,cfg.crops.local_crops_number=56,28,2
    cfg.ibot.mask_ratio_min_max=[.4,.5]
    cfg.dino.head_n_prototypes,cfg.dino.head_hidden_dim,cfg.dino.head_bottleneck_dim=32,64,16
    batch=batches(cfg,identity)[0]
    model=SingleGPUAdapter(cfg,None,allow_random_init=True)
    optimizer=torch.optim.AdamW(model.parameter_groups())
    events=[]
    class Event:
        def __init__(self,name):self.name=name
        def record(self):events.append(self.name)
    schedule=dict(lr=[1e-5],last_layer_lr=[1e-5],weight_decay=[.04],momentum=[.996],teacher_temp=[.04])
    result=train_step(model,optimizer,batch,schedule,0,diagnostics=True,defer_metrics=True,
                      non_blocking=False,transfer_events=(Event('start'),Event('end')))
    assert events==['start','end'] and torch.isfinite(result['total_loss'])
    assert all(p.grad is None for p in model.teacher.parameters())
