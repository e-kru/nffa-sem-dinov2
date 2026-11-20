"""Bounded real-image CPU loader profile. GPU end-to-end comparisons stay separate."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '1')
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['XFORMERS_DISABLED'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader
from src.ssl.checkpoints import atomic_json
from src.ssl.data import SEMTrainDataset, augmentation_and_collate, seed_worker
from src.ssl.experiment import ROOT, configuration, dataset_plan, digest
from src.ssl.loading import (build_cache, configure_dataset, effective_cpus, estimate_cache,
                             loader_kwargs, memory_info, process_memory)
from src.ssl.stream import CursorBatches, DrawCollate, DrawDataset
from src.ssl_benchmark import aggregate, dataset_from_inventory


def measure(cfg, dataset, collate, batch, workers, warmup, steps):
    cfg.train.num_workers = workers
    sampler = CursorBatches(len(dataset), batch, cfg.train.seed, 0, warmup+steps)
    loader = DataLoader(DrawDataset(dataset, cfg.train.seed), batch_sampler=sampler,
        collate_fn=DrawCollate(collate,cfg.train.seed), worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(cfg.train.seed), **loader_kwargs(cfg,'cpu'))
    iterator = None
    before = process_memory([os.getpid()])
    try:
        iterator = iter(loader)
        for _ in range(warmup):
            next(iterator)
        pids = [os.getpid(), *[w.pid for w in getattr(iterator,'_workers',[])]]
        waits, profiles = [], []
        begun = time.perf_counter()
        for _ in range(steps):
            started = time.perf_counter()
            item = next(iterator)
            waits.append(time.perf_counter()-started)
            profiles.append(item['loader_profile'])
        elapsed = time.perf_counter()-begun
        totals = {key:sum(r.get(key,0.) for r in profiles) for key in set().union(*profiles)}
        return dict(batch_size=batch,workers=workers,steps=steps,images_per_second=batch*steps/elapsed,
                    wall_seconds=elapsed,loader_wait_seconds=aggregate(waits),
                    stage_ms_per_image={key:1000*value/(batch*steps) for key,value in totals.items() if key.endswith('_seconds')},
                    group_memory_before=before,group_memory_after=process_memory(pids))
    finally:
        if iterator is not None and hasattr(iterator,'_shutdown_workers'):
            iterator._shutdown_workers()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/sem_ssl/nffa_200step.yaml')
    p.add_argument('--inventory',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cache-dir',type=Path,required=True,help='New node-local cache for this CPU sample only')
    p.add_argument('--reuse-cache',action='store_true',help='Verify and reuse the same completed sample cache')
    p.add_argument('--images',type=int,default=128)
    p.add_argument('--batch-sizes',type=int,nargs='+',default=[32,64])
    p.add_argument('--worker-counts',type=int,nargs='+',default=[8,12,16])
    p.add_argument('--warmup-batches',type=int,default=4)
    p.add_argument('--steps',type=int,default=128)
    p.add_argument('--prefetch-factor',type=int,default=2)
    p.add_argument('--max-cache-gb',type=float,default=2)
    args=p.parse_args()
    if args.output.exists() or (args.cache_dir.exists() and not args.reuse_cache):
        raise FileExistsError('Profile output and sample-cache directory must be new')
    if min(args.images,args.steps,args.prefetch_factor)<1 or args.images<max(args.batch_sizes) or min(args.worker_counts)<0:
        p.error('Invalid sample/batch/step/worker configuration')
    if any(w>0 and w<=effective_cpus() for w in args.worker_counts):
        from multiprocessing.connection import Listener
        # Fail immediately on sandbox-denied tensor IPC, instead of a queue hang.
        with Listener(authkey=b'sem-ssl-cpu-profile'):
            pass
    torch.set_num_threads(1)
    cfg=configuration(args.config)
    if 'loader' in cfg: cfg.loader={}
    if args.inventory:
        full_identity=json.loads(args.inventory.read_text());dataset_from_inventory(cfg,full_identity)
    else:
        _,_,full_identity=dataset_plan(cfg,8)
    order=torch.randperm(full_identity['count'],generator=torch.Generator().manual_seed(cfg.train.seed)).tolist()[:args.images]
    identity={**full_identity,'records':[full_identity['records'][i] for i in order],'count':len(order)}
    identity['records_sha256']=digest(identity['records'])
    args.output.mkdir(parents=True)
    atomic_json(args.output/'sample_inventory.json',identity)
    print('Cache estimate:',estimate_cache(identity['root'],identity['records']),flush=True)
    if args.reuse_cache:
        from src.ssl.loading import DecodedCache
        cache=DecodedCache(args.cache_dir,identity,verify_blob=True).metadata
    else:
        cache=build_cache(args.cache_dir,identity,max_bytes=int(args.max_cache_gb*1024**3),workers=8)
    results=[];skipped=[]
    for mode in ('raw','decoded'):
        for workers in args.worker_counts:
            if workers>effective_cpus():
                skipped.append(dict(mode=mode,workers=workers,reason=f'Only {effective_cpus()} CPUs available'));continue
            for batch in args.batch_sizes:
                cfg.loader=dict(profile=True,prefetch_factor=args.prefetch_factor,
                                decoded_cache=str(args.cache_dir.resolve()) if mode=='decoded' else None)
                cfg.train.num_workers=workers
                transform,collate=augmentation_and_collate(cfg)
                dataset=configure_dataset(cfg,SEMTrainDataset(Path(identity['root']),identity['records'],transform),identity)
                row=measure(cfg,dataset,collate,batch,workers,args.warmup_batches,args.steps)
                row['mode']=mode;results.append(row)
                print(json.dumps(row),flush=True)
                atomic_json(args.output/'profile.json',dict(cpu_only=True,augmented_views_cached=False,
                    full_train_count=full_identity['count'],profile_sample_count=len(order),
                    cache_bytes=cache['bytes'],results=results,skipped=skipped,
                    cpus=effective_cpus(),memory=memory_info(),source='Real train-only images; instrumented CPU diagnostic, not GPU throughput'))
    result=json.loads((args.output/'profile.json').read_text())
    result['skipped']=skipped
    atomic_json(args.output/'profile.json',result)


if __name__=='__main__':
    main()
