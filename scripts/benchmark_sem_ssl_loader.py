"""Explicit, sequential full-SSL A100 loader ablations at B32/B64; no training launch."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys

for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key,'1')
os.environ['XFORMERS_DISABLED']='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from src.ssl.checkpoints import atomic_json
from src.ssl.experiment import ROOT, configuration, dataset_plan
from src.ssl.loading import effective_cpus, memory_info
from src.ssl_benchmark import dataset_from_inventory, subprocess_case


def cases(worker_counts, cache=None, full=False):
    result=[]
    for workers in worker_counts:
        result.append(dict(name=f'raw_w{workers}',workers=workers,cache=None,persistent=False,prefetch=2,pin=True,non_blocking=True))
    if full:
        base=dict(workers=8,cache=None,persistent=False,prefetch=2,pin=True,non_blocking=True)
        for name,changes in [('persistent',dict(persistent=True)),('prefetch4',dict(persistent=True,prefetch=4)),
                             ('unpinned',dict(pin=False)),('blocking',dict(non_blocking=False))]:
            result.append({**base,'name':name,**changes})
    if cache:
        for workers in worker_counts:
            result.append(dict(name=f'decoded_w{workers}',workers=workers,cache=str(cache),persistent=False,prefetch=2,pin=True,non_blocking=True))
        if full:
            result.append(dict(name='decoded_prefetch4',workers=8,cache=str(cache),persistent=True,prefetch=4,pin=True,non_blocking=True))
    return result


def compare(results):
    baselines={r['batch_size']:r for r in results if r['case']=='raw_w8' and r['status']=='ok'}
    for row in results:
        baseline=baselines.get(row['batch_size'])
        row['improvement_percent']=(100*(row['images_per_second']/baseline['images_per_second']-1)
                                    if baseline and row['status']=='ok' else None)
        row['additional_pss_bytes']=(row['pss_bytes']-baseline['pss_bytes']
                                     if baseline and row.get('pss_bytes') is not None and baseline.get('pss_bytes') is not None else None)
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/sem_ssl/nffa_200step.yaml')
    p.add_argument('--inventory',type=Path)
    p.add_argument('--decoded-cache',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worker-counts',type=int,nargs='+',default=[8,12,16])
    p.add_argument('--batch-sizes',type=int,nargs='+',default=[32,64])
    p.add_argument('--warmup-steps',type=int,default=10)
    p.add_argument('--measured-steps',type=int,default=100)
    p.add_argument('--full-ablation',action='store_true')
    p.add_argument('--dry-run',action='store_true',help='Print commands/resource decisions; no cache audit or GPU calls')
    args=p.parse_args()
    if 8 not in args.worker_counts or len(set(args.worker_counts))!=len(args.worker_counts) or min(args.worker_counts)<1:
        p.error('Choose unique positive worker counts including baseline 8')
    if not 5<=args.warmup_steps<=10 or args.measured_steps<1 or min(args.batch_sizes)<2:
        p.error('Use 5–10 warmup steps, positive measurements and physical batch >=2')
    if args.output.exists():raise FileExistsError(args.output)
    output=args.output.resolve();config=args.config.resolve()
    inventory_path=output/'inventory.json'
    planned=cases(args.worker_counts,args.decoded_cache.resolve() if args.decoded_cache else None,args.full_ablation)
    skipped=[];commands=[]
    cpus=effective_cpus()
    for case in planned:
        queue_bytes=max(args.batch_sizes)*2126208*case['workers']*case['prefetch']*2
        if case['workers']>cpus or queue_bytes>memory_info()['MemAvailable']*.8:
            skipped.append(dict(case=case['name'],reason=f'Resources: {cpus} CPUs; conservative pageable+pinned queue estimate {queue_bytes} bytes'));continue
        command=[sys.executable,str(ROOT/'scripts/benchmark_sem_ssl_a100.py'),'--config',str(config),
            '--inventory',str(inventory_path),'--output',str(output/case['name']),
            '--batch-sizes',*map(str,args.batch_sizes),'--num-workers',str(case['workers']),
            '--warmup-steps',str(args.warmup_steps),'--measured-steps',str(args.measured_steps),
            '--prefetch-factor',str(case['prefetch']), '--persistent-workers' if case['persistent'] else '--no-persistent-workers',
            '--pin-memory' if case['pin'] else '--no-pin-memory','--non-blocking' if case['non_blocking'] else '--no-non-blocking']
        command+=['--decoded-cache',case['cache'] or 'none','--no-profile-loader']
        commands.append((case,command))
    if args.dry_run:
        import shlex
        print(json.dumps(dict(cpus=cpus,skipped=skipped),indent=2))
        for _,command in commands:print(shlex.join(command))
        return
    if not commands:
        raise ValueError('No requested case fits the available CPU/RAM resources')
    output.mkdir(parents=True)
    cfg=configuration(config)
    if args.inventory:
        identity=json.loads(args.inventory.read_text());dataset_from_inventory(cfg,identity)
    else:
        _,_,identity=dataset_plan(cfg,8)
    atomic_json(inventory_path,identity)
    results=[]
    for case,command in commands:
        code=subprocess_case(command,output/(case['name']+'.log'))
        report=output/case['name']/'benchmark.json'
        if report.is_file():
            for r in json.loads(report.read_text())['results']:
                results.append(dict(case=case['name'],batch_size=r['batch_size'],status=r['status'],
                    images_per_second=r.get('originals_per_second'),loader_wait_seconds=r.get('dataloader_wait_seconds'),
                    loader_wait_mean_seconds=r.get('dataloader_wait_seconds',{}).get('mean'),
                    loader_wait_p95_seconds=r.get('dataloader_wait_seconds',{}).get('p95'),
                    p95_to_median=r.get('p95_to_median'),
                    pss_bytes=r.get('host_memory',{}).get('pss_bytes'),cache_bytes=r.get('decoded_cache_bytes',0),
                    peak_gpu_reserved_bytes=r.get('peak_reserved_bytes')))
        else:
            results.extend(dict(case=case['name'],batch_size=b,status='error',error=f'Exit {code}; see case log') for b in args.batch_sizes)
        compare(results)
        atomic_json(output/'comparison.json',dict(results=results,skipped=skipped,baseline='raw_w8; same long measurement window',gpu_benchmark=True))
    columns=['case','batch_size','status','images_per_second','improvement_percent',
             'loader_wait_mean_seconds','loader_wait_p95_seconds','p95_to_median',
             'pss_bytes','additional_pss_bytes','cache_bytes','peak_gpu_reserved_bytes']
    with (output/'comparison.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(results)


if __name__=='__main__':main()
