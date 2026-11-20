"""Build/estimate/verify a lossless decoded train-image cache; CPU only."""
import argparse
import json
import os
from pathlib import Path
import sys

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '1')
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['XFORMERS_DISABLED'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ssl.checkpoints import atomic_json
from src.ssl.experiment import ROOT, configuration, dataset_plan
from src.ssl.loading import DecodedCache, build_cache, estimate_cache, filesystem, memory_info
from src.ssl_benchmark import dataset_from_inventory


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT/'configs/sem_ssl/nffa_200step.yaml')
    p.add_argument('--inventory', type=Path, help='Reuse an audited benchmark inventory; metadata/selection rechecked')
    p.add_argument('--cache-dir', type=Path, required=True, help='Node-local path, e.g. /dev/shm/sem_ssl_nffa_rgb_v1')
    p.add_argument('--max-cache-gb', type=float, default=8)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--estimate-only', action='store_true')
    p.add_argument('--verify', action='store_true', help='Validate complete cache and checksum every decoded byte')
    args = p.parse_args()
    if args.workers < 1 or args.max_cache_gb <= 0:
        p.error('workers and cache budget must be positive')
    cfg = configuration(args.config)
    # Cache preparation always reads the audited originals.
    if 'loader' in cfg:
        cfg.loader = {}
    if args.inventory:
        identity = json.loads(args.inventory.read_text())
        dataset_from_inventory(cfg, identity)
    else:
        _, _, identity = dataset_plan(cfg, args.workers)
    if args.verify:
        cache = DecodedCache(args.cache_dir, identity, verify_blob=True)
        print(json.dumps(dict(passed=True, bytes=cache.metadata['bytes'], images=len(cache.entries)), indent=2))
    elif args.estimate_only:
        result = estimate_cache(identity['root'], identity['records'], args.workers)
        print(json.dumps(dict(**result, memory=memory_info(), filesystem=filesystem(args.cache_dir)), indent=2))
    else:
        result = build_cache(args.cache_dir, identity, max_bytes=int(args.max_cache_gb*1024**3), workers=args.workers)
        print(json.dumps({k:v for k,v in result.items() if k != 'images'}, indent=2))


if __name__ == '__main__':
    main()
