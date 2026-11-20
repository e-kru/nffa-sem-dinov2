"""Fully decode every active V0 image on CPU and record unreadable files."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

from PIL import Image

from scripts.build_sem_posttraining_manifest import atomic_json, metadata_sha, rows


def decode(row):
    try:
        with Image.open(row['path']) as image:
            if image.size != (int(row['width']), int(row['height'])):
                raise ValueError(f'Unexpected image dimensions: {image.size}')
            image.load()
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
        return {key:row[key] for key in ('image_id','source','path','width','height')} | {
            'error_type': type(error).__name__, 'error': str(error)}
    return None


def audit(manifest, output, workers=8):
    fingerprint = metadata_sha(manifest)
    counts, failed = Counter(), []
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        batch = []
        checked = 0
        for row in rows(manifest):
            if row['included'] != '1' or row['split'] != 'train':
                raise ValueError('Expected an active training export')
            batch.append(row)
            counts[row['source']] += 1
            if len(batch) == 512:
                failed.extend(error for error in pool.map(decode,batch) if error is not None)
                checked += len(batch)
                batch.clear()
                if checked % 8192 == 0:
                    print(f'{checked:,} decoded, {len(failed)} failures, {time.monotonic()-start:.0f}s',flush=True)
        failed.extend(error for error in pool.map(decode,batch) if error is not None)
    if not counts or metadata_sha(manifest) != fingerprint:
        raise ValueError('Training manifest is empty or changed during the decode audit')
    # An intermittent storage read requires diagnosis rather than exclusion.
    confirmed, transient = [], []
    for row in failed:
        repeated = decode(row)
        if repeated and repeated == row:
            confirmed.append(row)
        else:
            transient.append(row | {'retest':repeated})
    result = dict(schema='sem-v0-full-pillow-decode-v1', manifest=str(manifest.resolve()),
                  manifest_sha256=fingerprint, checked=sum(counts.values()), source_counts=dict(counts),
                  failures=confirmed, transient_failures=transient, seconds=time.monotonic()-start)
    atomic_json(output,result)
    print(f'{result["checked"]:,} decoded, {len(failed)} failures. Results: {output}',flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    audit(args.manifest,args.output,args.workers)


if __name__ == '__main__':
    main()
