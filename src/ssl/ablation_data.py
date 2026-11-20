"""Reuse already audited corpus metadata, without reopening files to hash pixels.

Only explicit callers use this path. The original audit remains the default.
Manifest fingerprints are checked; the image corpus must remain unchanged.
"""
from collections import Counter
import copy
import json
from pathlib import Path

from src.aachen_ablation.data import sha256_file


def classification(reference, manifest, split_file, image_root):
    saved = json.loads(Path(reference).read_text())
    contract = saved['contract']
    if (saved.get('test_files_opened') is not False or contract['nonscientific_smoke_test']
            or contract['manifest_sha256'] != sha256_file(manifest)
            or contract['split_sha256'] != sha256_file(split_file)
            or contract['image_root'] != str(Path(image_root).resolve())):
        raise ValueError('Verified classification inventory does not match manifest/split/root')
    records = contract['records']
    if (len({r['image_id'] for r in records}) != len(records)
            or any(r['split'] not in ('train', 'val') or not r.get('image_sha256') for r in records)
            or dict(Counter(r['split'] for r in records)) != saved['counts']):
        raise ValueError('Invalid verified classification records')
    return copy.deepcopy(records)


def nffa_training_inventory(reference, manifest, split_file, image_root):
    """Derive a train-only SSL inventory from already hashed classification data."""
    from src.ssl.data import train_records
    from src.ssl.experiment import digest

    reference, manifest, split_file = map(Path, (reference, manifest, split_file))
    verified = classification(reference, manifest, split_file, image_root)
    verified = {row['image_id']: row for row in verified if row['split'] == 'train'}
    selected = train_records(manifest, split_file)
    records = []
    for row in selected:
        known = verified.get(row['image_id'])
        if known is None or known['group_id'] != row['group_id']:
            raise ValueError('Verified classification data does not match the selected training groups')
        records.append({**row, 'sha256': known['image_sha256']})
    return dict(provider='nffa_v3', root=str(Path(image_root).resolve()), count=len(records),
                manifests=[dict(path=str(p.resolve()), sha256=sha256_file(p)) for p in (manifest, split_file)],
                records=records, records_sha256=digest(records),
                selection='primary split train only; included crop_v2; no additional split intersection',
                provenance=dict(verified_classification=str(reference.resolve()),
                                verified_classification_sha256=sha256_file(reference),
                                image_contents_rehashed=False))


def segmentation(reference, manifest, rows, variants):
    from src.aachen_ablation.geometry import content_box, regions
    saved = json.loads(Path(reference).read_text())
    contract = saved['cache_metadata']['contract']
    if (contract['nonscientific_smoke_test']
            or contract['manifest_sha256'] != sha256_file(manifest)):
        raise ValueError('Verified segmentation inventory does not match manifest')
    records = contract['records']
    if (len(records) != len(rows) or any(
            r['split'] not in ('train', 'val') or not r.get('image_sha256') or not r.get('mask_sha256')
            or any(r.get(k) != v for k, v in row.items()) for r, row in zip(records, rows))):
        raise ValueError('Verified segmentation rows differ from the group manifest')
    sizes, padded = Counter(), Counter()
    for record in records:
        w, h = record['original_size']
        sizes[f'{w}x{h}'] += 1
        for variant in variants:
            for x0, y0, x1, y1 in regions(w, h, variant.quadrants):
                if content_box(x1-x0, y1-y0, variant) != (0, 0, variant.width, variant.height):
                    padded[variant.name] += 1
    return copy.deepcopy(records), dict(counts=dict(Counter(r['split'] for r in records)),
        original_sizes=dict(sizes), letterboxed_regions=dict(padded), test_files_opened=False,
        historical_baseline_comparable=False, reused_verified_metadata=str(Path(reference).resolve()))


def segmentation_reference(encoder_root):
    paths = list((Path(encoder_root)/'segmentation/runs/A3/linear').glob('seed42_*/config.json'))
    if len(paths) != 1:
        raise ValueError('Expected exactly one verified baseline A3 linear configuration')
    return paths[0]
