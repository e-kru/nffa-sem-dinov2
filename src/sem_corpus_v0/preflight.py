"""Read-only checks for the frozen V0 export and the benchmark split metadata."""

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
from pathlib import Path
from random import Random
import stat

import numpy as np
from PIL import Image

from src.preprocessing_v2 import crop_metadata_strip
from src.sem_cellstyle.data import CorpusDataset, SEMViews, collate_for
from src.ssl.data import sha256
from src.ssl.loading import load_image
from src.ssl.upstream import ROOT


MANIFEST_DIR = ROOT / 'outputs/sem_corpus/sem_posttraining_v0_decode_verified'
SPLIT_B = ROOT / 'outputs/dataset_v3/splits/split_B_full.csv'
BENCHMARKS = {'nffa_A', 'aachen_heerlen'}


def read(path):
    return json.loads(Path(path).read_text())


def csv_rows(path):
    with Path(path).open(newline='') as stream:
        yield from csv.DictReader(stream)


def overlap(records, active_ids, similarity=None):
    """Count direct overlaps and existing exact/similarity groups per partition."""
    by_id = {row['image_id']: row for row in records}
    if len(by_id) != len(records) or not active_ids <= by_id.keys():
        raise ValueError('Split table has duplicate IDs or does not cover the active benchmark rows')
    group = similarity if similarity is not None else {key: row['group_id'] for key, row in by_id.items()}
    if not by_id.keys() <= group.keys():
        raise ValueError('Duplicate-group metadata does not cover the split')
    active_groups = {group[key] for key in active_ids}
    return {
        split: dict(
            direct=sum(row['image_id'] in active_ids for row in records if row['split'] == split),
            by_group=sum(group[row['image_id']] in active_groups for row in records if row['split'] == split),
        )
        for split in ('train', 'val', 'test')
    }


def has_holdouts(table):
    return any(table[split][key] for split in ('val', 'test') for key in ('direct', 'by_group'))


def load_training_image(row):
    path = Path(row['path'])
    if row['source'] != 'jlu_cathodes':
        return load_image(path)
    with Image.open(path) as source:
        if source.mode.startswith('I;16'):
            # Scale the full sensor range before Pillow converts to RGB.
            pixels = np.asarray(source, dtype=np.uint32)
            gray = ((pixels + 128) // 257).astype(np.uint8)
            image = Image.fromarray(gray).convert('RGB')
        else:
            image = source.convert('RGB')
    return crop_metadata_strip(image)


class V0AugADataset(CorpusDataset):
    def __getitem__(self, index):
        row = self.records[index]
        try:
            image = load_training_image(row)
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            raise OSError(f'Cannot decode V0 image {row["image_id"]} at {row["path"]}: {error}') from error
        return self.transform(image, False), ()


def check_files(records):
    def check_chunk(chunk):
        bad = []
        for row in chunk:
            path = row['path']
            try:
                info = os.stat(path)
                if not stat.S_ISREG(info.st_mode) or not info.st_size or not os.access(path, os.R_OK):
                    bad.append((path, 'not a readable nonempty regular file'))
            except OSError as error:
                bad.append((path, str(error)))
        return bad

    chunks = (records[i:i+1024] for i in range(0, len(records), 1024))
    with ThreadPoolExecutor(max_workers=8) as pool:
        failures = [item for bad in pool.map(check_chunk, chunks) for item in bad]
    return dict(checked=len(records), failed=len(failures), examples=failures[:20])


def check_decode(by_source):
    rng = Random(42)
    failures, contrast_warnings, tested = [], [], 0
    for source, records in by_source.items():
        for row in rng.sample(records, min(3, len(records))):
            tested += 1
            path = Path(row['path'])
            try:
                with Image.open(path) as image:
                    if image.size != (int(row['width']), int(row['height'])):
                        raise ValueError('Image header differs from frozen dimensions')
                    if image.mode.startswith(('I', 'F')):
                        raw_min, raw_max = image.getextrema()
                    else:
                        raw_min = raw_max = 0
                image = load_training_image(row).convert('L')
                lower, upper = image.getextrema()
                if raw_min != raw_max and lower == upper:
                    contrast_warnings.append(dict(source=source, path=str(path),
                        raw_range=[raw_min, raw_max], aug_a_gray_range=[lower, upper]))
            except (OSError, ValueError, Image.DecompressionBombError) as error:
                failures.append(dict(source=source, path=str(path),
                                     error_type=type(error).__name__, error=str(error)))
    return dict(tested=tested, failures=failures, lost_contrast_examples=contrast_warnings)


def verified_decode_audit(directory, summary, paths):
    selection = summary['selection_policies'].get('full_decode_v1')
    if selection is None:
        return None
    path = directory/'full_decode_audit.json'
    digest = sha256(path)
    audit = read(path)
    failed = {Path(row['path']) for row in audit['failures']}
    if (digest != selection['audit_sha256'] or audit['schema'] != 'sem-v0-full-pillow-decode-v1'
            or audit['manifest_sha256'] != selection['parent_training_manifest_sha256']
            or audit['checked'] != selection['checked'] or audit['transient_failures']
            or len(failed) != selection['excluded'] or len(failed) != len(audit['failures'])
            or len(paths) + len(failed) != audit['checked'] or failed & paths):
        raise ValueError('Full pixel decode evidence differs from the revised export')
    return dict(sha256=digest, checked=audit['checked'], excluded=len(failed), active=len(paths))


def verified_policy(directory, manifest_sha, count, leakage, split_sha, decode_audit=None):
    path = directory/'execution_policy.json'
    if not path.is_file():
        return None
    policy = read(path)
    evidence = directory/'reviewed_preflight.json'
    if (policy.get('schema') != 'sem_v0_train_only_decode_trim20_v1'
            or policy.get('manifest_sha256') != manifest_sha
            or policy.get('nffa_training') != 'split_A_train'
            or policy.get('nffa_validation') != 'split_A_exact_groups'
            or policy.get('aachen') != 'train_groups_only'
            or policy.get('cross_source_leakage_review') != 'incomplete_operator_accepted'
            or policy.get('full_decode_audit_sha256') != (decode_audit['sha256'] if decode_audit else None)
            or policy.get('reviewed_preflight_sha256') != sha256(evidence)):
        raise ValueError('The approved V0 policy or its evidence differs from this export')
    previous = read(evidence)
    if (previous['identity']['count'] - count != 20 + (decode_audit['excluded'] if decode_audit else 0)
            or previous['identity']['benchmark_split_sha256'] != split_sha
            or previous['benchmark_overlap'] != leakage):
        raise ValueError('The V0 benchmark splits or overlap counts changed after approval')
    return sha256(path)


def validation_scope(leakage, policy, review):
    blockers, warnings = [], []
    a_exact = has_holdouts(leakage['nffa_A_exact'])
    a_similarity = has_holdouts(leakage['nffa_A_known_similarity'])
    if a_exact:
        blockers.append('NFFA Split A has held-out images in active exact-duplicate groups or SSL rows')
    if a_similarity:
        message = 'NFFA Split A has held-out images in known similarity groups of active SSL rows'
        (warnings if policy else blockers).append(message)
    if has_holdouts(leakage['aachen']):
        blockers.append('Aachen has held-out images or groups in SSL')
    aachen_train = leakage['aachen']['train']['direct']
    if aachen_train:
        message = f'Aachen contributes {aachen_train} SSL train images; validation measures in-domain transfer'
        (warnings if policy else blockers).append(message)
    if review is None:
        message = 'Cross-source benchmark leakage review is incomplete; the full image-hash audit was cancelled'
        (warnings if policy else blockers).append(message)
    reviewed = set(review['reviewed_benchmarks']) if review else set()
    safe_splits = []
    if not a_exact and (policy or ('nffa_A' in reviewed and not a_similarity)):
        safe_splits.append('A')
    if has_holdouts(leakage['nffa_B']):
        warnings.append('NFFA Split B has held-out overlap and is excluded from evaluation')
    elif not policy and 'nffa_B' in reviewed:
        safe_splits.append('B')
    return blockers, warnings, safe_splits


def preflight(directory, review_path=None):
    directory = Path(directory).resolve()
    if directory == MANIFEST_DIR and not directory.is_dir():
        raise ValueError('The full-decode V0 revision is pending. Finish scripts/prepare_sem_ssl_v0_decode_verified.sh first.')
    manifest = directory / 'training_manifest.csv'
    summary_file, draft_file = directory / 'manifest_summary.json', directory / 'receipt.draft.json'
    summary, draft = read(summary_file), read(draft_file)
    manifest_sha = sha256(manifest)
    if manifest_sha != summary['training_manifest_sha256'] or manifest_sha != draft['training_manifest_sha256']:
        raise ValueError('V0 export differs from the saved summary or receipt')
    counts, paths, ids = Counter(), set(), set()
    resolutions, by_source, origins = defaultdict(Counter), defaultdict(list), defaultdict(set)
    input_splits = defaultdict(Counter)
    expected_audits = {str(Path(item['path']) / 'images_manifest.csv') for item in summary['input_audits']}
    refs = summary['configuration']
    expected_audits.update((refs['nffa']['inventory'], refs['aachen']['reference']))
    if any(not Path(path).is_file() for path in expected_audits):
        raise ValueError('Missing saved provenance for the frozen V0 input')
    records, oversized = [], Counter()
    for row in csv_rows(manifest):
        name, path, image_id = row['source'], Path(row['path']), row['image_id']
        if (row['included'] != '1' or row['split'] != 'train' or row['exclusion_reason']
                or row['role'] not in ('microscopy_image', 'derived_image') or row['header_status'] != 'ok'
                or not path.is_absolute() or '..' in path.parts or name == 'lithos'):
            raise ValueError(f'Ineligible frozen training row: {image_id}')
        roots = refs['nffa']['root'] if name == 'nffa_sem_100' else (
            refs['aachen']['root'] if name == 'aachen_heerlen' else Path(refs['nffa']['root']).parent/name)
        if not path.is_relative_to(roots) or row['audit_source'] not in expected_audits:
            raise ValueError(f'Missing provenance or path outside its source: {image_id}')
        if path in paths or image_id in ids:
            raise ValueError(f'Duplicate path or image ID: {image_id}')
        if min(int(row['width']), int(row['height'])) < 128:
            raise ValueError(f'Invalid original resolution: {image_id}')
        if Image.MAX_IMAGE_PIXELS and int(row['width'])*int(row['height']) > 2*Image.MAX_IMAGE_PIXELS:
            oversized[name] += 1
        paths.add(path)
        ids.add(image_id)
        counts[name] += 1
        input_splits[name][row['original_split']] += 1
        resolutions[name][f"{row['width']}x{row['height']}"] += 1
        origins[name].add(row['audit_source'])
        by_source[name].append(row)
        records.append(row)
    if (len(records) != summary['included'] or len(records) != draft['candidate_training_images']
            or dict(counts) != {name: v['included'] for name,v in summary['sources'].items() if v['included']}):
        raise ValueError('Actual V0 count or source allocation differs from the saved summary')
    for name, definition in (('nffa_sem_100', refs['nffa']), ('aachen_heerlen', refs['aachen'])):
        if any(row['original_split'] != 'train' for row in by_source[name]):
            raise ValueError(f'Non-train original split in {name}')
    nffa_ids = {Path(row['path']).relative_to(refs['nffa']['root']).as_posix() for row in by_source['nffa_sem_100']}
    aachen_ids = {Path(row['path']).name for row in by_source['aachen_heerlen']}
    nffa_a_path, aachen_path = map(Path, (refs['nffa']['split'], refs['aachen']['manifest']))
    a, b, ac = (list(csv_rows(path)) for path in (nffa_a_path, SPLIT_B, aachen_path))
    b_groups = {row['image_id']: row['group_id'] for row in b}
    leakage = dict(nffa_A_exact=overlap(a, nffa_ids), nffa_A_known_similarity=overlap(a, nffa_ids, b_groups),
                   nffa_B=overlap(b, nffa_ids), aachen=overlap(ac, aachen_ids))
    split_sha = {key:sha256(path) for key,path in
                 (('nffa_A', nffa_a_path), ('nffa_B', SPLIT_B), ('aachen', aachen_path))}
    full_decode_audit = verified_decode_audit(directory, summary, paths)
    if directory == MANIFEST_DIR and full_decode_audit is None:
        raise ValueError('The default V0 revision requires a complete pixel-decode audit')
    policy = verified_policy(directory, manifest_sha, len(records), leakage, split_sha, full_decode_audit)
    if policy and nffa_ids != {row['image_id'] for row in a if row['split'] == 'train'}:
        raise ValueError('The approved V0 revision requires all NFFA Split A Train images')

    audit_details = []
    for prior in summary['input_audits']:
        path = Path(prior['path'])
        record = path / 'audit_summary.json'
        digest = sha256(record)
        if digest != prior['summary_sha256']:
            raise ValueError(f'Source audit summary changed: {path}')
        receipt = read(record)
        if not (receipt['discovery_complete'] and receipt['headers_complete']):
            raise ValueError(f'Incomplete source audit: {path}')
        audit_details.append(dict(path=str(path), sha256=digest, image_table_sha256=prior['image_table_sha256'],
                                  exact_hash_check_complete=receipt['exact_hash_check_complete'],
                                  hash_check_cancelled_by_user=receipt['hash_check_cancelled_by_user']))
    review = None
    if review_path is not None:
        review = read(review_path)
        if (review['manifest_sha256'] != manifest_sha or not review['evidence']
                or not BENCHMARKS <= set(review['reviewed_benchmarks']) or review['cross_source_leakage_resolved'] is not True):
            raise ValueError('Leakage review must document the exact export and protected benchmarks')
    blockers, warnings, safe_splits = validation_scope(leakage, policy, review)
    print(f'{len(records)} active rows, sampled decoding and access checks across {len(counts)} sources', flush=True)
    files = check_files(records)
    decodes = check_decode(by_source)
    if files['failed']:
        blockers.append(f'{files["failed"]} frozen inputs failed the filesystem access check')
    if decodes['failures']:
        blockers.append('One or more additional frozen inputs failed sample decoding')
    if oversized:
        blockers.append(f'{sum(oversized.values())} frozen rows exceed Pillow’s hard decoding pixel limit')
    if decodes['lost_contrast_examples']:
        blockers.append('Sampled high-bit-depth images lose all contrast in the required Aug-A preprocessing')
    if sha256(manifest) != manifest_sha:
        raise ValueError('V0 export changed during preflight')
    scope = dict(nffa_splits=safe_splits, aachen_ssl_train_images=len(aachen_ids),
                 cross_source_review='reviewed' if review else ('incomplete_operator_accepted' if policy else 'incomplete'),
                 warnings=warnings, benchmark_overlap=leakage)
    identity = dict(
        manifest_sha256=manifest_sha, count=len(records),
        manifest_summary_sha256=sha256(summary_file), draft_receipt_sha256=sha256(draft_file),
        source_counts=dict(counts), sampling_policy=summary['selection_policies'],
        benchmark_split_sha256=split_sha, execution_policy_sha256=policy,
        validation_scope=scope, jlu_loader='fixed_full_scale_16bit_to_8bit_v1',
        full_decode_audit=full_decode_audit,
        source_audits=audit_details, leakage_review_sha256=sha256(Path(review_path)) if review else None,
    )
    report = dict(identity=identity, source_provenance={name:sorted(values) for name,values in origins.items()},
                  source_sampling_fractions={name:value/len(records) for name,value in counts.items()},
                  source_original_split_counts={name:dict(values) for name,values in input_splits.items()},
                  resolutions={name:dict(values) for name,values in resolutions.items()},
                  original_receipt_status=draft['status'], benchmark_overlap=leakage,
                  allowed_nffa_validation_splits=safe_splits, file_access=files, sampled_decode=decodes,
                  above_decode_pixel_limit=dict(oversized),
                  blockers=blockers, warnings=warnings, ready=not blockers)
    return manifest, records, report


def aug_a_dataset(cfg, records):
    # The requested experiment uses exactly Aug-A, including on small originals.
    return V0AugADataset(records, {}, SEMViews(cfg)), collate_for(cfg)
