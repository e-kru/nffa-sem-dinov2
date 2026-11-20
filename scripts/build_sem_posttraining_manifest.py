"""Build a reviewable SEM corpus after audits finish; never starts training.

Only audit tables and existing split receipts are reused. NFFA train headers are
read without pixel decoding or image hashes; Aachen sizes come from its receipt.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import time
import uuid

THRESHOLDS = (128, 224, 256, 448, 512, 518)
FIELDS = ('image_id', 'path', 'source', 'original_id', 'specimen_id', 'split',
          'included', 'exclusion_reason', 'original_split', 'role', 'role_evidence',
          'modality', 'width', 'height', 'mode', 'channels', 'header_status',
          'resolution_class', *(f'min_side_ge_{n}' for n in THRESHOLDS),
          'group_evidence', 'existing_sha256', 'audit_source', 'cleaning_status')


def metadata_sha(path):
    """Metadata files only. This function is never called on image paths."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.partial')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    temporary.replace(path)


def rows(path):
    with Path(path).open(newline='') as stream:
        yield from csv.DictReader(stream)


def truth(value):
    return str(value).lower() in ('1', 'true')


def safe_path(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Image path must stay inside its dataset root')
    return Path(root).absolute()/relative


def audit_status(directory):
    directory = Path(directory)
    summary, progress = directory/'audit_summary.json', directory/'progress.json'
    runner = directory.parent/'run_status.json'
    if runner.is_file() and read_json(runner)['status'] == 'failed':
        raise ValueError('Source audit failed: '+str(directory))
    if not summary.is_file() or not progress.is_file():
        return None
    value = read_json(summary)
    if read_json(progress).get('stage') not in ('complete', 'incomplete'):
        return None
    if value.get('discovery_complete') is not True or value.get('header_pending') != 0:
        raise ValueError('Source audit has incomplete discovery/headers: '+str(directory))
    # Unreadable headers can be recorded and excluded; discovery gaps cannot.
    if not (directory/'images_manifest.csv').is_file():
        raise ValueError('Source audit is missing its image table')
    return value


def validate_groups(assignments):
    seen, partitions = set(), {}
    for row in assignments:
        if row['image_id'] in seen or row['split'] not in ('train', 'val', 'test') or not row.get('group_id'):
            raise ValueError('Invalid or duplicate group split row')
        seen.add(row['image_id'])
        previous = partitions.setdefault(row['group_id'], row['split'])
        if previous != row['split']:
            raise ValueError('A benchmark group crosses train/validation/test')


def header_inspector(path):
    # Reuse the frozen audit implementation, not a separate decoder or hash path.
    spec = importlib.util.spec_from_file_location('_sem_manifest_audit', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.Image.MAX_IMAGE_PIXELS = None
    return module


def nffa_rows(config):
    cfg = config['nffa']
    inventory = read_json(cfg['inventory'])
    if Path(inventory['root']).absolute() != Path(cfg['root']).absolute():
        raise ValueError('NFFA inventory root differs')
    if canonical(inventory['records']) != inventory['records_sha256'] or len(inventory['records']) != inventory['count']:
        raise ValueError('NFFA inventory records changed')
    fingerprints = {item['path']: item['sha256'] for item in inventory['manifests']}
    for path in (cfg['manifest'], cfg['split']):
        if fingerprints.get(str(Path(path).absolute())) != metadata_sha(path):
            raise ValueError('NFFA manifest/split differs from the existing receipt')
    curated = {row['image_id']: row for row in rows(cfg['manifest'])}
    assignments = list(rows(cfg['split']))
    validate_groups(assignments)
    selected = {row['image_id']: row for row in assignments
                if row['split'] == 'train' and curated[row['image_id']]['included'] == '1'}
    if set(selected) != {row['image_id'] for row in inventory['records']}:
        raise ValueError('NFFA inventory is not exactly the included Split A training set')
    inspector = header_inspector(config['header_inspector'])

    def inspect(record):
        assignment = selected[record['image_id']]
        if assignment['group_id'] != record['group_id']:
            raise ValueError('NFFA training group differs from inventory')
        if curated[record['image_id']].get('crop_mode') != 'crop_v2':
            raise ValueError('NFFA requires its existing crop_v2 protocol')
        path = safe_path(cfg['root'], record['image_id'])
        value = dict(dataset='nffa_sem_100', path=str(path), relative_path=record['image_id'],
                     is_microscopy=True, role='microscopy_image', role_evidence='existing included NFFA Split A train inventory',
                     modality='SEM', original_group=record['group_id'], sample_group='unknown',
                     group_evidence='existing Split A exact group; physical specimen identity unverified', split_hint='train',
                     sha256=record.get('sha256', ''), hash_status='skipped_by_request')
        try:
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode):
                raise ValueError('Not a regular training image')
            value.update(zip(('size_bytes', 'mtime_ns', 'ctime_ns', 'device', 'inode'), inspector.fingerprint(info)))
            value.update(inspector.inspect_header(value))
        except (OSError, ValueError) as error:
            value.update(header_status='read_error', width='', height='', mode='', channels='', read_error=str(error))
        return value

    with ThreadPoolExecutor(max_workers=config.get('header_workers', 2)) as pool:
        yield from pool.map(inspect, inventory['records'])


def aachen_rows(config):
    cfg = config['aachen']
    saved = read_json(cfg['reference'])
    contract = saved['cache_metadata']['contract']
    if contract.get('nonscientific_smoke_test') is not False or contract['manifest_sha256'] != metadata_sha(cfg['manifest']):
        raise ValueError('Aachen split differs from the verified downstream receipt')
    assignments = list(rows(cfg['manifest']))
    validate_groups(assignments)
    selected = {row['image_id']: row for row in assignments if row['split'] == 'train'}
    verified = {row['image_id']: row for row in contract['records'] if row['split'] == 'train'}
    if set(verified) != set(selected):
        raise ValueError('Aachen receipt is not the exact existing training set')
    for image_id, record in selected.items():
        known = verified[image_id]
        if any(known.get(key) != value for key, value in record.items()):
            raise ValueError('Aachen training metadata differs from receipt')
        relative = Path(record['image_path'])
        path = (Path(cfg['manifest']).absolute().parent/relative).resolve() if not relative.is_absolute() else relative
        if not path.is_relative_to(Path(cfg['root']).absolute()/'images'):
            raise ValueError('Aachen path is not in its image branch')
        width, height = known['original_size']
        yield dict(dataset='aachen_heerlen', path=str(path), relative_path=str(path.relative_to(cfg['root'])),
                   is_microscopy=True, role='microscopy_image', role_evidence='verified group-aware Aachen training image; mask never opened',
                   modality='SEM', original_group=image_id, sample_group=record['group_id'],
                   group_evidence='existing downstream training group; no new split', split_hint='train',
                   header_status='ok', width=width, height=height, mode='not_recorded_in_receipt', channels='',
                   sha256=known.get('image_sha256', ''))


def manifest_row(raw, audit_source):
    source = raw['dataset']
    width, height = int(raw.get('width') or 0), int(raw.get('height') or 0)
    readable = raw.get('header_status') == 'ok' and width > 0 and height > 0
    minimum = min(width, height) if readable else 0
    flags = {f'min_side_ge_{n}': int(readable and minimum >= n) for n in THRESHOLDS}
    reasons = []
    if source == 'lithos':
        reasons.append('excluded_source_lithos')
    if source == 'fields_metal_cement' and not Path(raw['path']).name.startswith('Rot0_'):
        reasons.append('excluded_fields_non_rot0')
    if raw.get('header_status') != 'ok':
        reasons.append('header_not_readable')
    if not truth(raw.get('is_microscopy', False)) or raw.get('role') not in ('microscopy_image', 'derived_image'):
        reasons.append('not_a_microscopy_image_or_unresolved_role')
    if readable and minimum < 128:
        reasons.append('original_min_side_below_128')
    if source in ('nffa_sem_100', 'aachen_heerlen') and raw.get('split_hint') != 'train':
        raise ValueError('Held-out benchmark row reached the training manifest builder')
    resolution_class = ('unreadable' if not readable else 'ge518' if minimum >= 518 else
                        '512_517' if minimum >= 512 else '256_511' if minimum >= 256 else
                        '128_255' if minimum >= 128 else 'lt128')
    return dict(image_id=source+':'+raw['relative_path'], path=raw['path'], source=source,
                original_id=raw.get('original_group') or 'unknown', specimen_id=raw.get('sample_group') or 'unknown',
                split='train', included=int(not reasons), exclusion_reason=';'.join(reasons),
                original_split=raw.get('split_hint', 'unknown'), role=raw.get('role', 'unknown'),
                role_evidence=raw.get('role_evidence', ''), modality=raw.get('modality', 'unknown'),
                width=width if readable else '', height=height if readable else '', mode=raw.get('mode', ''),
                channels=raw.get('channels', ''), header_status=raw.get('header_status', ''),
                resolution_class=resolution_class, **flags, group_evidence=raw.get('group_evidence', 'unknown'),
                existing_sha256=raw.get('sha256') or '', audit_source=str(audit_source), cleaning_status='pending')


def build(config, output):
    if config.get('minimum_side', 128) != 128 or config.get('training_resolution', 224) not in (224, 518):
        raise ValueError('V0 requires original minimum 128 and training resolution 224 or 518')
    output = Path(output).absolute()
    if output.exists():
        raise FileExistsError('Use a new output version; existing manifests are never overwritten')
    audits = [(Path(p), audit_status(p)) for p in config['audit_directories']]
    if any(summary is None for _, summary in audits):
        raise ValueError('Source audits are not finished')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent/('.'+output.name+'.building_'+uuid.uuid4().hex)
    staging.mkdir()
    counts, reasons, sources, seen, seen_ids = Counter(), Counter(), {}, set(), set()
    provenance = []

    def inputs():
        source_roots = set()
        for directory, summary in audits:
            metadata = summary['contract']
            datasets = set(summary['datasets'])
            if source_roots & datasets or datasets & {'nffa_sem_100', 'aachen_heerlen'}:
                raise ValueError('Overlapping audits or benchmark images in source audit')
            source_roots.update(datasets)
            table = directory/'images_manifest.csv'
            n = 0
            for raw in rows(table):
                if raw['dataset'] not in datasets:
                    raise ValueError('Audit row outside its declared source scope')
                expected = safe_path(Path(metadata['root'])/raw['dataset'], raw['relative_path'])
                if str(expected) != raw['path']:
                    raise ValueError('Audit path differs from its dataset root')
                n += 1
                yield raw, table
            if n != summary['image_files']:
                raise ValueError('Audit table row count differs from its final summary')
            provenance.append(dict(path=str(directory), images=n, summary_sha256=metadata_sha(directory/'audit_summary.json'),
                                   image_table_sha256=metadata_sha(table), unreadable_headers=summary.get('read_errors', 0)))
        for raw in nffa_rows(config):
            yield raw, config['nffa']['inventory']
        for raw in aachen_rows(config):
            yield raw, config['aachen']['reference']

    with (staging/'manifest.csv').open('w', newline='') as master, (staging/'training_manifest.csv').open('w', newline='') as train:
        all_writer, train_writer = csv.DictWriter(master, FIELDS), csv.DictWriter(train, FIELDS)
        all_writer.writeheader(); train_writer.writeheader()
        for raw, source_table in inputs():
            row = manifest_row(raw, source_table)
            if row['path'] in seen or row['image_id'] in seen_ids:
                raise ValueError('Duplicate absolute path or image ID in corpus inputs')
            seen.add(row['path']); seen_ids.add(row['image_id'])
            all_writer.writerow(row)
            if row['included']:
                train_writer.writerow(row)
            source = sources.setdefault(row['source'], dict(total_rows=0, included=0, excluded=0,
                          **{f'included_ge_{n}': 0 for n in THRESHOLDS}))
            source['total_rows'] += 1
            source['included'] += row['included']; source['excluded'] += not row['included']
            for n in THRESHOLDS:
                source[f'included_ge_{n}'] += row['included']*row[f'min_side_ge_{n}']
            counts['total_rows'] += 1; counts['included'] += row['included']; counts['excluded'] += not row['included']
            reasons.update(filter(None, row['exclusion_reason'].split(';')))
    summary = dict(schema='sem-posttraining-candidate-manifest-v0', status='candidate_cleaning_pending',
                   training_started=False, **counts, sources=sources, exclusion_reasons=dict(reasons),
                   minimum_original_side=128, resolution_basis='original_header_before_metadata_strip',
                   training_resolution=config.get('training_resolution', 224), supported_training_resolutions=[224, 518],
                   color_filter_applied=False, image_contents_rehashed=False, independent_original_count_verified=False,
                   nffa_split='A_exact_train_only', nffa_split_B_protected=False, aachen_split='existing_group_train_only',
                   lithos_excluded=True, selection_policies={'fields_metal_cement': 'Rot0_only'},
                   curation_pending=['color_content', 'visual_quality', 'cross_source_leakage',
                                                        'group_and_variant_sampling', 'full_pixel_decode'],
                   input_audits=provenance,
                   sources_orientation={name:dict(orientation_invariant=False) for name in sources},
                   manifest_sha256=metadata_sha(staging/'manifest.csv'),
                   training_manifest_sha256=metadata_sha(staging/'training_manifest.csv'),
                   configuration=config)
    atomic_json(staging/'manifest_summary.json', summary)
    # Deliberately not a verified_sem_manifest_v1 receipt: cleaning/group and
    # cross-source holdout reviews have not been performed or attested.
    atomic_json(staging/'receipt.draft.json', dict(schema='sem-posttraining-candidate-receipt-v0',
                 status='candidate_cleaning_pending', training_ready=False, candidate_training_images=counts['included'],
                 training_manifest_sha256=summary['training_manifest_sha256'],
                 all_aachen_excluded=False, aachen_train_only=True, nffa_split_A_train_only=True,
                 content_hash_deduplication_complete=False))
    (staging/'README.md').write_text(
        '# SEM-Posttraining-Kandidatenmanifest V0\n\n'
        '`manifest.csv` enthält alle erfassten Bildrollen mit editierbarem `included` (0/1) und Ausschlussgrund.\n'
        '`training_manifest.csv` ist der initiale Export der eingeschlossenen Zeilen.\n'
        'Originalauflösungen und Größenflags sind unverändert; mindestens 128 Pixel je Seite.\n'
        'LITHOS ist immer ausgeschlossen. NFFA nur Split A Train, Aachen nur bestehendes Group-Train.\n'
        'Fields Metal Cement: nur Rot0_; andere Rotationsvarianten bleiben mit included=0 im Gesamtinventar.\n'
        'Farbprüfung und weiteres Cleaning sind offen. Keine neue Bildhash- oder Duplikatprüfung.\n'
        'Die Bildzahl ist keine bestätigte Zahl unabhängiger Originalaufnahmen.\n'
        '224 ist die V0-Transformationsausgabe, 518 eine Option; keine Bilder wurden umgeschrieben.\n'
        'Manuelle Änderungen an included erfordern beim späteren Cleaning einen neuen Trainingsexport.\n'
        'Keine Konfiguration wird automatisch aktiviert und kein Training gestartet.\n')
    os.rename(staging, output)
    return summary


def wait_and_build(config, output, status_path, poll_seconds=30, sleep=time.sleep):
    while True:
        pending = [p for p in config['audit_directories'] if audit_status(p) is None]
        if not pending:
            break
        value = dict(status='waiting_for_audits', pending_audits=pending, output=str(output), training_started=False)
        atomic_json(status_path, value); print(json.dumps(value), flush=True)
        sleep(poll_seconds)
    atomic_json(status_path, dict(status='building_manifest', output=str(output), training_started=False))
    summary = build(config, output)
    value = dict(status='complete', output=str(output), included=summary['included'],
                 excluded=summary['excluded'], training_started=False)
    atomic_json(status_path, value); print(json.dumps(value), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--poll-seconds', type=float, default=30)
    parser.add_argument('--status', type=Path)
    args = parser.parse_args(argv)
    if args.poll_seconds <= 0 or args.watch and args.status is None:
        parser.error('--watch requires --status; poll seconds must be positive')
    config = read_json(args.config)
    try:
        if args.watch:
            wait_and_build(config, args.output, args.status, args.poll_seconds)
        else:
            print(json.dumps(build(config, args.output), indent=2))
    except Exception as error:
        if args.status:
            atomic_json(args.status, dict(status='failed', error=type(error).__name__+': '+str(error),
                                         output=str(args.output), training_started=False))
        raise


if __name__ == '__main__':
    main()
