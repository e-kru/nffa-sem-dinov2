"""Read-only, resumable header inventory and size-filtered SHA-256 audit.

Only explicitly selected, allowed datasets are traversed. SQLite stores discovery,
header and hash receipts; CSV/Markdown exports can be rebuilt without raw reads.
No model, GPU, pixel decoding, archive extraction or dataset mutation is used.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import sqlite3
import stat
import sys
import time

from PIL import Image

from src.config import DATASETS_ROOT


DATASETS = ('lithos', 'cross_microscope_steel', 'cigrocksem', 'uhcsdb', 'microal',
            'jlu_cathodes', 'bil_nmc811', 'bil_si_graphite', 'kit_sem_powder')
ADDITIONAL_DATASETS = ('drill_cuttings_sem', 'fields_metal_cement',
                       'porous_zirconia_fibsem', 'rwth_steel_sem', 'matscifig_sem')
EXCLUDED = ('nffa_sem_100', 'aachen_heerlen')
IMAGE_SUFFIXES = {
    '.png', '.jpg', '.jpeg', '.jpe', '.jfif', '.tif', '.tiff', '.bmp', '.dib',
    '.gif', '.webp', '.jp2', '.j2k', '.jpf', '.jpx', '.pnm', '.pbm', '.pgm',
    '.ppm', '.pcx', '.tga', '.ico', '.im', '.psd', '.sgi', '.rgb', '.rgba',
    '.bw', '.ras', '.mpo', '.dds', '.pcd', '.qoi', '.avif', '.heic', '.heif',
    '.svs', '.ndpi', '.dcm', '.dm3', '.dm4', '.mrc',
}
ROOT = DATASETS_ROOT
SCHEMA = 1
BOOL_COLUMNS = {'is_image', 'is_microscopy', 'resolution_eligible', 'ssl_candidate'}
FILE_COLUMNS = (
    'dataset', 'path', 'relative_path', 'suffix', 'size_bytes', 'mtime_ns', 'ctime_ns',
    'device', 'inode', 'is_image', 'role', 'role_evidence', 'is_microscopy',
    'modality', 'sample_group', 'original_group', 'slice_id', 'tile_id',
    'group_evidence', 'split_hint', 'header_status', 'format', 'width', 'height',
    'mode', 'channels', 'resolution_eligible', 'resolution_reason',
    'resolution_class', 'ssl_candidate', 'read_error', 'header_seconds',
    'sha256', 'hash_status', 'hash_error', 'hash_seconds',
)


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.partial')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    temporary.replace(path)


def words(value):
    return set(re.findall(r'[a-z0-9]+', str(value).lower()))


def classify(dataset, relative):
    """Path evidence only; image-like names and grayscale mode are not proof."""
    path = Path(relative)
    parts = [p.lower() for p in path.parts]
    tokens = words(relative)
    # This documented image subdirectory sits inside Labels; the other Labels
    # subdirectories remain targets and must never become microscopy candidates.
    if (dataset == 'drill_cuttings_sem' and parts[:3] == ['raw', 'labels', 'rawimage_downscaled']
            and re.fullmatch(r'TS_\d+_downscaled', path.stem, re.I)):
        return 'derived_image', True, 'explicit RawImage_downscaled path; downscaled SEM source image, not a label'
    if any(re.search(r'(^|[_ -])(masks?|ground[_ -]?truth|gt)([_ -]|$)', p)
           for p in parts):
        return 'segmentation_mask', False, 'explicit mask/ground-truth path or filename'
    if any(re.search(r'(^|[_ -])(labels?|annotations?|targets?)([_ -]|$)', p)
           for p in parts):
        return 'label_or_annotation', False, 'explicit label/annotation path; target semantics unverified'
    if tokens & {'overlay', 'overlays', 'preview', 'previews', 'thumbnail', 'thumbnails',
                 'montage', 'figures', 'figure', 'plots', 'plot', 'logo', 'logos'} or 'qa' in parts:
        return 'auxiliary_image', False, 'preview/overlay/figure context; not a canonical micrograph'
    if dataset == 'drill_cuttings_sem':
        if (parts[:2] == ['raw', 'splits'] and
                re.fullmatch(r'TS_\d+_\d+(?:_\d+){4}', path.stem, re.I)):
            return 'derived_image', True, 'SEM cutout naming under published splits; original groups and CONTAM/TOUCH quality need review'
    elif dataset == 'fields_metal_cement':
        if 'sem_flyash_cement_mortar' in parts and re.fullmatch(r'Rot\d+_.+', path.stem, re.I):
            return 'derived_image', True, 'SEM directory and rotation-prefixed variant; variants are not independent originals'
    elif dataset == 'porous_zirconia_fibsem':
        if parts[0] == 'raw' and re.fullmatch(r'Slice_\d+', path.stem, re.I):
            return 'microscopy_image', True, 'FIB-SEM stack slice path; adjacent slices are not independent specimens'
    elif dataset == 'rwth_steel_sem':
        if parts[0] == 'hr_images' and 'hr' in parts:
            return 'derived_image', True, 'SEM super-resolution high-resolution image branch; original/crop mapping unverified'
    elif dataset == 'matscifig_sem':
        return 'unknown', False, 'requires exact local manifest join and subtype_valid=True; figure/panel content not pixel-validated'
    if dataset == 'kit_sem_powder':
        if 'phenom_images' in parts or re.search(r'_(?:BO|RL|Phenom)$', path.stem, re.I):
            return 'unknown', False, 'KIT BO/RL/Phenom role not established; no automatic SSL eligibility'
        tile = re.fullmatch(r'Tile_r\d+_c\d+|Region\d+_y\d+_x\d+_fw.+', path.stem, re.I)
        if tile and 'sem_segmentation_ssl' in parts:
            return 'derived_image', True, 'SEM acquisition tile/region naming; likely microscopy, physical origin unverified'
        return 'unknown', False, 'KIT image role not established from known path rules'
    if dataset == 'cross_microscope_steel' and re.fullmatch(r'\d+_[A-Za-z]+_(?:Bainite|Martensite)_\d+', path.stem, re.I):
        return 'derived_image', True, 'microscopy patch naming corroborated by Dataset_Patches_Microscope_Metadata.xlsx'
    known = False
    if dataset == 'lithos':
        known = bool(tokens & {'ppl', 'xpl'})
    elif dataset == 'cigrocksem':
        known = 'imgs' in parts or 'images' in parts
    elif dataset == 'uhcsdb':
        known = 'micrographs' in parts
    elif dataset == 'microal':
        known = bool(tokens & {'image', 'images', 'img', 'imgs'}) and bool(tokens & {'electron', 'metallo'})
    elif dataset == 'jlu_cathodes':
        known = bool(tokens & {'sem'}) and bool(tokens & {'se', 'bse'})
    elif dataset == 'bil_nmc811':
        known = 'sem_images' in parts
    elif dataset == 'bil_si_graphite':
        known = bool(tokens & {'bse', 'etd', 't2', 'electron'})
    if known:
        derived = bool(tokens & {'crop', 'crops', 'patch', 'patches', 'tile', 'tiles', 'resized', 'resizeto'})
        if dataset == 'cigrocksem' and re.search(r'_(?:crop|resizeTo|top|bottom|up|down)(?:_|\b)', path.stem):
            derived = True
        return ('derived_image' if derived else 'microscopy_image'), True, 'dataset-specific image path/name; likely microscopy, no pixel validation'
    return 'unknown', False, 'no sufficiently clear image/mask role in path; no automatic SSL eligibility'


def identifiers(dataset, relative):
    p = Path(relative)
    tokens = words(relative)
    modality = 'unknown'
    variants = [x for x in ('HRTEM', 'TEM', 'PPL', 'XPL', 'BSE', 'SE', 'SEM', 'ETD', 'T2', 'RDBS') if x.lower() in tokens]
    if variants:
        modality = ';'.join(variants)
    elif dataset in ('cigrocksem', 'uhcsdb', 'kit_sem_powder', 'bil_nmc811',
                     'drill_cuttings_sem', 'fields_metal_cement', 'rwth_steel_sem'):
        modality = 'SEM (dataset/path context)'
    elif dataset == 'porous_zirconia_fibsem':
        modality = 'FIB-SEM (dataset/path context)'
    split_names = {'train': 'train', 'training_set': 'train', 'val': 'val',
                   'validation': 'validation', 'test': 'test', 'test_set': 'test'}
    row = dict(modality=modality, sample_group='unknown', original_group='unknown',
               slice_id='unknown', tile_id='unknown', group_evidence='unknown',
               split_hint=next((split_names[x.lower()] for x in p.parts if x.lower() in split_names), 'unknown'))
    if dataset == 'lithos' and tokens & {'ppl', 'xpl'}:
        row.update(original_group=str(p.parent), group_evidence='observed paired PPL/XPL leaf directory; sample/original hierarchy unverified')
    elif dataset == 'cross_microscope_steel':
        match = re.fullmatch(r'(\d+)_(.+_(?:Bainite|Martensite)_\d+)', p.stem, re.I)
        if match:
            row.update(original_group=match[2], tile_id=match[1], group_evidence='filename parent/patch candidate, not a verified specimen identifier')
    elif dataset == 'jlu_cathodes':
        sample = re.search(r'(BM\d+)_SEM_', relative)
        sl = re.search(r'(?:BSE|SE)-(\d+)$', p.stem)
        if sample:
            row.update(sample_group=sample[1], group_evidence='observed BM series and SE/BSE slice numbering; physical sample mapping unverified')
        if sample and sl:
            row.update(original_group=sample[1]+'/slice-'+sl[1], slice_id=sl[1])
    elif dataset == 'kit_sem_powder':
        match = re.fullmatch(r'Tile_r(\d+)_c(\d+)', p.stem, re.I)
        region = re.fullmatch(r'(Region\d+)_y(\d+)_x(\d+)_fw(.+)', p.stem, re.I)
        if match or region:
            # Include acquisition directory: a tile basename is not globally unique.
            row.update(original_group=str(p.parent), tile_id=p.stem,
                       group_evidence='observed acquisition directory/tile coordinates; no claim of independent original exposures')
    elif dataset == 'cigrocksem':
        sample = re.match(r'(plug_[A-Za-z0-9]+)_image_', p.stem)
        if sample:
            row['sample_group'] = sample[1]
        original = re.split(r'_(?:crop|resizeTo|top|bottom|up|down)(?:_|\b)', p.stem, maxsplit=1)[0]
        if original != p.stem:
            row.update(original_group=str(p.parent/original), tile_id=p.stem[len(original):].strip('_'),
                       group_evidence='filename crop/resize suffix candidate; not proof of pixel correspondence')
    elif dataset == 'drill_cuttings_sem':
        match = re.match(r'(TS_\d+)_', p.stem, re.I)
        if match:
            row.update(original_group=match[1], tile_id=p.stem,
                       group_evidence='shared TS image prefix across published train/test cutouts and downscaled image; specimen identity unverified')
    elif dataset == 'fields_metal_cement':
        match = re.fullmatch(r'Rot(\d+)_(.+)', p.stem, re.I)
        if match:
            row.update(original_group=str(p.parent/match[2]), tile_id=p.stem,
                       group_evidence='same basename after rotation prefix; candidate variant group, not verified pixel correspondence')
    elif dataset == 'porous_zirconia_fibsem':
        match = re.fullmatch(r'Slice_(\d+)', p.stem, re.I)
        if match:
            row.update(original_group=str(p.parent), slice_id=match[1],
                       group_evidence='observed stack directory and slice number; physical specimen mapping unverified')
    elif dataset == 'rwth_steel_sem' and 'hr' in (x.lower() for x in p.parts):
        row.update(tile_id=p.stem, group_evidence='numeric image ID within HR acquisition branch; independent originals and cross-scale correspondence unknown')
    return row


def resolution(width, height):
    if width is None or height is None:
        return None, 'read_error', 'read_error'
    failed = [key+'<128' for key, value in (('width', width), ('height', height)) if value < 128]
    if failed:
        return False, ';'.join(failed), 'very_small'
    smallest = min(width, height)
    return True, '', 'small' if smallest < 224 else 'medium' if smallest < 448 else 'large'


def fingerprint(info):
    return (info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_dev, info.st_ino)


def expected_fingerprint(row):
    return tuple(row[k] for k in ('size_bytes', 'mtime_ns', 'ctime_ns', 'device', 'inode'))


@contextmanager
def raw_file(row):
    """Reject symlinks/nonregular files and changes since the discovery receipt."""
    descriptor = os.open(row['path'], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, 'rb') as stream:
        current = os.fstat(stream.fileno())
        if not stat.S_ISREG(current.st_mode) or fingerprint(current) != expected_fingerprint(row):
            raise ValueError('file_changed_since_discovery_or_not_regular')
        yield stream
        if fingerprint(os.fstat(stream.fileno())) != expected_fingerprint(row):
            raise ValueError('file_changed_during_read')


def inspect_header(row):
    start = time.perf_counter()
    combined = row['hash_status'] == 'pending'
    result = dict(header_status='read_error', format='', width=None, height=None,
                  mode='', channels=None, resolution_eligible=None,
                  resolution_reason='read_error', resolution_class='read_error',
                  ssl_candidate=False, read_error='')
    try:
        with raw_file(row) as stream:
            if combined:
                # Streaming hash warms the same file's pages. Pillow then reads
                # original headers on this descriptor, without a second open.
                hash_start = time.perf_counter()
                result.update(sha256=hash_stream(stream), hash_status='hashed',
                              hash_error='', hash_seconds=time.perf_counter()-hash_start)
                stream.seek(0)
            elif hasattr(os, 'posix_fadvise'):
                # Avoid speculative full-image read-ahead for header-only reads.
                try:
                    os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_RANDOM)
                except OSError:
                    pass  # Advisory only; identical metadata without support.
            header_start = time.perf_counter()
            try:
                with Image.open(stream) as image:
                    width, height = image.size
                    result.update(format=image.format, width=width, height=height,
                                  mode=image.mode, channels=len(image.getbands()))
                eligible, reason, category = resolution(width, height)
                result.update(header_status='ok', resolution_eligible=eligible,
                              resolution_reason=reason, resolution_class=category,
                              ssl_candidate=eligible and bool(row['is_microscopy']))
            except Exception as error:
                # A byte hash is still valid if the unchanged file's header is
                # invalid. The outer receipt checks the file after this branch.
                result['read_error'] = type(error).__name__+': '+str(error)
            result['header_seconds'] = time.perf_counter()-header_start
    except Exception as error:
        # Header-only inspection does not prove that every compressed pixel decodes.
        result.update(header_status='read_error', width=None, height=None,
                      resolution_eligible=None, ssl_candidate=False,
                      read_error=type(error).__name__+': '+str(error))
        if combined:
            result.update(sha256=None, hash_status='hash_error',
                          hash_error=type(error).__name__+': '+str(error),
                          hash_seconds=time.perf_counter()-start)
    result.setdefault('header_seconds', time.perf_counter()-start)
    return result


def hash_stream(stream, block_size=1024*1024):
    digest = hashlib.sha256()
    while chunk := stream.read(block_size):
        digest.update(chunk)
    return digest.hexdigest()


def hash_file(row, block_size=1024*1024):
    start = time.perf_counter()
    try:
        with raw_file(row) as stream:
            digest = hash_stream(stream, block_size)
        return dict(sha256=digest, hash_status='hashed', hash_error='',
                    hash_seconds=time.perf_counter()-start)
    except Exception as error:
        return dict(sha256=None, hash_status='hash_error',
                    hash_error=type(error).__name__+': '+str(error),
                    hash_seconds=time.perf_counter()-start)


def open_state(output, root, resume=False, datasets=None):
    output, root = Path(output).absolute(), Path(root).absolute()
    datasets = tuple(DATASETS if datasets is None else datasets)
    if not datasets or len(set(datasets)) != len(datasets) or not set(datasets) <= set(DATASETS + ADDITIONAL_DATASETS):
        raise ValueError('Select distinct datasets from the explicit allowlist; protected roots are excluded')
    if output == root or output.is_relative_to(root) and not output.is_relative_to(root/'_audit'):
        raise ValueError('Output must be separate from raw datasets (only root/_audit is allowed)')
    if output.is_symlink():
        raise ValueError('Output must not be a symlink')
    output.mkdir(parents=True, exist_ok=True)
    database = output/'audit_state.sqlite3'
    if not resume and any(output.iterdir()):
        raise FileExistsError('Output already contains files; use --resume to reuse this audit')
    if resume and not database.is_file():
        raise FileNotFoundError('--resume requires an existing audit_state.sqlite3')
    db = sqlite3.connect(database)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA journal_mode=DELETE')
    db.executescript('''
        CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS directories (
            dataset TEXT, path TEXT PRIMARY KEY, status TEXT NOT NULL, error TEXT);
        CREATE TABLE IF NOT EXISTS files (
            dataset TEXT NOT NULL, path TEXT PRIMARY KEY, relative_path TEXT, suffix TEXT,
            size_bytes INTEGER, mtime_ns INTEGER, ctime_ns INTEGER, device INTEGER, inode INTEGER,
            is_image INTEGER, role TEXT, role_evidence TEXT, is_microscopy INTEGER,
            modality TEXT, sample_group TEXT, original_group TEXT, slice_id TEXT, tile_id TEXT,
            group_evidence TEXT, split_hint TEXT, header_status TEXT, format TEXT,
            width INTEGER, height INTEGER, mode TEXT, channels INTEGER,
            resolution_eligible INTEGER, resolution_reason TEXT, resolution_class TEXT,
            ssl_candidate INTEGER, read_error TEXT, header_seconds REAL,
            sha256 TEXT, hash_status TEXT, hash_error TEXT, hash_seconds REAL);
        CREATE TABLE IF NOT EXISTS issues (
            path TEXT PRIMARY KEY, dataset TEXT, kind TEXT, detail TEXT);
        CREATE INDEX IF NOT EXISTS file_size ON files(is_image, size_bytes);
        CREATE INDEX IF NOT EXISTS file_header ON files(is_image, header_status);
        CREATE INDEX IF NOT EXISTS file_hash ON files(is_image, hash_status);
        CREATE INDEX IF NOT EXISTS file_digest ON files(sha256);
        CREATE INDEX IF NOT EXISTS file_dataset ON files(dataset);
        CREATE INDEX IF NOT EXISTS directory_pending_queue ON directories(status);
    ''')
    contract = dict(schema=SCHEMA, root=str(root), datasets=list(datasets), excluded=list(EXCLUDED),
                    minimum_side=128, image_suffixes=sorted(IMAGE_SUFFIXES),
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    stored = db.execute("SELECT value FROM meta WHERE key='contract'").fetchone()
    if stored and json.loads(stored[0]) != contract:
        db.close()
        raise ValueError('Audit code/root/rules differ from saved contract; use a new output directory')
    if not stored:
        db.execute('INSERT INTO meta VALUES (?,?)', ('contract', json.dumps(contract)))
        db.execute('INSERT INTO meta VALUES (?,?)', ('started', now()))
        for dataset in datasets:
            db.execute('INSERT INTO directories VALUES (?,?,?,?)', (dataset, str(root/dataset), 'pending', None))
        db.commit()
    return db


class Progress:
    def __init__(self, output, db, seconds=15):
        self.output, self.db, self.seconds = Path(output), db, seconds
        self.last = 0.
        self.started = time.monotonic()

    def update(self, stage, force=False, **extra):
        if not force and time.monotonic()-self.last < self.seconds:
            return
        self.last = time.monotonic()
        images = self.db.execute('SELECT count(*) FROM files WHERE is_image=1').fetchone()[0]
        headers = self.db.execute("SELECT count(*) FROM files WHERE is_image=1 AND header_status!='pending'").fetchone()[0]
        hashed = self.db.execute("SELECT count(*) FROM files WHERE hash_status='hashed'").fetchone()[0]
        event = dict(time=now(), stage=stage, image_files=images, headers_done=headers,
                     files_hashed=hashed, invocation_seconds=round(time.monotonic()-self.started, 3), **extra)
        atomic_json(self.output/'progress.json', event)
        with (self.output/'audit.log').open('a') as stream:
            stream.write(json.dumps(event, ensure_ascii=False)+'\n')
        print(json.dumps(event, ensure_ascii=False), flush=True)


def discover(db, root, progress):
    """Commit an entire directory at once; interruption repeats only that directory."""
    root = Path(root).absolute()
    committed = time.monotonic()
    directories_since_commit = 0
    while item := db.execute("SELECT * FROM directories WHERE status='pending' ORDER BY rowid LIMIT 1").fetchone():
        dataset, directory = item['dataset'], Path(item['path'])
        children, files, issues = [], [], []
        error = None
        try:
            if directory.is_symlink():
                raise ValueError('directory_symlink_not_followed')
            with os.scandir(directory) as entries:
                for entry in entries:
                    try:
                        if entry.is_symlink():
                            issues.append((entry.path, dataset, 'symlink_not_followed', 'target never opened'))
                        elif entry.is_dir(follow_symlinks=False):
                            children.append((dataset, entry.path, 'pending', None))
                        elif entry.is_file(follow_symlinks=False):
                            info = entry.stat(follow_symlinks=False)
                            relative = str(Path(entry.path).relative_to(root/dataset))
                            suffix = Path(entry.name).suffix.lower()
                            sidecar = entry.name.startswith('._') or '__MACOSX' in Path(relative).parts
                            image = suffix in IMAGE_SUFFIXES and not sidecar
                            role, microscopy, evidence = classify(dataset, relative) if image else ('non_image', False, 'OS sidecar' if sidecar else 'not a supported image filename; no content decoding')
                            row = dict(dataset=dataset, path=entry.path, relative_path=relative,
                                suffix=suffix, size_bytes=info.st_size, mtime_ns=info.st_mtime_ns,
                                ctime_ns=info.st_ctime_ns, device=info.st_dev, inode=info.st_ino,
                                is_image=image, role=role, role_evidence=evidence,
                                is_microscopy=microscopy, header_status='pending' if image else 'not_image',
                                hash_status='unclassified' if image else 'not_image',
                                **identifiers(dataset, relative))
                            files.append(tuple(row.get(k) for k in FILE_COLUMNS))
                        else:
                            issues.append((entry.path, dataset, 'nonregular_not_opened', 'not a regular file or directory'))
                    except OSError as exc:
                        issues.append((entry.path, dataset, 'entry_error', str(exc)))
        except (OSError, ValueError) as exc:
            error = str(exc)
        db.executemany('INSERT OR IGNORE INTO directories VALUES (?,?,?,?)', children)
        db.executemany('INSERT OR IGNORE INTO files ('+','.join(FILE_COLUMNS)+') VALUES ('+','.join('?' for _ in FILE_COLUMNS)+')', files)
        db.executemany('INSERT OR REPLACE INTO issues VALUES (?,?,?,?)', issues)
        db.execute('UPDATE directories SET status=?,error=? WHERE path=?',
                   ('error' if error else 'done', error, str(directory)))
        directories_since_commit += 1
        if directories_since_commit >= 64 or time.monotonic()-committed >= 2:
            db.commit()
            directories_since_commit, committed = 0, time.monotonic()
        progress.update('discovery', current_dataset=dataset, current_directory=str(directory))
    db.commit()
    progress.update('discovery_finished', force=True)


def update_row(db, path, values):
    db.execute('UPDATE files SET '+','.join(k+'=?' for k in values)+' WHERE path=?', (*values.values(), path))


def metadata_join(db, root):
    """Exact local CSV joins only; never traverse new image paths from metadata."""
    for item in db.execute("SELECT * FROM files WHERE dataset='uhcsdb' AND relative_path LIKE '%/metadata.csv'").fetchall():
        path = Path(item['path'])
        micrographs = path.parent/'micrographs'
        with raw_file(item) as stream:
            reader = csv.DictReader(line.decode('utf-8-sig') for line in stream)
            if not {'path', 'sample_key', 'detector'} <= set(reader.fieldnames or []):
                continue
            for row in reader:
                relative = Path(row['path'])
                if relative.is_absolute() or '..' in relative.parts:
                    continue
                target = micrographs/relative
                # An update only affects paths already discovered under allowed data.
                sample = row['sample_key'].strip()
                modality = row['detector'].strip()
                if sample:
                    db.execute('UPDATE files SET sample_group=?, group_evidence=?, modality=? WHERE dataset=? AND path=? AND is_microscopy=1',
                        ('uhcsdb/sample_key='+sample, 'exact filename join to local metadata.csv sample_key; SE/BSE registration not inferred',
                         'SEM'+('/'+modality if modality else '/unknown detector'), 'uhcsdb', str(target)))
    for item in db.execute("SELECT * FROM files WHERE dataset='matscifig_sem' AND relative_path='manifest.csv'").fetchall():
        base = Path(root).absolute()/'matscifig_sem'
        with raw_file(item) as stream:
            reader = csv.DictReader(line.decode('utf-8-sig') for line in stream)
            if not {'path', 'source_file', 'doi', 'subtype_valid'} <= set(reader.fieldnames or []):
                continue
            for row in reader:
                target = Path(row['path'])
                if '..' in target.parts:
                    continue
                if not target.is_absolute():
                    target = base/target
                if not target.is_relative_to(base/'images'):
                    continue
                valid = row['subtype_valid'].strip().lower() == 'true'
                doi = row['doi'].strip().lower()
                source = Path(row['source_file'])
                figure = re.fullmatch(r'(.+_img\d+)_[A-Z]+\d*', source.stem)
                original = (('doi='+doi+'/') if doi else '')+figure[1] if figure else 'unknown'
                db.execute('''UPDATE files SET role=?,role_evidence=?,is_microscopy=?,modality=?,
                           sample_group=?,original_group=?,tile_id=?,group_evidence=?
                           WHERE dataset='matscifig_sem' AND path=? AND is_image=1 AND role='unknown' ''',
                           ('microscopy_image' if valid else 'unknown',
                            'exact local SEM-download manifest join; subtype_valid='+str(valid)+'; publication panel content not pixel-validated',
                            valid, 'SEM (local filtered-download metadata; not pixel-validated)' if valid else 'unknown',
                            'doi='+doi if doi else 'unknown', original, source.stem,
                            'DOI publication group from manifest; source figure/panel basename is a grouping candidate, not specimen proof', str(target)))
    db.commit()


def process_pending(db, progress, stage, worker, workers=1, batch_size=32):
    condition = "is_image=1 AND header_status='pending'" if stage == 'headers' else "is_image=1 AND hash_status='pending'"
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        while rows := db.execute('SELECT * FROM files WHERE '+condition+' ORDER BY rowid LIMIT ?', (batch_size,)).fetchall():
            with db:
                for row, result in zip(rows, pool.map(worker, map(dict, rows))):
                    update_row(db, row['path'], result)
                    progress.update(stage, current_dataset=row['dataset'])
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    progress.update(stage+'_finished', force=True)


def prepare_hashes(db):
    # Global size counts include every role and all selected datasets, including errors.
    with db:
        # A formerly unreadable directory may have been recovered on --resume.
        # Reclassify size singletons without reopening successful hashed files.
        db.execute("UPDATE files SET hash_status='unclassified' WHERE hash_status='skipped_unique_size'")
        db.execute('''UPDATE files SET hash_status='skipped_unique_size'
                      WHERE is_image=1 AND hash_status='unclassified' AND size_bytes IN
                      (SELECT size_bytes FROM files WHERE is_image=1 GROUP BY size_bytes HAVING count(*)=1)''')
        db.execute("UPDATE files SET hash_status='pending' WHERE is_image=1 AND hash_status='unclassified'")


def hash_policy(db, skip_requested=False):
    """An explicit cancellation persists across later --resume invocations."""
    if skip_requested:
        with db:
            db.execute("INSERT OR REPLACE INTO meta VALUES ('hash_policy','disabled_by_user')")
    row = db.execute("SELECT value FROM meta WHERE key='hash_policy'").fetchone()
    enabled = row is None or row[0] != 'disabled_by_user'
    if not enabled:
        with db:
            db.execute("UPDATE files SET hash_status='skipped_by_request' WHERE is_image=1 AND hash_status IN ('pending','unclassified')")
    return enabled


def csv_export(path, fields, rows):
    temporary = Path(path).with_name(Path(path).name+'.partial')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (str(bool(row[key])).lower() if key in BOOL_COLUMNS and row.get(key) is not None else row.get(key)) for key in fields})
    temporary.replace(path)


def percentiles(values):
    values = sorted(values)
    if not values:
        return {k: None for k in ('min', 'p25', 'median', 'p75', 'p95', 'max', 'mean')}
    def q(fraction):
        position = fraction*(len(values)-1)
        left = math.floor(position)
        right = min(left+1, len(values)-1)
        return values[left]+(values[right]-values[left])*(position-left)
    return dict(min=values[0], p25=q(.25), median=q(.5), p75=q(.75), p95=q(.95),
                max=values[-1], mean=sum(values)/len(values))


def export(db, output, progress):
    output = Path(output)
    selected_datasets = json.loads(db.execute("SELECT value FROM meta WHERE key='contract'").fetchone()[0])['datasets']
    progress.update('export', force=True)
    policy = db.execute("SELECT value FROM meta WHERE key='hash_policy'").fetchone()
    hashes_enabled = policy is None or policy[0] != 'disabled_by_user'
    exact_complete = hashes_enabled and not (
        db.execute("SELECT count(*) FROM directories WHERE status!='done'").fetchone()[0]
        or db.execute('SELECT count(*) FROM issues').fetchone()[0]
        or db.execute("SELECT count(*) FROM files WHERE is_image=1 AND hash_status IN ('pending','unclassified','hash_error','skipped_by_request')").fetchone()[0])
    # One indexed digest lookup builds all duplicate group statistics in memory.
    groups = defaultdict(list)
    for row in db.execute("SELECT dataset,path,role,is_microscopy,ssl_candidate,split_hint,sha256 FROM files WHERE hash_status='hashed' ORDER BY sha256,dataset,path"):
        groups[row['sha256']].append(dict(row))
    duplicates = {key: rows for key, rows in groups.items() if len(rows) > 1}
    sizes = {key: len(rows) for key, rows in duplicates.items()}
    manifests = (dict(row) | {'duplicate_group_size': sizes.get(row['sha256'], 1) if row['hash_status'] in ('hashed', 'skipped_unique_size') else None}
                 for row in db.execute('SELECT * FROM files WHERE is_image=1 ORDER BY dataset,relative_path'))
    csv_export(output/'images_manifest.csv', (*FILE_COLUMNS, 'duplicate_group_size'), manifests)
    csv_export(output/'all_files.csv', FILE_COLUMNS, (dict(r) for r in db.execute('SELECT * FROM files ORDER BY dataset,relative_path')))
    duplicate_rows = []
    for digest, rows in sorted(duplicates.items()):
        datasets = Counter(r['dataset'] for r in rows)
        roles = Counter(r['role'] for r in rows)
        microscopy = sum(r['is_microscopy'] for r in rows)
        kit_splits = {r['split_hint'] for r in rows if r['dataset'] == 'kit_sem_powder'}
        for row in rows:
            duplicate_rows.append(dict(sha256=digest, group_size=len(rows),
                group_complete=exact_complete,
                dataset_group_size=datasets[row['dataset']], role_group_size=roles[row['role']],
                microscopy_group_size=microscopy, cross_dataset=len(datasets)>1,
                cross_role=len(roles)>1, kit_train_val_overlap={'train', 'val'} <= kit_splits,
                **{k:v for k,v in row.items() if k != 'sha256'}))
    duplicate_fields = ('sha256', 'group_size', 'group_complete', 'dataset_group_size', 'role_group_size',
        'microscopy_group_size', 'cross_dataset', 'cross_role', 'kit_train_val_overlap',
        'dataset', 'path', 'role', 'is_microscopy', 'ssl_candidate', 'split_hint')
    csv_export(output/'exact_duplicates.csv', duplicate_fields, duplicate_rows)
    csv_export(output/'scan_issues.csv', ('dataset', 'path', 'kind', 'detail'),
        [dict(r) for r in db.execute('SELECT * FROM issues ORDER BY dataset,path')]+
        [dict(dataset=r['dataset'], path=r['path'], kind='directory_error', detail=r['error'])
         for r in db.execute("SELECT * FROM directories WHERE status='error'")])
    summaries, distributions, ambiguities = [], [], []
    for dataset in selected_datasets:
        records = [dict(r) for r in db.execute('SELECT * FROM files WHERE dataset=? AND is_image=1', (dataset,))]
        roles = Counter(r['role'] for r in records)
        size_classes = Counter(r['resolution_class'] for r in records)
        eligible = sum(bool(r['ssl_candidate']) for r in records)
        same = [sum(r['dataset']==dataset for r in rows) for rows in duplicates.values()]
        micro_same = [sum(r['dataset']==dataset and bool(r['is_microscopy']) for r in rows) for rows in duplicates.values()]
        candidate_same = [sum(r['dataset']==dataset and bool(r['ssl_candidate']) for r in rows) for rows in duplicates.values()]
        summary = dict(dataset=dataset, image_files=len(records),
            microscopy_images=sum(bool(r['is_microscopy']) for r in records),
            microscopy_images_ge128=eligible,
            microscopy_too_small=sum(bool(r['is_microscopy']) and r['resolution_class']=='very_small' for r in records),
            too_small_all_roles=size_classes['very_small'], masks_labels=roles['segmentation_mask']+roles['label_or_annotation'],
            masks=roles['segmentation_mask'], labels=roles['label_or_annotation'],
            derived_images=roles['derived_image'], auxiliary_images=roles['auxiliary_image'], unknown_roles=roles['unknown'],
            read_errors=sum(r['header_status']=='read_error' for r in records),
            headers_pending=sum(r['header_status']=='pending' for r in records),
            all_roles_ge128=sum(r['resolution_eligible']==1 for r in records),
            very_small=size_classes['very_small'], small=size_classes['small'], medium=size_classes['medium'], large=size_classes['large'],
            files_hashed=sum(r['hash_status']=='hashed' for r in records),
            skipped_unique_size=sum(r['hash_status']=='skipped_unique_size' for r in records),
            hash_errors=sum(r['hash_status']=='hash_error' for r in records),
            hashes_skipped_by_request=sum(r['hash_status']=='skipped_by_request' for r in records),
            duplicate_groups_within_dataset=sum(n>1 for n in same),
            duplicate_files_within_dataset=sum(n for n in same if n>1),
            duplicate_extra_files_within_dataset=sum(max(0,n-1) for n in same),
            microscopy_duplicate_extra_within_dataset=sum(max(0,n-1) for n in micro_same),
            candidates_unique_within_dataset=eligible-sum(max(0,n-1) for n in candidate_same) if exact_complete else None,
            duplicate_counts_complete=exact_complete,
            files_in_cross_dataset_duplicate_groups=sum(r['dataset']==dataset for rows in duplicates.values() if len({r['dataset'] for r in rows})>1 for r in rows),
            all_regular_files=db.execute('SELECT count(*) FROM files WHERE dataset=?', (dataset,)).fetchone()[0],
            directories=db.execute("SELECT count(*) FROM directories WHERE dataset=? AND status='done'", (dataset,)).fetchone()[0])
        summaries.append(summary)
        for role in sorted(roles)+['ALL_ROLES']:
            selected = records if role == 'ALL_ROLES' else [r for r in records if r['role']==role]
            readable = [r for r in selected if r['header_status']=='ok']
            item = dict(dataset=dataset, role=role, count=len(selected), readable=len(readable),
                        read_errors=sum(r['header_status']=='read_error' for r in selected),
                        headers_pending=sum(r['header_status']=='pending' for r in selected),
                        **{key:sum(r['resolution_class']==key for r in selected) for key in ('very_small', 'small', 'medium', 'large')})
            for field in ('width', 'height', 'pixels'):
                values = [r['width']*r['height'] if field=='pixels' else r[field] for r in readable]
                item.update({field+'_'+key:value for key,value in percentiles(values).items()})
            distributions.append(item)
        for role in ('unknown', 'auxiliary_image', 'label_or_annotation'):
            paths = [r['relative_path'] for r in records if r['role']==role]
            if paths:
                ambiguities.append(dict(dataset=dataset, role=role, count=len(paths), examples=sorted(paths)[:4]))
    csv_export(output/'dataset_summary.csv', list(summaries[0]), summaries)
    csv_export(output/'resolution_summary.csv', list(distributions[0]) if distributions else ('dataset','role','count'), distributions)
    scan_errors = db.execute("SELECT count(*) FROM directories WHERE status!='done'").fetchone()[0]
    issues = db.execute('SELECT count(*) FROM issues').fetchone()[0]
    header_pending = db.execute("SELECT count(*) FROM files WHERE header_status='pending'").fetchone()[0]
    hash_pending = db.execute("SELECT count(*) FROM files WHERE is_image=1 AND hash_status IN ('pending','unclassified')").fetchone()[0]
    hash_errors = sum(r['hash_errors'] for r in summaries)
    read_errors = sum(r['read_errors'] for r in summaries)
    kit = {digest:rows for digest,rows in duplicates.items()
           if {'train','val'} <= {r['split_hint'] for r in rows if r['dataset']=='kit_sem_powder'}}
    cross_dataset = {digest:rows for digest,rows in duplicates.items() if len({r['dataset'] for r in rows})>1}
    role_groups = Counter(r['role'] for rows in duplicates.values() for r in rows)
    candidate_extras = sum(max(0, sum(bool(r['ssl_candidate']) for r in rows)-1) for rows in duplicates.values())
    inventory_complete=not(scan_errors or issues or header_pending or read_errors)
    totals = dict(complete=inventory_complete and (exact_complete or not hashes_enabled),
        inventory_complete=inventory_complete, hash_check_enabled=hashes_enabled,
        hash_check_cancelled_by_user=not hashes_enabled,
        discovery_complete=not(scan_errors or issues), headers_complete=not(header_pending or read_errors),
        exact_hash_check_complete=exact_complete,
        time=now(), schema=SCHEMA, datasets=selected_datasets, excluded=list(EXCLUDED),
        image_files=sum(r['image_files'] for r in summaries),
        all_regular_files=sum(r['all_regular_files'] for r in summaries),
        microscopy_images=sum(r['microscopy_images'] for r in summaries),
        microscopy_images_ge128=sum(r['microscopy_images_ge128'] for r in summaries),
        microscopy_candidates_unique_global=sum(r['microscopy_images_ge128'] for r in summaries)-candidate_extras if exact_complete else None,
        all_roles_ge128=sum(r['all_roles_ge128'] for r in summaries),
        too_small_all_roles=sum(r['too_small_all_roles'] for r in summaries),
        masks_labels=sum(r['masks_labels'] for r in summaries), unknown_roles=sum(r['unknown_roles'] for r in summaries),
        auxiliary_images=sum(r['auxiliary_images'] for r in summaries),
        read_errors=read_errors, hash_errors=hash_errors, header_pending=header_pending,
        hash_pending=hash_pending, directory_errors=scan_errors, scan_issues=issues,
        files_hashed=sum(r['files_hashed'] for r in summaries),
        skipped_unique_size=sum(r['skipped_unique_size'] for r in summaries),
        hashes_skipped_by_request=sum(r['hashes_skipped_by_request'] for r in summaries),
        bytes_hashed=db.execute("SELECT coalesce(sum(size_bytes),0) FROM files WHERE hash_status='hashed'").fetchone()[0],
        duplicate_groups=len(duplicates), duplicate_group_files=sum(len(r) for r in duplicates.values()),
        duplicate_extra_files=sum(len(r)-1 for r in duplicates.values()),
        duplicate_files_by_role=dict(role_groups),
        cross_dataset_groups=len(cross_dataset), cross_dataset_files=sum(len(r) for r in cross_dataset.values()),
        cross_role_groups=sum(len({r['role'] for r in rows})>1 for rows in duplicates.values()),
        kit_train_val_duplicate_groups=len(kit),
        kit_train_val_duplicate_files=sum(sum(r['dataset']=='kit_sem_powder' for r in rows) for rows in kit.values()),
        header_seconds_total=db.execute('SELECT coalesce(sum(header_seconds),0) FROM files').fetchone()[0],
        hash_seconds_total=db.execute('SELECT coalesce(sum(hash_seconds),0) FROM files').fetchone()[0],
        contract=json.loads(db.execute("SELECT value FROM meta WHERE key='contract'").fetchone()[0]),
        started=db.execute("SELECT value FROM meta WHERE key='started'").fetchone()[0],
        dataset_summary=summaries)
    atomic_json(output/'audit_summary.json', totals)
    write_report(output, totals, ambiguities)
    progress.update('complete' if totals['complete'] else 'incomplete', force=True, complete=totals['complete'])
    return totals


def write_report(output, t, ambiguities):
    lines = ['# Vollständiges Bildinventar und exakte SHA-256-Duplikate', '',
        f'Beginn: {t["started"]}; Berichtsstand: {t["time"]}.', '',
        '**Status Bildinventar: '+('VOLLSTÄNDIG innerhalb des dokumentierten Dateiformat-Umfangs.' if t['inventory_complete'] else 'UNVOLLSTÄNDIG; Fehler/offene Prüfungen unten beachten.')+'**', '',
        '**SHA-256-Duplikattest: '+('AUF NUTZERWUNSCH ABGEBROCHEN; keine vollständige Duplikataussage.' if t['hash_check_cancelled_by_user'] else 'VOLLSTÄNDIG.' if t['exact_hash_check_complete'] else 'UNVOLLSTÄNDIG.')+'**', '',
        '## Wichtigste Zahlen', '',
        '| Datensatz | Alle Bilder | Mikroskopiebilder ≥128×128 | Zu klein | Masken/Labels | Exakte Duplikate |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in t['dataset_summary']:
        cells=[str(r[k]) for k in ('image_files', 'microscopy_images_ge128', 'too_small_all_roles', 'masks_labels')]
        cells.append(str(r['duplicate_extra_files_within_dataset']) if t['exact_hash_check_complete'] else 'nicht vollständig geprüft')
        lines.append('| '+r['dataset']+' | '+' | '.join(cells)+' |')
    lines += ['', '„Zu klein“ zählt alle Bildrollen mit mindestens einer Seite unter 128 px; '
        'unlesbare Header zählen separat. „Exakte Duplikate“ zählt überzählige Dateien innerhalb '
        'des jeweiligen Datensatzes (Gruppengröße minus eins), einschließlich Masken/Labels. '
        'Alle Gruppenteilnehmer, Rollen und datensatzübergreifenden Gruppen stehen in exact_duplicates.csv. '
        'Die Summen dieser Spalte sind deshalb nicht die globale Zahl überzähliger Dateien. '
        'Bei abgebrochenem Hashlauf sind vorhandene Gruppen und deren Größen nur bestätigte Untergrenzen; '
        'group_complete=false kennzeichnet sie. Null beobachtete Gruppen bedeutet dann keinen Duplikatausschluss.', '',
        f'- Vollständig aufgelistete reguläre Dateien: **{t["all_regular_files"]}**; Bilddateien: **{t["image_files"]}**.',
        f'- Wahrscheinliche Mikroskopiebilder einschließlich geeigneter Crops/Tiles: **{t["microscopy_images"]}**; davon ≥128×128: **{t["microscopy_images_ge128"]}**.',
        f'- Alle Bildrollen ≥128×128: **{t["all_roles_ge128"]}**; Masken/Labels: **{t["masks_labels"]}**; unbekannte Rollen: **{t["unknown_roles"]}**; Zusatzbilder: **{t["auxiliary_images"]}**.',
        '- Nach rein bytebasierter globaler Zusammenfassung geeigneter Mikroskopie-Kandidaten: **'+
        (str(t['microscopy_candidates_unique_global']) if t['exact_hash_check_complete'] else 'nicht bestimmt; Hashprüfung unvollständig')+
        '**. Keine Dateien wurden tatsächlich entfernt oder als Trainingssatz freigegeben.', '',
        '## Auflösungen', '',
        '| Datensatz | <128 auf einer Seite | ≥128, <224 auf einer Seite | ≥224, <448 auf einer Seite | Beide ≥448 | Headerfehler |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in t['dataset_summary']:
        lines.append('| '+r['dataset']+' | '+' | '.join(str(r[k]) for k in ('very_small', 'small', 'medium', 'large', 'read_errors'))+' |')
    lines += ['', 'Die Tabelle umfasst alle Bildrollen. resolution_summary.csv trennt jede Rolle und enthält '
        'Minimum, 25./50./75./95. Perzentil, Maximum und Mittelwert für Breite, Höhe und Pixelzahl. '
        'Originalheader, keine Skalierung. resolution_eligible ist bei Headerfehlern leer (unbekannt), '
        'ssl_candidate ist dort immer false.', '', '## SHA-256, Rollen und KIT train/val', '',
        f'- Tatsächlich vollständig gehashte Bilddateien: **{t["files_hashed"]}** ({t["bytes_hashed"]/2**30:.3f} GiB).',
        f'- Ohne Hash als exakt einzigartig ausgeschlossene Bilddateien wegen global einmaliger Dateigröße: **{t["skipped_unique_size"]}**.',
        f'- Weitere Hashberechnungen auf Nutzerwunsch ausgelassen: **{t["hashes_skipped_by_request"]}**. Bereits berechnete Hashes bleiben erhalten.',
        f'- Byteidentische Gruppen: **{t["duplicate_groups"]}**; beteiligte Dateien: **{t["duplicate_group_files"]}**; überzählige Dateien global: **{t["duplicate_extra_files"]}**.',
        f'- Datensatzübergreifende Gruppen: **{t["cross_dataset_groups"]}**, beteiligte Dateien: **{t["cross_dataset_files"]}**.',
        f'- Rollenübergreifende Gruppen: **{t["cross_role_groups"]}**. Teilnehmer je Rolle: `{json.dumps(t["duplicate_files_by_role"], sort_keys=True)}`.',
        f'- **KIT train/val: {t["kit_train_val_duplicate_groups"]} bestätigte SHA-256-Gruppen**, {t["kit_train_val_duplicate_files"]} beteiligte KIT-Dateien. '
        'Nachweis basiert ausschließlich auf Inhaltshashes, nicht auf Namensgleichheit.', '',
        'Bei aktivierter Hashprüfung werden Dateigrößen global über die ausgewählten Datensätze und Bildrollen gruppiert. Nur Gruppen '
        'mit mindestens zwei Dateien werden dann mit SHA-256 gehasht. Header und nötiger Hash werden '
        'nach Möglichkeit über dieselbe geöffnete Originaldatei gelesen; für bereits gespeicherte '
        'Header werden noch fehlende Hashes separat ergänzt. Erfolgreich committete Hashes werden gespeichert; '
        'Fortsetzen liest bereits erfolgreich geprüfte Inhalte nicht erneut. Keine pHashes, '
        'keine Pixelgleichheits- oder Ähnlichkeitsprüfung. Andere Kodierungen, Nachbarslices, '
        'Crops und gleiche Proben können unerkannt verwandt sein.', '', '## Vollständigkeit, Methode und Grenzen', '',
        f'- Verzeichnisfehler/offene Verzeichnisse: {t["directory_errors"]}; sonstige Scan-Hinweise: {t["scan_issues"]}; '
        f'Headerfehler: {t["read_errors"]}; offene Header: {t["header_pending"]}; Hashfehler: {t["hash_errors"]}; offene Hashes: {t["hash_pending"]}.',
        '- os.scandir besucht ausschließlich die explizit ausgewählten Dataset-Wurzeln ('+', '.join(t['datasets'])+'), rekursiv und ohne Symlinks. '
        'nffa_sem_100 und aachen_heerlen werden weder aufgelistet noch geöffnet oder gehasht.',
        '- Bilddateien werden anhand der in audit_summary.json dokumentierten Suffixe erkannt. '
        'OS-Ressourcenforks (._*, __MACOSX) sind keine Bilddateien. Alle anderen regulären Dateien '
        'bleiben mit Pfad/Suffix/Größe in all_files.csv nachweisbar. Archive und Array-/Dokumentcontainer '
        'werden nicht entpackt oder als einzelne Rasterbilder gezählt; Bilder mit unbekannten/falschen '
        'Suffixen sind durch diesen Audit nicht ausgeschlossen.',
        '- Pillow liest nur Header der ersten Seite/des ersten Frames; kein load(), convert(), '
        'verify() oder vollständiges Dekodieren. Daher keine Aussage über Lesbarkeit aller Bildpixel, '
        'weiterer TIFF-Seiten, Qualitätsmängel oder Bildinhalte. Der Pillow-Pixelgrößenschutz wird '
        'für diesen ausschließlich headerbasierten Prozess deaktiviert; keine Raster werden angelegt.',
        '- Maximal zwei I/O-Threads, keine GPU. Hashes in 1-MiB-Blöcken; Größen/mtime/ctime/inode werden '
        'vor und nach dem Lesen gegen den Discovery-Stand geprüft. Dies ist ein Snapshot unveränderter Rohdaten; '
        'später neu hinzugekommene Dateien erfordern einen neuen Audit-Ausgabeordner.',
        f'- Summierte Header-Lesezeit: {t["header_seconds_total"]:.1f} s; summierte Hash-Lesezeit: {t["hash_seconds_total"]:.1f} s '
        '(bei zwei Threads keine Wandzeit). Phasenfortschritt und Wandzeiten stehen in audit.log.',
        '- Rolle und Modalität stammen aus dokumentierten Pfadregeln, nicht aus Sichtprüfung. '
        'Wahrscheinliche Mikroskopie-Tiles/Crops dürfen Kandidaten sein; Overlays und unklare Rollen nicht. '
        'LITHOS PPL/XPL sind optische Mikroskopie, keine SEM-Aufnahmen. MicroAl enthält mehrere Modalitäten.',
        '- Gruppenfelder beschreiben beobachtete Pfad-/Namensgruppen, keine nachgewiesene statistische '
        'Unabhängigkeit. UHCS sample_key kommt aus einem exakten lokalen metadata.csv-Dateinamenjoin. '
        'LITHOS-Zahlenhierarchie, Cross-Microscope-Mikroskopzuordnung, JLU-Slice-Registrierung und '
        'BIL-Serienursprünge benötigen weitere Interpretation.', '', '## Unklarheiten für spätere Sichtprüfung', '',
        '- KIT BO/RL/Phenom bleiben unknown; insbesondere sind gleich große Graustufenbilder kein Maskennachweis.',
        '- MicroAl label/annotation-Pfade zählen als Ziele, ihre Pixel-/Klassenbedeutung bleibt offen.',
        '- Headerfehler und unübliche Container anhand images_manifest.csv, scan_issues.csv und all_files.csv prüfen.']
    for row in ambiguities:
        lines.append(f'- {row["dataset"]}, {row["role"]}: {row["count"]}; Beispiele: '+ '; '.join('`'+p+'`' for p in row['examples']))
    notes = {
        'drill_cuttings_sem': 'Veröffentlichte Training_Set/Test_Set bleiben im split_hint erhalten; gemeinsame TS-Präfixe können über Splits hinweg vorkommen. CONTAM/TOUCH, downscaled Originale und Cutouts vor einer Trainingsauswahl getrennt prüfen.',
        'fields_metal_cement': 'Rot*-Dateien sind Variantenkandidaten; Dateizahl ist keine Zahl unabhängiger Originale. Keine festen Rotationen ungeprüft mehrfach gewichten.',
        'porous_zirconia_fibsem': 'Slices desselben Stacks sind stark verwandt; Stack-/Proben-Gruppierung und spätere Auswahl erforderlich. Nur erste TIFF-Seite inventarisiert.',
        'rwth_steel_sem': 'Nur bekannte HR-Zweige als Mikroskopiekandidaten; Patch-Ursprung und mögliche Beziehungen zwischen 10mu/100mu sind nicht verifiziert.',
        'matscifig_sem': 'Nur exakte Manifesttreffer mit subtype_valid=True als wahrscheinliche SEM-Bilder. DOI ist eine Publikationsgruppe, keine unabhängige Probe. Panels, Text, Maßstäbe, Wiederverwendung und Eignung müssen noch geprüft werden.',
    }
    for dataset in t['datasets']:
        if dataset in notes:
            lines.append('- '+dataset+': '+notes[dataset])
    lines += ['', '## Dateien und Fortsetzen', '',
        '- images_manifest.csv: eine Zeile pro Bilddatei, Rolle, Originalheader, Auflösung, '
        'Hashstatus und gegebenenfalls SHA-256.',
        '- dataset_summary.csv / resolution_summary.csv: Aggregate und Rollenverteilungen.',
        '- exact_duplicates.csv: eine Zeile pro Teilnehmer einer bestätigten Gruppe.',
        '- all_files.csv / scan_issues.csv: vollständige reguläre Dateiliste und ausgelassene/problematische Einträge.',
        '- audit_state.sqlite3 / progress.json / audit.log / audit_summary.json: Fortsetzungsstand und Provenienz.', '',
        'Discovery wird spätestens nach 64 Verzeichnissen bzw. zwei Sekunden, Header/Hashes nach '
        'je 32 Dateien gespeichert. Bei Abbruch muss höchstens der noch uncommittete Block erneut '
        'gelesen werden; vollständige, bereits gespeicherte Hashes werden wiederverwendet.', '',
        'Fortsetzen mit identischem Skript und unveränderten Daten:', '```bash',
        'python scripts/audit_image_inventory.py \\',
        f'  --root {t["contract"]["root"]} --output {Path(output).absolute()} --datasets '+ ' '.join(t['datasets'])+' --workers 2 --resume'+(' --skip-hashes' if t['hash_check_cancelled_by_user'] else ''),
        '```', '', 'Nur Berichte erneut aus der Audit-Datenbank erzeugen, ohne Rohdaten zu öffnen: '
        '`--resume --render-only`. Fehler werden nicht stillschweigend als Erfolg behandelt. '
        'Rohdaten, Holdouts, Masken und Trainingsprozesse wurden nicht verändert.', '']
    temporary = Path(output)/'report.md.partial'
    temporary.write_text('\n'.join(lines))
    temporary.replace(Path(output)/'report.md')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, default=ROOT/'_audit/full_inventory')
    parser.add_argument('--datasets', nargs='+', choices=DATASETS+ADDITIONAL_DATASETS, default=list(DATASETS),
                        help='Explicit roots only; default preserves the original nine-dataset scope')
    parser.add_argument('--workers', type=int, choices=(1, 2), default=1)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--render-only', action='store_true')
    parser.add_argument('--skip-hashes', action='store_true', help='Persistently cancel further SHA-256 work; preserve existing receipts and continue headers')
    args = parser.parse_args(argv)
    if args.render_only and not args.resume:
        parser.error('--render-only requires --resume')
    # Header inspection allocates no raster; large originals must still be counted.
    Image.MAX_IMAGE_PIXELS = None
    db = open_state(args.output, args.root, args.resume, args.datasets)
    progress = Progress(args.output, db)
    lock = (args.output/'audit.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    def interrupted(*_):
        raise KeyboardInterrupt()
    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        if not args.render_only:
            # Persist the no-hash decision even if discovery is interrupted.
            if args.skip_hashes:
                hash_policy(db, True)
            if args.resume:
                with db:
                    db.execute("UPDATE directories SET status='pending' WHERE status='error'")
            discover(db, args.root, progress)
            metadata_join(db, args.root)
            hashes_enabled = hash_policy(db, args.skip_hashes)
            if hashes_enabled:
                prepare_hashes(db)
            else:
                progress.update('hash_cancelled_by_user', force=True)
            if hashes_enabled:
                progress.update('hash_start', force=True,
                    hash_candidates=db.execute("SELECT count(*) FROM files WHERE hash_status IN ('pending','hashed','hash_error')").fetchone()[0])
            process_pending(db, progress, 'headers', inspect_header, args.workers)
            if hashes_enabled:
                process_pending(db, progress, 'hashes', hash_file, args.workers)
        result = export(db, args.output, progress)
        print('Report: '+str(args.output/'report.md'), flush=True)
        return 0 if result['complete'] else 2
    except KeyboardInterrupt:
        db.rollback()
        progress.update('interrupted', force=True)
        print('Audit interrupted; repeat with --resume. Raw datasets unchanged.', flush=True)
        return 130
    finally:
        signal.signal(signal.SIGTERM, previous)
        db.close()
        lock.close()


if __name__ == '__main__':
    raise SystemExit(main())
