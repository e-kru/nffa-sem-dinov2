"""Apply the audited RWTH HR selection to an existing SEM V0 manifest."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import fcntl
from itertools import zip_longest
import json
from pathlib import Path
import random
import re
import shutil
from tempfile import TemporaryDirectory

from scripts.build_sem_posttraining_manifest import THRESHOLDS, atomic_json, metadata_sha, read_json, rows

from src.config import DATASETS_ROOT


SOURCE = 'rwth_steel_sem'
DATASET_ROOT = DATASETS_ROOT / SOURCE
HR_ROOT = DATASET_ROOT / 'hr_images/SR for SEM/data'
SOURCE_COUNTS = {'16MnCrS5': 144000, 'DP/10mu': 25281, 'DP/100mu': 22119}
TARGET_COUNTS = {'16MnCrS5': 16000, 'DP/10mu': 10000, 'DP/100mu': 0}
EXCLUSION = {'16MnCrS5': 'excluded_rwth_outside_frame_grid',
             'DP/10mu': 'excluded_rwth_outside_seed42_sample',
             'DP/100mu': 'excluded_rwth_dp100mu_v0'}
GRID = tuple(range(1, 30, 3))
BLOCK_SIZE = 900
FRAME_COUNT = 160
REVISION = 'rwth_hr26k_v0'
PROBE_POINTS = ((0, 0), (0, 28), (14, 14), (28, 0), (28, 28))
HR_NAME = re.compile(r'(16MnCrS5|DP/10mu|DP/100mu)/hr/(0|[1-9][0-9]*)\.png')


def hr_key(path):
    relative = Path(path).relative_to(HR_ROOT).as_posix()
    match = HR_NAME.fullmatch(relative)
    if not match:
        raise ValueError(f'Unexpected RWTH manifest path: {path}')
    return match[1], int(match[2])


def selected_ids():
    return {
        '16MnCrS5': {frame * BLOCK_SIZE + y * 30 + x
                    for frame in range(FRAME_COUNT) for y in GRID for x in GRID},
        'DP/10mu': set(random.Random(42).sample(range(SOURCE_COUNTS['DP/10mu']), TARGET_COUNTS['DP/10mu'])),
        'DP/100mu': set(),
    }


def overlap_shifts(first, second, nominal):
    """Find exact pixel overlaps near a nominal (row, column) displacement."""
    import numpy as np

    size = 128
    matches = []
    for dy in range(nominal[0] - 4, nominal[0] + 5):
        for dx in range(nominal[1] - 4, nominal[1] + 5):
            y0, y1 = max(0, dy), min(size, size + dy)
            x0, x1 = max(0, dx), min(size, size + dx)
            if np.array_equal(first[y0:y1, x0:x1], second[y0-dy:y1-dy, x0-dx:x1-dx]):
                matches.append([dy, dx])
    return matches


def probe_frame(frame):
    import numpy as np
    from PIL import Image

    directory = HR_ROOT / '16MnCrS5/hr'
    start = frame * BLOCK_SIZE
    cached = {}

    def tile(tile_id):
        if tile_id not in cached:
            with Image.open(directory / f'{tile_id}.png') as image:
                value = np.array(image)
            if value.shape != (128, 128, 3) or not (
                np.array_equal(value[..., 0], value[..., 1]) and
                np.array_equal(value[..., 0], value[..., 2])
            ):
                raise ValueError(f'Unexpected 16MnCrS5 HR pixels in tile {tile_id}')
            cached[tile_id] = value[..., 0].copy()
        return cached[tile_id]

    matches, failures = [], []
    for row, column in PROBE_POINTS:
        anchor = start + row * 30 + column
        for delta, nominal in ((1, (0, 96)), (30, (96, 0))):
            positions = overlap_shifts(tile(anchor), tile(anchor + delta), nominal)
            if len(positions) == 1:
                matches.append([anchor, anchor + delta, *positions[0]])
            else:
                failures.append(dict(tile_ids=[anchor, anchor + delta], possible_shifts=positions))

    # Consecutive blocks must not match as consecutive rows of the same grid.
    if frame < FRAME_COUNT - 1:
        for column in (1, 14, 28):
            last_row = start + 29 * 30 + column
            positions = overlap_shifts(tile(last_row), tile(last_row + 30), (96, 0))
            if positions:
                failures.append(dict(tile_ids=[last_row, last_row + 30], unexpected_cross_block_shifts=positions))
    return dict(frame=frame, first_tile_id=start, last_tile_id=start + BLOCK_SIZE - 1,
                intra_block_matches=matches, boundary_pairs_tested=3 if frame < FRAME_COUNT - 1 else 0,
                failures=failures)


def verify_frames():
    with ThreadPoolExecutor(max_workers=2) as pool:
        evidence = list(pool.map(probe_frame, range(FRAME_COUNT)))
    uncertain = [item for item in evidence if item['failures']]
    if uncertain:
        raise ValueError(f'Unreliable RWTH frame mapping: {json.dumps(uncertain)}')
    return evidence


def revise_csv(source_dir, destination):
    chosen = selected_ids()
    found = {group: set() for group in SOURCE_COUNTS}
    rwth_counts = Counter()
    per_frame = Counter()
    sources, counts, reasons = {}, Counter(), Counter()
    with (source_dir / 'manifest.csv').open(newline='') as master, \
         (destination / 'manifest.csv').open('w', newline='') as out, \
         (destination / 'training_manifest.csv').open('w', newline='') as train:
        reader = csv.DictReader(master)
        manifest_writer = csv.DictWriter(out, reader.fieldnames)
        training_writer = csv.DictWriter(train, reader.fieldnames)
        manifest_writer.writeheader()
        training_writer.writeheader()
        for row in reader:
            source = row['source']
            if source == SOURCE:
                group, tile_id = hr_key(row['path'])
                if tile_id in found[group] or row['included'] != '1' or row['exclusion_reason']:
                    raise ValueError(f'Expected unique, currently included RWTH HR row: {row["path"]}')
                if (row['split'], row['role'], row['header_status'], row['width'], row['height']) != (
                    'train', 'derived_image', 'ok', '128', '128'
                ):
                    raise ValueError(f'Unexpected RWTH HR metadata: {row["path"]}')
                found[group].add(tile_id)
                if tile_id in chosen[group]:
                    rwth_counts[group] += 1
                    if group == '16MnCrS5':
                        per_frame[tile_id // BLOCK_SIZE] += 1
                else:
                    row['included'] = '0'
                    row['exclusion_reason'] = EXCLUSION[group]
            if row['included'] not in ('0', '1'):
                raise ValueError('Expected included=0 or included=1')
            include = int(row['included'])
            manifest_writer.writerow(row)
            if include:
                training_writer.writerow(row)
            per_source = sources.setdefault(source, dict(total_rows=0, included=0, excluded=0,
                **{f'included_ge_{threshold}': 0 for threshold in THRESHOLDS}))
            for counter in (counts, per_source):
                counter['total_rows'] += 1
                counter['included'] += include
                counter['excluded'] += 1 - include
            if include:
                for threshold in THRESHOLDS:
                    key = f'min_side_ge_{threshold}'
                    per_source[f'included_ge_{threshold}'] += int(row[key])
            reasons.update(filter(None, row['exclusion_reason'].split(';')))

    for group, count in SOURCE_COUNTS.items():
        if found[group] != set(range(count)):
            raise ValueError(f'Incomplete or noncontiguous audited RWTH HR IDs in {group}')
    if dict(rwth_counts) | {'DP/100mu': rwth_counts['DP/100mu']} != TARGET_COUNTS or per_frame != {
        frame: 100 for frame in range(FRAME_COUNT)
    }:
        raise ValueError('RWTH counts or per-frame tile selection differ from target')
    return counts, sources, reasons, per_frame


def verify_export(source_dir, destination):
    def other_rows(directory):
        return (row for row in rows(directory / 'training_manifest.csv') if row['source'] != SOURCE)

    if any(before != after for before, after in zip_longest(other_rows(source_dir), other_rows(destination))):
        raise ValueError('Non-RWTH rows changed in the training export')
    found = {group: set() for group in SOURCE_COUNTS}
    count = 0
    for row in rows(destination / 'training_manifest.csv'):
        count += 1
        if row['included'] != '1' or row['split'] != 'train':
            raise ValueError(f'Unexpected training export row: {row["image_id"]}')
        if row['source'] in ('nffa_sem_100', 'aachen_heerlen') and row['original_split'] != 'train':
            raise ValueError(f'Unexpected held-out row: {row["image_id"]}')
        if row['source'] == SOURCE:
            group, tile_id = hr_key(row['path'])
            if tile_id in found[group]:
                raise ValueError(f'Duplicate RWTH export row: {row["image_id"]}')
            found[group].add(tile_id)
    if found != selected_ids():
        raise ValueError('Training export does not contain exactly the selected RWTH IDs')
    return count


def write_revision(source_dir, destination, backup):
    summary_path = source_dir / 'manifest_summary.json'
    old = read_json(summary_path)
    fingerprints = {'manifest.csv': old['manifest_sha256'],
                    'training_manifest.csv': old['training_manifest_sha256']}

    def verify_input():
        if read_json(summary_path) != old or any(
            metadata_sha(source_dir / name) != expected for name, expected in fingerprints.items()
        ):
            raise ValueError('Input manifest does not match its saved summary')

    verify_input()
    evidence = verify_frames()
    print(f'Confirmed {len(evidence)} RWTH frame blocks; writing manifest revision', flush=True)
    counts, sources, reasons, per_frame = revise_csv(source_dir, destination)
    if (counts['total_rows'] != old['total_rows'] or
        counts['included'] != old['included'] - sum(SOURCE_COUNTS.values()) + sum(TARGET_COUNTS.values()) or
        {key: value for key, value in sources.items() if key != SOURCE} != {
            key: value for key, value in old['sources'].items() if key != SOURCE
        }
    ):
        raise ValueError('Unexpected source counts after RWTH selection')
    if verify_export(source_dir, destination) != counts['included']:
        raise ValueError('Training export count differs from the summary')

    evidence_path = destination / 'rwth_frame_verification.json'
    atomic_json(evidence_path, dict(source=str(HR_ROOT), grid=GRID,
        nominal_stride_pixels=96, drift_search_radius_pixels=4,
        known_source_names={56: 'SE_0', 87: 'SE_19', 122: 'SE_144', 125: 'SE_22', 145: 'SE_7'},
        frame_blocks=evidence))
    policy = dict(revision=REVISION, total_included=sum(TARGET_COUNTS.values()),
        by_group={group: dict(included=count, excluded=SOURCE_COUNTS[group]-count)
                  for group, count in TARGET_COUNTS.items()},
        frame_count=FRAME_COUNT, confirmed_frame_blocks=len(evidence), tiles_per_frame=100, grid=GRID,
        dp10mu_seed=42, frame_verification_file=evidence_path.name)
    policies = old.get('selection_policies', {}) | {SOURCE: policy}
    event = dict(action=REVISION, before_included=old['included'], after_included=counts['included'],
        input_manifest_sha256=fingerprints['manifest.csv'],
        input_training_manifest_sha256=fingerprints['training_manifest.csv'],
        previous_manifest_directory=str(backup or source_dir), selection=policy,
        old_and_new_non_rwth_training_rows_identical=True, training_started=False)
    training_sha = metadata_sha(destination / 'training_manifest.csv')
    summary = old | dict(counts) | dict(sources=sources, exclusion_reasons=dict(reasons),
        selection_policies=policies, curation_history=[*old.get('curation_history', []), event],
        manifest_sha256=metadata_sha(destination / 'manifest.csv'), training_manifest_sha256=training_sha)
    receipt = read_json(source_dir / 'receipt.draft.json') | dict(
        candidate_training_images=counts['included'], training_manifest_sha256=training_sha, selection_policies=policies)
    atomic_json(destination / 'manifest_summary.json', summary)
    atomic_json(destination / 'receipt.draft.json', receipt)
    atomic_json(destination / 'curation_rwth_hr26k_v0.json', event | {'selected_per_frame': per_frame})
    for path in source_dir.glob('curation_*.json'):
        shutil.copy2(path, destination / path.name)
    (destination / 'README.md').write_text((source_dir / 'README.md').read_text() + '\n'
        f'RWTH-Auswahl {REVISION} mit {policy["total_included"]:,} aktiven HR-Dateien. '
        f'Aktuell insgesamt {counts["included"]:,} eingeschlossen und {counts["excluded"]:,} ausgeschlossen.\n'
        'Die Verteilung und die Pixelprüfung stehen in curation_rwth_hr26k_v0.json '
        'und rwth_frame_verification.json.\n')
    verify_input()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest-dir', required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--output', type=Path, help='Write a separate revision')
    mode.add_argument('--backup-dir', type=Path, help='Archive input here and replace its active path')
    parser.add_argument('--status', type=Path, help='Update the existing V0 job status after publication')
    args = parser.parse_args()
    if args.status and not args.backup_dir:
        parser.error('--status requires --backup-dir')
    source = args.manifest_dir.absolute()
    backup = args.backup_dir.absolute() if args.backup_dir else None
    target = args.output.absolute() if args.output else source
    sibling = backup or target
    if sibling.parent != source.parent or sibling == source:
        parser.error('Output or backup must be a new sibling directory of the input')
    with (source.parent / f'.{source.name}.curation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if sibling.exists():
            raise FileExistsError(sibling)
        status = read_json(args.status) if args.status else None
        if status is not None and (status['status'] != 'complete' or Path(status['output']) != source):
            raise ValueError('Status does not describe the completed input manifest')
        with TemporaryDirectory(prefix=f'.{source.name}.{REVISION}.', dir=source.parent) as workspace:
            revision = Path(workspace) / 'revision'
            revision.mkdir()
            summary = write_revision(source, revision, backup)
            if backup:
                source.rename(backup)
            try:
                revision.rename(target)
            except BaseException:
                if backup:
                    backup.rename(source)
                raise
        if status is not None:
            atomic_json(args.status, status | dict(included=summary['included'], excluded=summary['excluded'],
                selection_revision=REVISION, current_summary=str(target / 'manifest_summary.json')))
    print(json.dumps(dict(output=str(target), rwth=TARGET_COUNTS, included=summary['included'],
                          excluded=summary['excluded']), indent=2))


if __name__ == '__main__':
    main()
