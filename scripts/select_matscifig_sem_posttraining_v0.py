"""Apply the MatSciFig review groups and native RGB limits to the SEM V0 manifest."""

import argparse
from collections import Counter
import csv
import fcntl
from itertools import zip_longest
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from scripts.build_sem_posttraining_manifest import THRESHOLDS, atomic_json, metadata_sha, read_json, rows


SOURCE = 'matscifig_sem'
REVISION = 'matscifig_diagram_white_color5_v0'
GROUP_REASONS = {
    'diagrammverdacht': 'excluded_matscifig_diagram_suspect',
    'grenzfaelle': 'excluded_matscifig_diagram_borderline',
}
WHITE_REASON = 'excluded_matscifig_white_over_80pct_rgb245'
COLOR_REASON = 'excluded_matscifig_strong_color_ge_5pct'


def exclusion_reasons(record, group):
    pixels = int(record['native_pixels'])
    white = int(record['near_white_pixels'])
    color = int(record['strong_color_pixels'])
    if record['error'] or pixels <= 0 or not (0 <= white <= pixels and 0 <= color <= pixels):
        raise ValueError(f'Invalid pixel-audit row {record["image_id"]}')
    # Integer comparisons distinguish exactly 5% (included in the rule)
    # and exactly 80% white (outside the rule).
    reasons = [GROUP_REASONS[group]] if group in GROUP_REASONS else []
    if 5 * white > 4 * pixels:
        reasons.append(WHITE_REASON)
    if 20 * color >= pixels:
        reasons.append(COLOR_REASON)
    return reasons


def select(pixel_audit, diagram_audit, export_sha):
    prior = read_json(pixel_audit / 'summary.json')
    provenance = next(rows(diagram_audit / 'provenance.csv'))
    pixel_path = pixel_audit / 'pixel_counts.csv'
    review_path = diagram_audit / 'review_samples.csv'
    hashes = {str(path): metadata_sha(path) for path in (pixel_path, review_path)}
    if (prior['training_export_sha256'] != export_sha or provenance['training_export_sha256'] != export_sha
            or hashes[str(pixel_path)] != provenance['input_pixel_counts_sha256']):
        raise ValueError('MatSciFig audits do not correspond to the input training export')
    groups = {}
    group_counts = Counter()
    for row in rows(review_path):
        group = row['review_group']
        if group not in GROUP_REASONS:
            continue
        key = row['image_id'], row['path']
        if row['error'] or key in groups:
            raise ValueError(f'Invalid or repeated contact-sheet row {key}')
        groups[key] = group
        group_counts[group] += 1
    if group_counts != Counter({group: 100 for group in GROUP_REASONS}):
        raise ValueError(f'Expected 100 images in each diagram contact-sheet group, got {group_counts}')

    decisions = {}
    for row in rows(pixel_path):
        key = row['image_id'], row['path']
        if key in decisions:
            raise ValueError(f'Duplicate pixel-audit row {key}')
        decisions[key] = exclusion_reasons(row, groups.get(key))
    if not groups.keys() <= decisions.keys() or len(decisions) != prior['images']:
        raise ValueError('Pixel audit does not cover all expected MatSciFig images')
    if any(metadata_sha(path) != expected for path, expected in hashes.items()):
        raise ValueError('Audit CSV changed while constructing selection')
    return decisions, hashes


def revise_csv(source, target, decisions):
    found = set()
    removed = Counter()
    total = active = 0
    with (source / 'manifest.csv').open(newline='') as original, \
         (target / 'manifest.csv').open('w', newline='') as master, \
         (target / 'training_manifest.csv').open('w', newline='') as training:
        reader = csv.DictReader(original)
        manifest = csv.DictWriter(master, reader.fieldnames)
        export = csv.DictWriter(training, reader.fieldnames)
        manifest.writeheader()
        export.writeheader()
        for row in reader:
            total += 1
            if row['included'] not in ('0', '1'):
                raise ValueError(f'Invalid included flag for {row["image_id"]}')
            if row['included'] == '1':
                active += 1
                if row['source'] == SOURCE:
                    key = row['image_id'], row['path']
                    if key in found or key not in decisions or row['exclusion_reason']:
                        raise ValueError(f'Unexpected active MatSciFig row {key}')
                    found.add(key)
                    if decisions[key]:
                        row['included'] = '0'
                        row['exclusion_reason'] = ';'.join(decisions[key])
                        removed['included'] += 1
                        for threshold in THRESHOLDS:
                            removed[f'included_ge_{threshold}'] += int(row[f'min_side_ge_{threshold}'])
            manifest.writerow(row)
            if row['included'] == '1':
                export.writerow(row)
    if found != decisions.keys():
        raise ValueError('Audited MatSciFig rows differ from the active master rows')
    expected = (
        row for row in rows(source / 'training_manifest.csv')
        if row['source'] != SOURCE or not decisions[row['image_id'], row['path']]
    )
    actual = rows(target / 'training_manifest.csv')
    if any(before != after for before, after in zip_longest(expected, actual)):
        raise ValueError('New training export differs from the exact requested selection')
    return total, active, removed


def write_revision(source, target, backup, pixel_audit, diagram_audit):
    old = read_json(source / 'manifest_summary.json')
    hashes = {name: old[key] for name, key in (
        ('manifest.csv', 'manifest_sha256'), ('training_manifest.csv', 'training_manifest_sha256')
    )}
    for name, expected in hashes.items():
        if metadata_sha(source / name) != expected:
            raise ValueError(f'Input {name} differs from its saved fingerprint')
    old_receipt = read_json(source / 'receipt.draft.json')
    if old_receipt['training_manifest_sha256'] != hashes['training_manifest.csv']:
        raise ValueError('Input training export differs from its draft receipt')
    decisions, audit_hashes = select(pixel_audit, diagram_audit, hashes['training_manifest.csv'])
    total, active, removed = revise_csv(source, target, decisions)
    if (total, active, len(decisions)) != (old['total_rows'], old['included'], old['sources'][SOURCE]['included']):
        raise ValueError('Input counts differ from the manifest summary')
    count = removed['included']
    included, excluded = active - count, total - active + count
    source_counts = old['sources'][SOURCE].copy()
    for field, decrease in removed.items():
        source_counts[field] -= decrease
    source_counts['excluded'] += count
    hits = Counter(reason for reasons in decisions.values() for reason in reasons)
    policy = dict(revision=REVISION, source=SOURCE,
                  diagram_contact_groups=list(GROUP_REASONS),
                  white_pixel='min(R,G,B) >= 245', white_image='white pixel fraction > 0.80',
                  strong_color_pixel='max(R,G,B) - min(R,G,B) >= 40',
                  color_image='strong-color pixel fraction >= 0.05',
                  excluded_unique=count, included=source_counts['included'], rule_counts=dict(hits),
                  overlap_counts=dict(Counter(';'.join(reasons) for reasons in decisions.values() if reasons)),
                  audit_csv_sha256=audit_hashes)
    policies = old.get('selection_policies', {}) | {SOURCE: policy}
    event = dict(action=REVISION, before_included=active, after_included=included,
                 input_manifest_sha256=hashes['manifest.csv'],
                 input_training_manifest_sha256=hashes['training_manifest.csv'],
                 previous_manifest_directory=str(backup), selection=policy)
    summary = old | dict(included=included, excluded=excluded, sources=old['sources'] | {SOURCE: source_counts},
                         exclusion_reasons=dict(Counter(old['exclusion_reasons']) + hits),
                         color_filter_applied=True, color_filter_sources=[SOURCE],
                         selection_policies=policies, curation_history=[*old.get('curation_history', []), event],
                         manifest_sha256=metadata_sha(target / 'manifest.csv'),
                         training_manifest_sha256=metadata_sha(target / 'training_manifest.csv'))
    receipt = old_receipt | dict(candidate_training_images=included, selection_policies=policies,
                                 training_manifest_sha256=summary['training_manifest_sha256'])
    updated = {'manifest.csv', 'training_manifest.csv', 'manifest_summary.json', 'receipt.draft.json', 'README.md'}
    for path in source.iterdir():
        if path.name not in updated:
            shutil.copy2(path, target / path.name)
    atomic_json(target / 'manifest_summary.json', summary)
    atomic_json(target / 'receipt.draft.json', receipt)
    atomic_json(target / f'curation_{REVISION}.json', event)
    readme = (source / 'README.md').read_text().replace(
        'Farbprüfung und weiteres Cleaning sind offen.',
        'Die MatSciFig-Auswahl enthält eine Farbprüfung. Weiteres Cleaning ist offen.',
    )
    (target / 'README.md').write_text(readme + f'\nMatSciFig-Auswahl `{REVISION}`: '
        f'{count} deaktiviert, {source_counts["included"]} aktiv. Insgesamt {included} Trainingsbilder.\n'
        f'Einzelregeln und Überschneidungen stehen in `curation_{REVISION}.json`.\n'
        f'Vorheriger Manifeststand: {backup}\n')
    if read_json(source / 'manifest_summary.json') != old or any(
        metadata_sha(source / name) != expected for name, expected in hashes.items()
    ):
        raise ValueError('Input manifest changed during curation')
    return summary, policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest-dir', type=Path, required=True)
    parser.add_argument('--backup-dir', type=Path, required=True)
    parser.add_argument('--pixel-audit', type=Path, required=True)
    parser.add_argument('--diagram-audit', type=Path, required=True)
    parser.add_argument('--status', type=Path, required=True)
    args = parser.parse_args()
    source, backup = args.manifest_dir.absolute(), args.backup_dir.absolute()
    if backup.parent != source.parent or backup == source:
        parser.error('Backup must be a new sibling of the active manifest directory')
    with (source.parent / f'.{source.name}.curation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if backup.exists():
            raise FileExistsError(backup)
        status = read_json(args.status)
        if status['status'] != 'complete' or Path(status['output']) != source:
            raise ValueError('Status does not describe the active manifest')
        with TemporaryDirectory(prefix=f'.{source.name}.{REVISION}.', dir=source.parent) as workspace:
            target = Path(workspace) / 'revision'
            target.mkdir()
            summary, policy = write_revision(source, target, backup, args.pixel_audit, args.diagram_audit)
            source.rename(backup)
            try:
                target.rename(source)
                atomic_json(args.status, status | dict(
                    included=summary['included'], excluded=summary['excluded'], selection_revision=REVISION,
                    current_summary=str(source / 'manifest_summary.json')))
            except BaseException:
                if source.exists():
                    source.rename(target)
                backup.rename(source)
                raise
    print(json.dumps(dict(removed=policy['excluded_unique'], matscifig=policy['included'],
                          included=summary['included'], rule_counts=policy['rule_counts']), indent=2))


if __name__ == '__main__':
    main()
