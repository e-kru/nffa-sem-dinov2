"""Derive a new V0 export by excluding reproducibly unreadable image files."""

import argparse
from collections import Counter
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from scripts.audit_sem_v0_decode import decode
from scripts.build_sem_posttraining_manifest import atomic_json, metadata_sha, read_json
from scripts.select_sem_v0_decode_revision import TARGET as SOURCE, write_csvs


TARGET = SOURCE.with_name('sem_posttraining_v0_decode_verified')
REASON = 'excluded_confirmed_pixel_decode_failure'
PROTECTED = {'nffa_sem_100', 'aachen_heerlen'}


def curate(source, target, audit_path):
    source, target = source.resolve(), target.resolve()
    if target.exists():
        raise FileExistsError(target)
    audit = read_json(audit_path)
    summary = read_json(source/'manifest_summary.json')
    receipt = read_json(source/'receipt.draft.json')
    policy = read_json(source/'execution_policy.json')
    audit_sha = metadata_sha(audit_path)
    export_sha = metadata_sha(source/'training_manifest.csv')
    source_counts = {name:v['included'] for name,v in summary['sources'].items() if v['included']}
    if (audit['schema'] != 'sem-v0-full-pillow-decode-v1' or audit['manifest_sha256'] != export_sha
            or summary['training_manifest_sha256'] != export_sha
            or receipt['training_manifest_sha256'] != export_sha or policy['manifest_sha256'] != export_sha
            or audit['checked'] != summary['included'] or audit['source_counts'] != source_counts
            or audit['transient_failures'] or metadata_sha(source/'manifest.csv') != summary['manifest_sha256']):
        raise ValueError('Full decode audit does not cover the unchanged parent export')
    failures = audit['failures']
    if any(row['source'] in PROTECTED for row in failures):
        raise ValueError('A protected NFFA or Aachen image requires recovery before publication')
    for row in failures:
        if row['error_type'] in ('PermissionError', 'FileNotFoundError') or decode(row) != row:
            raise ValueError(f'Cannot confirm a persistent decode failure: {row["path"]}')
    selected = {row['path']:(row['image_id'],REASON) for row in failures}
    if len(selected) != len(failures):
        raise ValueError('Duplicate decode failure paths')
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.decode_verified_',dir=target.parent) as tmp:
        stage = Path(tmp)/'revision'
        stage.mkdir()
        details = write_csvs(source,stage,selected,summary)
        if {row['path']:(row['image_id'],row['source']) for row in details} != {
                row['path']:(row['image_id'],row['source']) for row in failures}:
            raise ValueError('Reported decode failures differ from selected rows')
        for name in ('manifest','training_manifest'):
            if metadata_sha(source/f'{name}.csv') != summary[f'{name}_sha256']:
                raise ValueError('Parent V0 changed during publication')
            summary[f'{name}_sha256'] = metadata_sha(stage/f'{name}.csv')
        selection = dict(audit_sha256=audit_sha, parent_training_manifest_sha256=export_sha,
                         checked=audit['checked'], excluded=len(failures))
        event = dict(action='full_decode_v1', previous_manifest_directory=str(source),
                     **selection, exclusions=failures)
        summary['curation_history'].append(event)
        summary['selection_policies']['full_decode_v1'] = selection
        receipt.update(candidate_training_images=summary['included'],
                       training_manifest_sha256=summary['training_manifest_sha256'],
                       selection_policies=summary['selection_policies'])
        policy['manifest_sha256'] = summary['training_manifest_sha256']
        policy['full_decode_audit_sha256'] = audit_sha
        atomic_json(stage/'manifest_summary.json',summary)
        atomic_json(stage/'receipt.draft.json',receipt)
        atomic_json(stage/'execution_policy.json',policy)
        atomic_json(stage/'curation_full_decode_v1.json',event)
        shutil.copyfile(audit_path,stage/'full_decode_audit.json')
        if metadata_sha(stage/'full_decode_audit.json') != audit_sha:
            raise ValueError('Audit receipt changed during publication')
        replaced = {p.name for p in stage.iterdir()} | {'README.md'}
        for path in source.iterdir():
            if path.is_file() and path.name not in replaced:
                shutil.copy2(path,stage/path.name)
        (stage/'README.md').write_text((source/'README.md').read_text()+
            f'\nPillow-Vollprüfung mit {audit["checked"]} Dateien. '
            f'{len(failures)} bestätigte Dekodierfehler, {summary["included"]} aktive Bilder.\n')
        stage.rename(target)
    print(f'{summary["included"]:,} aktive Bilder in {target}, Ausschlüsse {dict(Counter(r["source"] for r in failures))}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=TARGET)
    args = parser.parse_args()
    curate(args.source,args.output,args.audit)


if __name__ == '__main__':
    main()
