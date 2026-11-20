"""Publish the reviewed 20-file exclusion as a separate, immutable V0 revision."""

import argparse
from collections import Counter
import csv
import fcntl
from itertools import zip_longest
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from PIL import Image

from scripts.build_sem_posttraining_manifest import THRESHOLDS, atomic_json, metadata_sha, read_json, rows
from src.ssl.upstream import ROOT


SOURCE = ROOT/'outputs/sem_corpus/sem_posttraining_v0'
TARGET = SOURCE.with_name('sem_posttraining_v0_decode_trim20')
PREVIOUS = ROOT/'outputs/sem_ssl/sem_v0_vits14_aug_a_300epochs_seed42/preflight.json'
OVERSIZE = 'excluded_pillow_hard_pixel_limit'
WHITE_JLU = 'excluded_reviewed_white_jlu_sample'


def selection(export, proof):
    losses = {row['path'] for row in proof['sampled_decode']['lost_contrast_examples']
              if row['source']=='jlu_cathodes' and row['aug_a_gray_range']==[255,255]}
    if len(losses) != 3:
        raise ValueError('Expected exactly the three reviewed white JLU examples')
    if Image.MAX_IMAGE_PIXELS is None:
        raise ValueError('Expected the audited Pillow pixel limit')
    limit = 2*Image.MAX_IMAGE_PIXELS
    large, selected = Counter(), {}
    for row in rows(export):
        path, source = row['path'], row['source']
        if int(row['width'])*int(row['height']) > limit:
            large[source] += 1
            selected[path] = (row['image_id'], OVERSIZE)
        if path in losses:
            if source != 'jlu_cathodes' or path in selected:
                raise ValueError('Unexpected reviewed JLU match')
            selected[path] = (row['image_id'], WHITE_JLU)
    if dict(large) != proof['above_decode_pixel_limit'] or sum(large.values()) != 17 or len(selected) != 20:
        raise ValueError('Export does not contain the audited 17 oversized and three white files')
    return selected


def write_csvs(source, target, selected, summary):
    seen, details = set(), []
    source_counts = summary['sources']
    total = active = 0
    with (source/'manifest.csv').open(newline='') as stream, \
         (target/'manifest.csv').open('w',newline='') as master, \
         (target/'training_manifest.csv').open('w',newline='') as export:
        reader = csv.DictReader(stream)
        inventory, train = (csv.DictWriter(dest,reader.fieldnames) for dest in (master,export))
        inventory.writeheader()
        train.writeheader()
        for row in reader:
            total += 1
            active += row['included']=='1'
            path = row['path']
            if path in selected:
                image_id, reason = selected[path]
                if path in seen or row['included']!='1' or row['image_id']!=image_id or row['exclusion_reason']:
                    raise ValueError(f'Unexpected master row for {path}')
                seen.add(path)
                details.append({'image_id':image_id,'source':row['source'],'path':path,'reason':reason,
                                'width':row['width'],'height':row['height']})
                stats = source_counts[row['source']]
                stats['included'] -= 1
                stats['excluded'] += 1
                for threshold in THRESHOLDS:
                    stats[f'included_ge_{threshold}'] -= int(row[f'min_side_ge_{threshold}'])
                row['included'],row['exclusion_reason'] = '0',reason
            inventory.writerow(row)
            if row['included']=='1':
                train.writerow(row)
    if seen!=selected.keys() or (total,active)!=(summary['total_rows'],summary['included']):
        raise ValueError('Master counts or exact selection differ')
    expected = (row for row in rows(source/'training_manifest.csv') if row['path'] not in selected)
    if any(left!=right for left,right in zip_longest(expected,rows(target/'training_manifest.csv'))):
        raise ValueError('Unrequested change to the training export')
    summary['included'] -= len(selected)
    summary['excluded'] += len(selected)
    summary['exclusion_reasons'] = dict(Counter(summary['exclusion_reasons'])+
                                        Counter(reason for _,reason in selected.values()))
    return details


def curate(source, target, preflight):
    source, target = source.resolve(),target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with (source.parent/f'.{source.name}.curation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        if target.exists():
            raise FileExistsError(target)
        proof = read_json(preflight)
        summary = read_json(source/'manifest_summary.json')
        receipt = read_json(source/'receipt.draft.json')
        before = {name:metadata_sha(source/f'{name}.csv') for name in ('manifest','training_manifest')}
        if (any(digest!=summary[f'{name}_sha256'] for name,digest in before.items())
                or before['training_manifest']!=receipt['training_manifest_sha256']
                or before['training_manifest']!=proof['identity']['manifest_sha256']):
            raise ValueError('Previously audited V0 fingerprints differ')
        selected = selection(source/'training_manifest.csv',proof)
        with TemporaryDirectory(prefix='.decode_trim20_',dir=target.parent) as tmp:
            stage = Path(tmp)/'revision'
            stage.mkdir()
            details = write_csvs(source,stage,selected,summary)
            for name in before:
                summary[f'{name}_sha256'] = metadata_sha(stage/f'{name}.csv')
                if metadata_sha(source/f'{name}.csv')!=before[name]:
                    raise ValueError('Original V0 export changed during publication')
            shutil.copy2(preflight,stage/'reviewed_preflight.json')
            proof_sha = metadata_sha(stage/'reviewed_preflight.json')
            event = dict(action='decode_trim20_v0', previous_manifest_directory=str(source),
                         previous_training_manifest_sha256=before['training_manifest'],
                         reviewed_preflight_sha256=proof_sha, excluded=details)
            summary.setdefault('curation_history',[]).append(event)
            summary['selection_policies']['decode_trim20_v0'] = dict(
                removed=20, exclusions=dict(Counter(x['reason'] for x in details)),
                source=str(source), reviewed_preflight_sha256=proof_sha)
            receipt.update(candidate_training_images=summary['included'],
                           training_manifest_sha256=summary['training_manifest_sha256'],
                           selection_policies=summary['selection_policies'])
            atomic_json(stage/'manifest_summary.json',summary)
            atomic_json(stage/'receipt.draft.json',receipt)
            atomic_json(stage/'curation_decode_trim20_v0.json',event)
            atomic_json(stage/'execution_policy.json',dict(
                schema='sem_v0_train_only_decode_trim20_v1', manifest_sha256=summary['training_manifest_sha256'],
                reviewed_preflight_sha256=proof_sha, nffa_training='split_A_train', nffa_validation='split_A_exact_groups',
                aachen='train_groups_only', cross_source_leakage_review='incomplete_operator_accepted',
                rationale='Retain the training rows and accept the disclosed audit limitations after the reviewed 20-file exclusion.'))
            excluded = {'manifest.csv','training_manifest.csv','manifest_summary.json','receipt.draft.json','README.md'}
            for path in source.iterdir():
                if path.is_file() and path.name not in excluded:
                    shutil.copy2(path,stage/path.name)
            (stage/'README.md').write_text((source/'README.md').read_text()+
                f'\nDecode-Revision trim20 mit {summary["included"]} aktiven Bildern.\n'
                'Einzelne Ausschlüsse stehen in curation_decode_trim20_v0.json.\n'
                f'Verknüpfter Originalexport: {source}\n')
            stage.rename(target)
    print(f'{summary["included"]} aktive Bilder in {target}',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=SOURCE)
    parser.add_argument('--output',type=Path,default=TARGET)
    evidence = TARGET/'reviewed_preflight.json'
    parser.add_argument('--preflight',type=Path,default=evidence if evidence.is_file() else PREVIOUS)
    args = parser.parse_args()
    curate(args.source,args.output,args.preflight)


if __name__ == '__main__':
    main()
