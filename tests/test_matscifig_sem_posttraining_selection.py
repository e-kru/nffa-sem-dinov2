import csv

from scripts.build_sem_posttraining_manifest import THRESHOLDS, rows
from scripts.select_matscifig_sem_posttraining_v0 import (
    COLOR_REASON,
    GROUP_REASONS,
    SOURCE,
    WHITE_REASON,
    exclusion_reasons,
    revise_csv,
)


def test_pixel_rule_boundaries_and_overlapping_reasons():
    record = dict(image_id='example', error='', native_pixels=1000,
                  near_white_pixels=800, strong_color_pixels=49)
    assert exclusion_reasons(record, None) == []
    record['strong_color_pixels'] = 50
    assert exclusion_reasons(record, None) == [COLOR_REASON]
    record['near_white_pixels'] = 801
    assert exclusion_reasons(record, 'grenzfaelle') == [
        GROUP_REASONS['grenzfaelle'], WHITE_REASON, COLOR_REASON,
    ]


def test_revision_updates_only_target_rows_and_preserves_existing_exclusions(tmp_path):
    source = tmp_path / 'before'
    target = tmp_path / 'after'
    source.mkdir()
    target.mkdir()
    flags = {f'min_side_ge_{threshold}': str(int(threshold <= 224)) for threshold in THRESHOLDS}
    base = dict(source=SOURCE, included='1', exclusion_reason='', **flags)
    remove = dict(base, image_id='remove', path='/remove')
    keep = dict(base, image_id='keep', path='/keep')
    other = dict(base, source='aachen_heerlen', image_id='other', path='/other')
    already_excluded = dict(base, image_id='small', path='/small', included='0', exclusion_reason='small')
    for name, records in (
        ('manifest.csv', (remove, keep, other, already_excluded)),
        ('training_manifest.csv', (remove, keep, other)),
    ):
        with (source / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(base) + ['image_id', 'path'])
            writer.writeheader()
            writer.writerows(records)

    total, active, removed = revise_csv(
        source, target, {('remove', '/remove'): [WHITE_REASON, COLOR_REASON], ('keep', '/keep'): []},
    )

    assert (total, active) == (4, 3)
    assert removed['included'] == removed['included_ge_224'] == 1
    assert removed['included_ge_256'] == 0
    assert list(rows(target / 'manifest.csv')) == [
        dict(remove, included='0', exclusion_reason=f'{WHITE_REASON};{COLOR_REASON}'),
        keep, other, already_excluded,
    ]
    assert list(rows(target / 'training_manifest.csv')) == [keep, other]
