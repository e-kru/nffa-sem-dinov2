"""CPU-only manifest orchestration, original-size selection and held-out safety."""
import copy
import csv
import json
from pathlib import Path
from unittest import mock

from PIL import Image
import pytest

from scripts import build_sem_posttraining_manifest as builder


def table(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, list(records[0]))
        writer.writeheader(); writer.writerows(records)


@pytest.fixture
def inputs(tmp_path):
    root, audit = tmp_path/'datasets', tmp_path/'audit'
    raw = []
    for source, name, side, role, microscopy in [
        ('lithos', 'ppl.png', 256, 'microscopy_image', True),
        ('matscifig_sem', 'color.png', 512, 'microscopy_image', True),
        ('matscifig_sem', 'gray.png', 518, 'microscopy_image', True),
        ('matscifig_sem', 'small.png', 128, 'microscopy_image', True),
        ('matscifig_sem', 'too_small.png', 127, 'microscopy_image', True),
        ('matscifig_sem', 'mask.png', 1024, 'segmentation_mask', False),
    ]:
        raw.append(dict(dataset=source, path=str(root/source/name), relative_path=name,
                        is_microscopy=str(microscopy).lower(), role=role, role_evidence='fixture',
                        header_status='ok', width=str(side), height=str(side), mode='RGB', channels='3',
                        original_group='unknown', sample_group='unknown', split_hint='unknown'))
    audit.mkdir()
    table(audit/'images_manifest.csv', raw)
    summary = dict(discovery_complete=True, header_pending=0, image_files=len(raw), read_errors=0,
                   datasets=['lithos', 'matscifig_sem'], contract=dict(root=str(root)))
    builder.atomic_json(audit/'audit_summary.json', summary)
    builder.atomic_json(audit/'progress.json', dict(stage='complete'))

    nffa = root/'nffa_sem_100'
    nffa.mkdir(parents=True)
    names = [('train.png', 'train', 512), ('small.png', 'train', 128), ('tiny.png', 'train', 127),
             ('val.png', 'val', 1024), ('test.png', 'test', 1024)]
    for name, split, side in names:
        if split == 'train':
            Image.new('RGB', (side, side), (50, 20, 70)).save(nffa/name)
    manifest, split, inventory = tmp_path/'nffa.csv', tmp_path/'split_A.csv', tmp_path/'inventory.json'
    table(manifest, [dict(image_id=name, included='1', crop_mode='crop_v2') for name, _, _ in names])
    table(split, [dict(image_id=name, split=part, group_id='group_'+name) for name, part, _ in names])
    records = [dict(image_id=name, group_id='group_'+name, sha256='old_receipt_hash')
               for name, part, _ in names if part == 'train']
    builder.atomic_json(inventory, dict(root=str(nffa), count=len(records), records=records,
                         records_sha256=builder.canonical(records),
                         manifests=[dict(path=str(p), sha256=builder.metadata_sha(p)) for p in (manifest, split)]))

    aachen = root/'aachen_heerlen'
    aachen_manifest, reference = tmp_path/'aachen.csv', tmp_path/'aachen_reference.json'
    assignments = [dict(image_id=name+'.png', image_path=str(aachen/'images'/(name+'.png')),
                        mask_path=str(aachen/'masks'/(name+'.png')), split=part, group_id=name)
                   for name, part in [('a_train', 'train'), ('a_val', 'val'), ('a_test', 'test')]]
    table(aachen_manifest, assignments)
    verified = [dict(r, original_size=[1024, 768], image_sha256='old_image_hash', mask_sha256='old_mask_hash')
                for r in assignments if r['split'] != 'test']
    builder.atomic_json(reference, dict(cache_metadata=dict(contract=dict(nonscientific_smoke_test=False,
                         manifest_sha256=builder.metadata_sha(aachen_manifest), records=verified))))
    cfg = dict(minimum_side=128, training_resolution=224, header_workers=2,
               header_inspector=str(Path('scripts/audit_image_inventory.py').resolve()), audit_directories=[str(audit)],
               nffa=dict(root=str(nffa), manifest=str(manifest), split=str(split), inventory=str(inventory)),
               aachen=dict(root=str(aachen), manifest=str(aachen_manifest), reference=str(reference)))
    return cfg, tmp_path/'output'


def read(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def test_build_preserves_resolutions_flags_color_and_train_only_without_pixels_or_hashes(inputs):
    cfg, output = inputs
    # Reused audit/Aachen data must not reopen any images. Only three NFFA
    # training headers are inspected; test/validation files do not even exist.
    real_open = Image.open
    opened = []
    def opening(stream, *args, **kwargs):
        opened.append(stream.name)
        return real_open(stream, *args, **kwargs)
    with mock.patch.object(Image, 'open', side_effect=opening), \
         mock.patch.object(Image.Image, 'load', side_effect=AssertionError('No pixels decoded')):
        summary = builder.build(cfg, output)
    assert len(opened) == 3
    all_rows, train = read(output/'manifest.csv'), read(output/'training_manifest.csv')
    assert len(all_rows) == 10 and len(train) == summary['included'] == 6
    assert not any(r['source'] == 'lithos' for r in train)
    lithos = next(r for r in all_rows if r['source'] == 'lithos')
    assert lithos['included'] == '0' and 'excluded_source_lithos' in lithos['exclusion_reason']
    color = next(r for r in train if r['path'].endswith('/color.png'))
    assert color['width'] == color['height'] == '512'
    assert color['mode'] == 'RGB' and color['min_side_ge_512'] == '1' and color['min_side_ge_518'] == '0'
    small = next(r for r in train if r['source'] == 'matscifig_sem' and r['path'].endswith('/small.png'))
    assert small['included'] == '1' and small['min_side_ge_128'] == '1'
    assert all(small[f'min_side_ge_{n}'] == '0' for n in (224, 256, 448, 512, 518))
    assert all(r['original_split'] == 'train' for r in train if r['source'] in ('nffa_sem_100', 'aachen_heerlen'))
    assert not any(r['path'].endswith(('/val.png', '/test.png', '/a_val.png', '/a_test.png')) for r in all_rows)
    assert summary['training_resolution'] == 224 and summary['supported_training_resolutions'] == [224, 518]
    assert not summary['color_filter_applied'] and not summary['image_contents_rehashed']
    assert not builder.read_json(output/'receipt.draft.json')['training_ready']
    with pytest.raises(FileExistsError):
        builder.build(cfg, output)


def test_watch_waits_for_final_audit_then_builds_once(inputs, tmp_path):
    cfg, output = inputs
    audit = Path(cfg['audit_directories'][0])
    (audit/'audit_summary.json').rename(audit/'summary_waiting.json')
    calls = []
    def finish_audit(seconds):
        calls.append(seconds)
        assert not output.exists()
        (audit/'summary_waiting.json').rename(audit/'audit_summary.json')
    with mock.patch.object(builder, 'build', wraps=builder.build) as build:
        builder.wait_and_build(cfg, output, tmp_path/'job/status.json', sleep=finish_audit)
    assert calls == [30] and build.call_count == 1
    assert builder.read_json(tmp_path/'job/status.json')['status'] == 'complete'


def test_fields_rotation_selection_keeps_inventory_and_filters_only_fields(inputs):
    cfg, output = inputs
    audit = Path(cfg['audit_directories'][0])
    records = read(audit/'images_manifest.csv')
    # A similarly named file from another source must remain eligible.
    records[2]['path'] = str(Path(records[2]['path']).with_name('Rot90_control.png'))
    records[2]['relative_path'] = 'Rot90_control.png'
    for rotation in (0, 90, 180, 270):
        record = dict(records[1], dataset='fields_metal_cement',
                      relative_path=f'Rot{rotation}_123.png', role='derived_image')
        record['path'] = str(Path(records[1]['path']).parent.parent/
                             'fields_metal_cement'/record['relative_path'])
        records.append(record)
    table(audit/'images_manifest.csv', records)
    summary = builder.read_json(audit/'audit_summary.json')
    summary['datasets'].append('fields_metal_cement')
    summary['image_files'] = len(records)
    builder.atomic_json(audit/'audit_summary.json', summary)
    result = builder.build(cfg, output)
    master, train = read(output/'manifest.csv'), read(output/'training_manifest.csv')
    fields = [r for r in master if r['source'] == 'fields_metal_cement']
    assert len(fields) == 4
    assert [Path(r['path']).name for r in fields if r['included'] == '1'] == ['Rot0_123.png']
    assert all(r['included'] == '0' and r['exclusion_reason'] == 'excluded_fields_non_rot0'
               for r in fields if not Path(r['path']).name.startswith('Rot0_'))
    assert sum(r['source'] == 'fields_metal_cement' for r in train) == 1
    assert any(Path(r['path']).name == 'Rot90_control.png' for r in train)
    assert result['sources']['fields_metal_cement']['excluded'] == 3
    assert result['exclusion_reasons']['excluded_fields_non_rot0'] == 3
    assert result['selection_policies']['fields_metal_cement'] == 'Rot0_only'


def test_draft_refuses_partial_discovery_and_changed_benchmark_receipts(inputs):
    cfg, output = inputs
    audit = Path(cfg['audit_directories'][0])
    summary = builder.read_json(audit/'audit_summary.json')
    summary['discovery_complete'] = False
    builder.atomic_json(audit/'audit_summary.json', summary)
    with pytest.raises(ValueError, match='incomplete discovery'):
        builder.build(cfg, output)
    summary['discovery_complete'] = True
    builder.atomic_json(audit/'audit_summary.json', summary)
    Path(cfg['nffa']['split']).write_text('changed')
    with pytest.raises(ValueError, match='differs'):
        list(builder.nffa_rows(cfg))
    assert not output.exists()


@pytest.mark.parametrize('width,height,expected', [(128, 128, True), (127, 1000, False),
                                                   (518, 127, False), (512, 518, True)])
def test_minimum_uses_both_original_sides(width, height, expected):
    row = builder.manifest_row(dict(dataset='fixture', relative_path='x.png', path='/fixture/x.png',
                                   header_status='ok', width=width, height=height,
                                   is_microscopy=True, role='microscopy_image'), 'fixture')
    assert bool(row['included']) is expected
    assert bool(row['min_side_ge_518']) == (min(width, height) >= 518)
