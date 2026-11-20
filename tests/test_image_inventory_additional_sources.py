"""Five additional sources: explicit scope, original headers, conservative roles."""
import csv
import json
from pathlib import Path
from unittest.mock import patch

from PIL import Image
import pytest

from scripts import audit_image_inventory as audit
from test_image_inventory_audit import image, read_csv


def test_selected_new_sources_headers_roles_groups_and_no_hash_or_pixels(tmp_path):
    root, output = tmp_path/'datasets', tmp_path/'audit'
    for name in audit.ADDITIONAL_DATASETS:
        (root/name).mkdir(parents=True)
    for name in (*audit.DATASETS, *audit.EXCLUDED):
        image(root/name/'never_open.png')
    examples = [
        ('drill_cuttings_sem', 'raw/Labels/RawImage_downscaled/TS_01_downscaled.tif'),
        ('drill_cuttings_sem', 'raw/Labels/Labeled_images/TS_01_labels.tif'),
        ('drill_cuttings_sem', 'raw/Splits/Training_Set/DBM/TS_01_101_2248_11039_2830_11320.tif'),
        ('drill_cuttings_sem', 'raw/Splits/Test_Set/DBM/TS_01_103_2291_19224_2708_19409.tif'),
        ('fields_metal_cement', 'raw/SEM_flyash_cement_mortar/Rot0_17.png'),
        ('fields_metal_cement', 'raw/SEM_flyash_cement_mortar/Rot90_17.png'),
        ('porous_zirconia_fibsem', 'raw/stack/stack/Slice_001.tif'),
        ('porous_zirconia_fibsem', 'raw/stack/stack/Slice_002.tif'),
        ('rwth_steel_sem', 'hr_images/SR for SEM/data/DP/10mu/hr/17.png'),
        ('rwth_steel_sem', 'hr_images/SR for SEM/data/DP/10mu/lr/17.png'),
        ('matscifig_sem', 'images/0000000_ni_alloy_img0_A.jpg'),
        ('matscifig_sem', 'images/0000001_ni_alloy_img0_B.jpg'),
        ('matscifig_sem', 'images/0000002_unlisted.jpg'),
        ('matscifig_sem', 'images/0000003_invalid.jpg'),
    ]
    for dataset, relative in examples:
        image(root/dataset/relative, size=(224, 256))
    (root/'porous_zirconia_fibsem/archive.zip').write_bytes(b'never unpack')
    with (root/'matscifig_sem/manifest.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['path', 'source_file', 'doi', 'subtype_valid'])
        writer.writeheader()
        for i, panel in enumerate(('A', 'B')):
            writer.writerow(dict(path=str(root/'matscifig_sem'/f'images/{i:07d}_ni_alloy_img0_{panel}.jpg'),
                                 source_file=f'images/ni_alloy_img0_{panel}.jpg', doi='10.1234/ARTICLE', subtype_valid='True'))
        writer.writerow(dict(path='images/0000003_invalid.jpg', source_file='invalid.jpg', doi='', subtype_valid='False'))
        writer.writerow(dict(path=str(root/'nffa_sem_100/never_open.png'), source_file='forbidden.jpg', doi='', subtype_valid='True'))
    # Tests may create/read fixture pixels; the audit itself must read headers only.
    before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    real_scandir, real_open = audit.os.scandir, audit.os.open
    allowed = [root/name for name in audit.ADDITIONAL_DATASETS]
    def scandir(path):
        assert any(Path(path).is_relative_to(base) for base in allowed)
        return real_scandir(path)
    def raw_open(path, *args, **kwargs):
        assert any(Path(path).is_relative_to(base) for base in allowed)
        return real_open(path, *args, **kwargs)
    args = ['--root', str(root), '--output', str(output), '--datasets', *audit.ADDITIONAL_DATASETS, '--skip-hashes']
    with patch.object(audit.os, 'scandir', side_effect=scandir), patch.object(audit.os, 'open', side_effect=raw_open), \
         patch.object(Image.Image, 'load', side_effect=AssertionError('No raster decoding')), \
         patch.object(audit, 'hash_stream', side_effect=AssertionError('No image SHA')), \
         patch.object(audit, 'hash_file', side_effect=AssertionError('No image SHA')):
        assert audit.main(args) == 0
    assert all(p.read_bytes() == content for p, content in before.items())
    summary = json.loads((output/'audit_summary.json').read_text())
    assert summary['datasets'] == list(audit.ADDITIONAL_DATASETS)
    assert summary['complete'] and summary['files_hashed'] == 0
    rows = {(row['dataset'], row['relative_path']): row for row in read_csv(output/'images_manifest.csv')}
    assert len(rows) == len(examples)
    def row(dataset, end):
        return next(r for (ds, path), r in rows.items() if ds == dataset and path.endswith(end))
    assert row('drill_cuttings_sem', 'TS_01_downscaled.tif')['ssl_candidate'] == 'true'
    assert row('drill_cuttings_sem', 'TS_01_labels.tif')['ssl_candidate'] == 'false'
    train, test = (row('drill_cuttings_sem', suffix) for suffix in
                   ('TS_01_101_2248_11039_2830_11320.tif', 'TS_01_103_2291_19224_2708_19409.tif'))
    assert train['split_hint'] == 'train' and test['split_hint'] == 'test'
    assert train['original_group'] == test['original_group'] == 'TS_01'
    rot0, rot90 = row('fields_metal_cement', 'Rot0_17.png'), row('fields_metal_cement', 'Rot90_17.png')
    assert rot0['role'] == rot90['role'] == 'derived_image'
    assert rot0['original_group'] == rot90['original_group']
    first, second = row('porous_zirconia_fibsem', 'Slice_001.tif'), row('porous_zirconia_fibsem', 'Slice_002.tif')
    assert first['original_group'] == second['original_group'] and first['slice_id'] == '001'
    assert row('rwth_steel_sem', '/hr/17.png')['ssl_candidate'] == 'true'
    assert row('rwth_steel_sem', '/lr/17.png')['ssl_candidate'] == 'false'
    a, b = row('matscifig_sem', '_img0_A.jpg'), row('matscifig_sem', '_img0_B.jpg')
    assert a['sample_group'] == b['sample_group'] == 'doi=10.1234/article'
    assert a['original_group'] == b['original_group'] == 'doi=10.1234/article/ni_alloy_img0'
    assert a['ssl_candidate'] == b['ssl_candidate'] == 'true'
    assert row('matscifig_sem', 'unlisted.jpg')['ssl_candidate'] == 'false'
    assert row('matscifig_sem', 'invalid.jpg')['ssl_candidate'] == 'false'
    assert all(r['width'] == '224' and r['height'] == '256' for r in rows.values())
    report = (output/'report.md').read_text()
    assert '--datasets '+' '.join(audit.ADDITIONAL_DATASETS) in report
    assert 'DOI ist eine Publikationsgruppe' in report


@pytest.mark.parametrize('selection', [('nffa_sem_100',), ('../outside',), ('unknown',),
                                      ('matscifig_sem', 'matscifig_sem'), ()])
def test_internal_dataset_selection_rejects_unapproved_duplicate_or_empty_roots(tmp_path, selection):
    with pytest.raises(ValueError, match='allowlist'):
        audit.open_state(tmp_path/'output', tmp_path/'data', datasets=selection)
    assert not (tmp_path/'output').exists()


def test_selected_scope_is_bound_on_resume_and_sha_skip_survives_discovery_interrupt(tmp_path):
    root, output = tmp_path/'data', tmp_path/'output'
    source = 'fields_metal_cement'
    image(root/source/'raw/SEM_flyash_cement_mortar/Rot0_1.png')
    args = ['--root', str(root), '--output', str(output), '--datasets', source]
    with patch.object(audit, 'discover', side_effect=KeyboardInterrupt):
        assert audit.main(args+['--skip-hashes']) == 130
    with patch.object(audit, 'hash_stream', side_effect=AssertionError('No SHA after resume')):
        assert audit.main(args+['--resume']) == 0
    with pytest.raises(ValueError, match='differ'):
        audit.main(['--root', str(root), '--output', str(output), '--datasets', 'matscifig_sem', '--resume'])
    summary = json.loads((output/'audit_summary.json').read_text())
    assert summary['datasets'] == [source] and summary['files_hashed'] == 0
