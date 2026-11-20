"""Small real headers, synthetic byte duplicates, no model or raw dataset access."""
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
from unittest.mock import patch

from PIL import Image
import pytest

from scripts import audit_image_inventory as audit


def roots(tmp_path):
    root, output = tmp_path/'data', tmp_path/'result'
    for name in (*audit.DATASETS, *audit.EXCLUDED):
        (root/name).mkdir(parents=True)
    return root, output


def image(path, size=(128, 128), color=17, format='PNG'):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('L', size, color).save(path, format=format)
    return path


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def fixture(root):
    original = image(root/'cigrocksem/raw/data/shale/imgs/original.bmp', format='BMP')
    content = original.read_bytes()
    for target in (root/'cigrocksem/raw/data/shale/imgs/copy.bmp',
                   root/'cigrocksem/raw/data/shale/masks/original.bmp',
                   root/'microal/raw/Electron/SEM image/copied.bmp'):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    # True KIT overlap, retained as unknown (not silently released for SSL).
    for split in ('train', 'val'):
        image(root/f'kit_sem_powder/raw/sem_segmentation_ssl/{split}/Phenom_Images/0_0_BO.png', color=31)
        # Same size/name, DIFFERENT content: must not become a duplicate group.
        image(root/f'kit_sem_powder/raw/sem_segmentation_ssl/{split}/Acquisitions_first_run/Tile_r000_c000.bmp',
              color=61 if split=='train' else 62, format='BMP')
    single = image(root/'lithos/raw/ppl-0.png', size=(224, 448), color=77)
    single.write_bytes(single.read_bytes()+b'unique size tail for singleton test')
    image(root/'cross_microscope_steel/raw/Dataset/A_Bainite_1/38_A_Bainite_01.png', size=(96, 96))
    # Plain non-image, resource fork, and protected datasets must never be opened.
    (root/'lithos/raw/notes.txt').write_text('metadata')
    (root/'lithos/raw/._pretend.png').write_bytes(b'not an image')
    for dataset in audit.EXCLUDED:
        image(root/dataset/'protected.png')
    return single


def test_resolution_boundaries_and_roles():
    assert audit.resolution(127, 500) == (False, 'width<128', 'very_small')
    assert audit.resolution(96, 96)[1] == 'width<128;height<128'
    assert audit.resolution(128, 223)[2] == 'small'
    assert audit.resolution(224, 447)[2] == 'medium'
    assert audit.resolution(448, 448)[2] == 'large'
    assert audit.resolution(None, None)[0] is None
    assert audit.classify('microal', 'raw/Metallo/Heat-treated/label_test/a.png')[:2] == ('label_or_annotation', False)
    assert audit.classify('cigrocksem', 'raw/data/imgs/mask_a.png')[:2] == ('segmentation_mask', False)
    assert audit.classify('cigrocksem', 'raw/data/imgs/a_crop_3.png')[:2] == ('derived_image', True)
    assert audit.classify('cross_microscope_steel', 'raw/Dataset/A_Bainite_1/38_A_Bainite_01.png')[:2] == ('derived_image', True)
    assert audit.classify('kit_sem_powder', 'raw/sem_segmentation_ssl/train/Phenom_Images/0_BO.tiff')[:2] == ('unknown', False)
    assert audit.classify('microal', 'raw/something.png')[:2] == ('unknown', False)


def test_full_pipeline_no_decode_protected_paths_and_raw_immutability(tmp_path):
    root, output = roots(tmp_path)
    single = fixture(root)
    before = {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file()}
    real_scandir, real_open = audit.os.scandir, audit.os.open
    traversed = []
    def scandir(path):
        path = Path(path)
        assert not any(path.is_relative_to(root/x) for x in audit.EXCLUDED)
        assert path != root  # no listing of root to discover other datasets
        traversed.append(str(path))
        return real_scandir(path)
    def safe_open(path, *args, **kwargs):
        assert not any(Path(path).is_relative_to(root/x) for x in audit.EXCLUDED)
        return real_open(path, *args, **kwargs)
    with patch.object(audit.os, 'scandir', side_effect=scandir), patch.object(audit.os, 'open', side_effect=safe_open), \
         patch.object(Image.Image, 'load', side_effect=AssertionError('Pixels decoded')):
        assert audit.main(['--root', str(root), '--output', str(output), '--workers', '2']) == 0
    manifest = read_csv(output/'images_manifest.csv')
    assert len(manifest) == 10
    assert all(not any(x in r['path'] for x in audit.EXCLUDED) for r in manifest)
    singleton = next(r for r in manifest if r['path']==str(single))
    assert singleton['sha256']=='' and singleton['hash_status']=='skipped_unique_size'
    assert singleton['width']=='224' and singleton['height']=='448'
    assert singleton['mode']=='L' and singleton['channels']=='1'
    assert singleton['ssl_candidate']=='true'
    summary = json.loads((output/'audit_summary.json').read_text())
    assert summary['complete'] and summary['cross_dataset_groups']==1
    assert summary['cross_role_groups']==1
    assert summary['kit_train_val_duplicate_groups']==1
    assert summary['duplicate_groups']==2
    assert summary['duplicate_extra_files']==4  # 4-file group + 2-file group
    assert summary['files_hashed']+summary['skipped_unique_size']==len(manifest)
    duplicates = read_csv(output/'exact_duplicates.csv')
    assert len(duplicates)==6
    assert not any('Tile_r' in r['path'] for r in duplicates)
    cross = next(r for r in summary['dataset_summary'] if r['dataset']=='cross_microscope_steel')
    assert cross['too_small_all_roles']==1 and cross['microscopy_images']==1 and cross['microscopy_images_ge128']==0
    kit = next(r for r in summary['dataset_summary'] if r['dataset']=='kit_sem_powder')
    assert kit['unknown_roles']==2 and kit['microscopy_images_ge128']==2
    assert all((Path(p).read_bytes(), Path(p).stat().st_mtime_ns)==receipt for p,receipt in before.items())
    assert (output/'resolution_summary.csv').is_file() and (output/'report.md').is_file()


def test_resume_and_render_reuse_receipts_without_raw_image_reads(tmp_path):
    root, output = roots(tmp_path)
    fixture(root)
    args = ['--root', str(root), '--output', str(output)]
    assert audit.main(args)==0
    manifest = (output/'images_manifest.csv').read_bytes()
    with patch.object(audit, 'inspect_header', side_effect=AssertionError('Header reopened')), \
         patch.object(audit, 'hash_file', side_effect=AssertionError('Hash repeated')), \
         patch.object(audit.os, 'scandir', side_effect=AssertionError('Directory rescanned')):
        assert audit.main(args+['--resume'])==0
        with patch.object(audit, 'metadata_join', side_effect=AssertionError('Metadata reopened')):
            assert audit.main(args+['--resume', '--render-only'])==0
    assert (output/'images_manifest.csv').read_bytes()==manifest
    with pytest.raises(FileExistsError):
        audit.main(args)
    assert json.loads((output/'audit_summary.json').read_text())['complete']


def test_hashes_streaming_and_changed_files_rejected(tmp_path):
    root, output = roots(tmp_path)
    original = image(root/'lithos/ppl-0.bmp', size=(448, 448), format='BMP')
    db = audit.open_state(output, root)
    progress = audit.Progress(output, db)
    try:
        audit.discover(db, root, progress)
        row = dict(db.execute('SELECT * FROM files').fetchone())
        assert audit.hash_file(row, block_size=73)['sha256'] == hashlib.sha256(original.read_bytes()).hexdigest()
        original.write_bytes(original.read_bytes()+b'changed')
        assert audit.hash_file(row)['hash_status']=='hash_error'
        assert 'file_changed' in audit.inspect_header(row)['read_error']
    finally:
        db.close()


def test_interrupted_hash_phase_can_resume_at_committed_file(tmp_path):
    root, output = roots(tmp_path)
    for n in range(3):
        image(root/f'lithos/{n}/ppl-0.bmp', color=n, format='BMP')
    db = audit.open_state(output, root)
    progress = audit.Progress(output, db)
    try:
        audit.discover(db, root, progress)
        audit.process_pending(db, progress, 'headers', audit.inspect_header)
        audit.prepare_hashes(db)
        calls = []
        def interrupt(row):
            calls.append(row['path'])
            if len(calls)>1:
                raise KeyboardInterrupt()
            return audit.hash_file(row)
        with pytest.raises(KeyboardInterrupt):
            audit.process_pending(db, progress, 'hashes', interrupt, batch_size=1)
        assert db.execute("SELECT count(*) FROM files WHERE hash_status='hashed'").fetchone()[0]==1
    finally:
        db.close()
    with patch.object(audit, 'hash_file', wraps=audit.hash_file) as reader:
        assert audit.main(['--root',str(root),'--output',str(output),'--resume'])==0
        assert reader.call_count==2


def test_read_error_and_missing_directory_are_explicitly_incomplete(tmp_path):
    root, output = roots(tmp_path)
    (root/'lithos/broken.png').write_bytes(b'not a PNG')
    (root/'microal').rmdir()
    assert audit.main(['--root',str(root),'--output',str(output)])==2
    summary = json.loads((output/'audit_summary.json').read_text())
    assert summary['read_errors']==1 and summary['directory_errors']==1
    assert not summary['complete'] and not summary['exact_hash_check_complete']
    row = read_csv(output/'images_manifest.csv')[0]
    assert row['header_status']=='read_error' and row['resolution_eligible']=='' and row['ssl_candidate']=='false'
    assert 'UNVOLLSTÄNDIG' in (output/'report.md').read_text()


def test_symlinks_and_resource_forks_never_become_images(tmp_path):
    root, output = roots(tmp_path)
    target = image(root/'aachen_heerlen/protected.png')
    (root/'lithos/link.png').symlink_to(target)
    (root/'lithos/link_dir').symlink_to(target.parent, target_is_directory=True)
    (root/'lithos/._fake.png').write_bytes(b'resource fork')
    assert audit.main(['--root',str(root),'--output',str(output)])==2
    summary = json.loads((output/'audit_summary.json').read_text())
    assert summary['image_files']==0 and summary['scan_issues']==2
    assert len(read_csv(output/'all_files.csv'))==1
    assert all(r['kind']=='symlink_not_followed' for r in read_csv(output/'scan_issues.csv'))


def test_explicit_uhcs_metadata_join_and_no_fabricated_sample(tmp_path):
    root, output = roots(tmp_path)
    base = root/'uhcsdb/raw/mirror'
    image(base/'micrographs/micrograph1.png')
    image(base/'micrographs/micrograph2.png', color=18)
    (base/'metadata.csv').write_text('path,sample_key,detector\nmicrograph1.png,42.0,SE\n../../outside.png,99,BSE\n')
    assert audit.main(['--root',str(root),'--output',str(output)])==0
    rows = read_csv(output/'images_manifest.csv')
    assert rows[0]['sample_group']=='uhcsdb/sample_key=42.0' and rows[0]['modality']=='SEM/SE'
    assert rows[1]['sample_group']=='unknown'


def test_guardrails_and_resolution_distribution(tmp_path):
    root, output = roots(tmp_path)
    with pytest.raises(ValueError, match='separate'):
        audit.open_state(root/'lithos/output', root)
    with pytest.raises(SystemExit):
        audit.main(['--workers','8'])
    image(root/'lithos/ppl-0.png', size=(128,256))
    assert audit.main(['--root',str(root),'--output',str(output)])==0
    rows = read_csv(output/'resolution_summary.csv')
    assert rows[0]['width_median']=='128.0' and rows[0]['pixels_mean']=='32768.0'
    db=sqlite3.connect(output/'audit_state.sqlite3')
    contract=json.loads(db.execute("SELECT value FROM meta WHERE key='contract'").fetchone()[0])
    contract['minimum_side']=99
    with db:
        db.execute("UPDATE meta SET value=? WHERE key='contract'",(json.dumps(contract),))
    db.close()
    with pytest.raises(ValueError,match='differ'):
        audit.main(['--root',str(root),'--output',str(output),'--resume'])


def test_combined_header_hash_uses_one_open_and_matches_separate_receipts(tmp_path):
    root, output = roots(tmp_path)
    first=image(root/'lithos/1/ppl-0.bmp',format='BMP')
    image(root/'lithos/2/ppl-0.bmp',format='BMP')
    db=audit.open_state(output,root)
    try:
        audit.discover(db,root,audit.Progress(output,db))
        original=dict(db.execute('SELECT * FROM files WHERE path=?',(str(first),)).fetchone())
        separate=audit.inspect_header(original)
        expected_hash=audit.hash_file(original)['sha256']
        audit.prepare_hashes(db)
        combined=dict(db.execute('SELECT * FROM files WHERE path=?',(str(first),)).fetchone())
        with patch.object(audit,'raw_file',wraps=audit.raw_file) as opened, \
             patch.object(Image.Image,'load',side_effect=AssertionError('Pixels decoded')):
            result=audit.inspect_header(combined)
            assert opened.call_count==1
        assert result['sha256']==expected_hash and result['hash_status']=='hashed'
        for key,value in separate.items():
            if key!='header_seconds': assert result[key]==value
    finally: db.close()


def test_broken_header_can_have_valid_combined_byte_hash(tmp_path):
    root, output = roots(tmp_path)
    for name in ('one.png','two.png'): (root/'lithos'/name).write_bytes(b'not a PNG')
    assert audit.main(['--root',str(root),'--output',str(output)])==2
    rows=read_csv(output/'images_manifest.csv')
    assert all(r['header_status']=='read_error' and r['hash_status']=='hashed' for r in rows)
    assert all(r['sha256']==hashlib.sha256(b'not a PNG').hexdigest() for r in rows)
    summary=json.loads((output/'audit_summary.json').read_text())
    assert summary['read_errors']==2 and summary['hash_errors']==0


def test_sha_cancellation_persists_and_does_not_claim_deduplication(tmp_path):
    root, output = roots(tmp_path)
    fixture(root)
    args=['--root',str(root),'--output',str(output)]
    with patch.object(audit,'hash_stream',side_effect=AssertionError('SHA must be cancelled')), \
         patch.object(audit,'hash_file',side_effect=AssertionError('SHA must be cancelled')):
        assert audit.main(args+['--skip-hashes'])==0
        assert audit.main(args+['--resume'])==0
    result=json.loads((output/'audit_summary.json').read_text())
    assert result['inventory_complete'] and result['complete']
    assert not result['exact_hash_check_complete'] and result['hash_check_cancelled_by_user']
    assert result['files_hashed']==0 and result['hashes_skipped_by_request']==10
    assert result['microscopy_candidates_unique_global'] is None
    assert 'AUF NUTZERWUNSCH ABGEBROCHEN' in (output/'report.md').read_text()
    assert all(r['hash_status']=='skipped_by_request' for r in read_csv(output/'images_manifest.csv'))


def test_cancel_preserves_existing_hash_and_marks_pending_headers_honestly(tmp_path):
    root, output = roots(tmp_path)
    for n in range(2): image(root/f'lithos/{n}/ppl-0.bmp',format='BMP')
    db=audit.open_state(output,root)
    progress=audit.Progress(output,db)
    try:
        audit.discover(db,root,progress)
        audit.prepare_hashes(db)
        row=dict(db.execute('SELECT * FROM files ORDER BY rowid LIMIT 1').fetchone())
        receipt=audit.inspect_header(row)
        with db: audit.update_row(db,row['path'],receipt)
        assert not audit.hash_policy(db,True)
        result=audit.export(db,output,progress)
        assert not result['inventory_complete'] and result['header_pending']==1
        assert result['files_hashed']==1 and result['hashes_skipped_by_request']==1
        assert not result['exact_hash_check_complete']
        assert db.execute('SELECT sha256 FROM files WHERE path=?',(row['path'],)).fetchone()[0]==receipt['sha256']
        all_roles=next(r for r in read_csv(output/'resolution_summary.csv') if r['role']=='ALL_ROLES')
        assert all_roles['read_errors']=='0' and all_roles['headers_pending']=='1'
    finally: db.close()
