"""Stdlib-only tests: bounded scans, conservative labels, no pixel reads or mutation."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import csv
import json

from scripts.audit_datasets_fast import classify, identifiers, main, scan_dataset, summary, write_artifacts


class FastAuditTests(unittest.TestCase):
    def test_entry_and_filename_budgets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root/'lithos'
            base.mkdir()
            for i in range(50):
                (base/f'{i}.jpg').write_bytes(b'not an image; must never be decoded')
            result = scan_dataset(root, 'lithos', 17, entries_per_directory=30)
            self.assertEqual(result['inspected_entries'], 17)
            self.assertEqual(len(result['files']), 17)
            self.assertFalse(result['complete'])
            self.assertTrue(result['entry_budget_reached'])
            result = scan_dataset(root, 'lithos', 50, max_examples=3)
            self.assertEqual(len(result['files']), 3)
            self.assertLessEqual(result['inspected_entries'], 3)
            self.assertEqual(summary(result)['reported_image_count'], 211604)
            self.assertEqual(summary(result)['modality'], 'unknown')

    def test_depth_symlinks_and_metadata_preview(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root/'lithos'
            base.mkdir()
            (base/'metadata.json').write_text('x'*10000)
            (base/'images').mkdir()
            (base/'images'/'deep').mkdir()
            (base/'images'/'deep'/'secret.jpg').write_bytes(b'never read')
            (base/'link').symlink_to(base/'images', target_is_directory=True)
            result = scan_dataset(root, 'lithos', 100, max_depth=0, preview_bytes=16)
            self.assertEqual(len(result['directories']), 1)
            self.assertEqual(result['metadata_previews'][0]['bytes_read'], 16)
            self.assertTrue(result['metadata_previews'][0]['possibly_truncated'])
            self.assertFalse(any('secret' in row['relative_path'] for row in result['files']))
            self.assertTrue((base/'images'/'deep'/'secret.jpg').exists())
            self.assertEqual((base/'metadata.json').stat().st_size, 10000)

    def test_image_files_are_never_opened(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'lithos').mkdir()
            (root/'lithos'/'ppl-0.png').write_bytes(b'not a valid image')
            with patch.object(Path, 'open', side_effect=AssertionError('Pixel file opened')):
                result = scan_dataset(root, 'lithos', 10)
            self.assertEqual(len(result['files']), 1)

    def test_continuation_reuses_global_budget_and_render_never_scans(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root/'lithos'
            (base/'images').mkdir(parents=True)
            for i in range(12):
                (base/'images'/f'{i}.png').write_bytes(b'never decode')
            initial = scan_dataset(root, 'lithos', 10, max_depth=0)
            first, second, rendered = root/'first', root/'second', root/'rendered'
            write_artifacts(first, [initial], 5)
            with patch('scripts.audit_datasets_fast.DATASETS', {'lithos': 211604}):
                main(['--root', str(root), '--continue-from', str(first/'audit_evidence.json'),
                      '--max-entries', '5', '--output', str(second)])
                evidence = json.loads((second/'audit_evidence.json').read_text())
                self.assertEqual(evidence['inspected_entries'], 5)
                paths = [d['path'] for d in evidence['results'][0]['directories']]
                self.assertEqual(paths, ['.', 'images'])
                with patch('scripts.audit_datasets_fast.os.scandir', side_effect=AssertionError('Rescanned dataset')):
                    main(['--render-from', str(second/'audit_evidence.json'), '--output', str(rendered)])
                with (rendered/'sample_manifest.csv').open() as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 4)
                self.assertEqual(len({r['relative_path'] for r in rows}), 4)
                with self.assertRaises(FileExistsError):
                    main(['--render-from', str(second/'audit_evidence.json'), '--output', str(rendered)])

    def test_categories_and_candidate_ids(self):
        self.assertEqual(classify('images/a.tif')[0], 'microscopy_image')
        self.assertEqual(classify('masks/a.png')[0], 'segmentation_mask')
        self.assertEqual(classify('ground_truth/a.png')[0], 'segmentation_mask')
        self.assertEqual(classify('images/labels/a.png')[0], 'unknown')
        self.assertEqual(classify('tiles/a.jpg')[0], 'derived_image')
        self.assertEqual(classify('a.jpg')[0], 'unknown')
        self.assertEqual(classify('labels.csv')[0], 'label_or_annotation')
        row = identifiers('sample_12/PPL/rock001_tile_03_04.tif')
        self.assertEqual(row['sample_id_candidate'], '12')
        self.assertEqual(row['original_image_id_candidate'], 'rock001')
        self.assertEqual(row['tile_candidate'], '03_04')
        self.assertEqual(row['variants'], 'PPL')
        self.assertEqual(identifiers('samples/a.tif')['sample_id_candidate'], 'unknown')

    def test_observed_ambiguities_and_grouping_candidates(self):
        self.assertEqual(classify('raw/a.png')[0], 'unknown')
        self.assertEqual(classify('qa/mask_IMG_1.png', 'aachen_heerlen')[0], 'derived_image')
        self.assertEqual(classify('raw/sem_segmentation_ssl/train/Phenom_Images/0_0_BO.tiff', 'kit_sem_powder')[0], 'unknown')
        self.assertEqual(classify('__MACOSX/._a.tif')[0], 'metadata')
        self.assertEqual(classify('center_component.csv', 'lithos')[0], 'label_or_annotation')
        row = identifiers('raw/BM01_SEM_BSE/BM01_SEM_BSE/slice-In-Beam BSE-00001.png', 'jlu_cathodes')
        self.assertEqual(row['sample_id_candidate'], 'BM01')
        self.assertEqual(row['original_image_id_candidate'], 'BM01/slice-00001')
        tile = identifiers('raw/sem_segmentation_ssl/train/Run1/Tile_r000_c001.tiff', 'kit_sem_powder')
        self.assertEqual(tile['tile_candidate'], 'r000_c001')
        self.assertEqual(tile['original_image_id_candidate'], 'unknown')
        tile = identifiers('raw/sem_segmentation_ssl/train/Run2/Tile_r000_c001.tiff', 'kit_sem_powder')
        self.assertEqual(tile['original_image_id_candidate'], 'unknown')


if __name__ == '__main__':
    unittest.main()
