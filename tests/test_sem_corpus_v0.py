"""Load-time V0 geometry and real tiny Meta SSL integration; no GPU or downloads."""
import csv
import random

import numpy as np
from PIL import Image
import pytest
import torch

from test_sem_cellstyle import fixture
from src.sem_cellstyle.data import SEMViews
from src.sem_cellstyle.recipe import schedules
from src.sem_cellstyle.runner import initialize, loader
from src.sem_cellstyle.training import accumulated_step
from src.sem_corpus_v0.data import ResolutionAwareSEMViews, prepare_training_data


@pytest.mark.parametrize('side', [128, 160, 224, 256, 376, 512, 517])
def test_small_originals_never_produce_extreme_random_crops(side):
    for scale in ((.5, 1.), (.15, .5)):
        for _ in range(100):
            left, top, right, bottom = ResolutionAwareSEMViews.square_box((side+51, side), scale)
            assert right-left == bottom-top >= min(side, 224)
            assert 0 <= left < right <= side+51 and 0 <= top < bottom <= side


def test_large_originals_keep_existing_crop_distribution_exactly():
    for scale in ((.5, 1.), (.15, .5)):
        random.seed(42)
        expected = [SEMViews.square_box((1024, 768), scale) for _ in range(100)]
        random.seed(42)
        actual = [ResolutionAwareSEMViews.square_box((1024, 768), scale) for _ in range(100)]
        assert actual == expected


def manifest(tmp_path):
    records = []
    for i, side in enumerate((128, 512)):
        path = tmp_path/f'color{i}.png'
        # Color remains permitted until the later cleaning stage.
        Image.fromarray(np.random.default_rng(i).integers(0, 230, (side, side, 3), dtype=np.uint8)).save(path)
        records.append(dict(image_id=str(i), path=str(path), source='fixture', original_id=str(i),
                            specimen_id='unknown', split='train', original_split='train', included='1',
                            header_status='ok', width=str(side), height=str(side), role='microscopy_image'))
    path = tmp_path/'manifest.csv'
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, list(records[0])); writer.writeheader(); writer.writerows(records)
    return path, records


@pytest.mark.parametrize('resolution,local', [(224, 98), (518, 224)])
def test_configurable_resolution_complete_meta_step_and_unchanged_originals(fixture, tmp_path, resolution, local):
    cfg, _, _, _, initialization = fixture
    path, records = manifest(tmp_path)
    before = {r['path']: open(r['path'], 'rb').read() for r in records}
    configured, dataset, collate, identity = prepare_training_data(cfg, path, training_resolution=resolution)
    assert cfg.crops.global_crops_size == 56  # caller's existing recipe is untouched
    assert configured.crops.global_crops_size == resolution and configured.crops.local_crops_size == local
    views, _ = dataset[0]
    assert len(views['global_crops']) == 2 and views['global_crops'][0].shape == (3, resolution, resolution)
    assert views['local_crops'][0].shape == (3, local, local)
    assert identity['original_minimum_side'] == 128 and not identity['original_files_modified']
    model, optimizer = initialize(configured, initialization, 'cpu')
    row = accumulated_step(model, optimizer, iter(loader(configured, dataset, collate, 'cpu', 0, 1)),
                           schedules(configured), 0, precision='fp32')
    assert np.isfinite(row['total_loss']) and np.isfinite(row['gradient_norm_backbone'])
    assert all(open(path, 'rb').read() == content for path, content in before.items())


def test_default_is_224_and_inclusion_lithos_and_holdout_guards(fixture, tmp_path):
    cfg, _, _, _, _ = fixture
    path, records = manifest(tmp_path)
    configured, _, _, _ = prepare_training_data(cfg, path)
    assert configured.crops.global_crops_size == 224
    for field, value, error in [('source', 'lithos', 'LITHOS'), ('split', 'val', 'train rows')]:
        changed = [dict(r) for r in records]; changed[0][field] = value
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, list(changed[0])); writer.writeheader(); writer.writerows(changed)
        with pytest.raises(ValueError, match=error):
            prepare_training_data(cfg, path)
