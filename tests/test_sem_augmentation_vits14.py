import json
import random

import numpy as np
from PIL import Image
import pytest
import torch
from torchvision.transforms import functional as TF

from scripts.run_sem_augmentation_vits14 import (
    POLICY, PREVIOUS_RUNNER_SHA256, frozen_plan, geometry_policy, parse_args, scientific_identity,
)
from src.sem_augmentation import FactorialViews, MEAN, STD, noise, orientation, training_dataset
from src.sem_cellstyle.data import SEMViews
from src.sem_cellstyle.runner import execute, loader
from src.ssl.data import SEMTrainDataset
from src.ssl.stream import data_rng, draw_seed

from test_sem_cellstyle import fixture


def test_baseline_pixels_and_rng_match_at_all_crop_indices(fixture):
    cfg, _, _, _, _ = fixture
    image = Image.fromarray(np.random.default_rng(13).integers(0, 256, (85, 91), dtype=np.uint8))
    original = SEMViews(cfg)
    study = FactorialViews(cfg, True)
    for seed in range(10):
        with data_rng(seed):
            views = original(image)
            baseline_random_state = random.getstate()
        with data_rng(seed):
            crops = [(study.global_size, study.global_scale)] * 2 + [(study.local_size, study.local_scale)] * study.locals
            pairs = [study.gray_view(image, size, scale) for size,scale in crops]
            assert random.getstate() == baseline_random_state
        for plane, baseline in zip(pairs, views['global_crops'] + views['local_crops']):
            assert torch.equal(TF.normalize(plane.repeat(3,1,1), MEAN, STD), baseline)


def test_gaussian_and_poisson_operate_on_one_intensity_plane():
    class ForcedNoise:
        def random(self, count):
            assert count == 2
            return np.zeros(2)

        def uniform(self, low, high):
            assert (low, high) in ((.005,.02), (200,1000))
            return low

        def normal(self, mean, std, size):
            assert mean == 0 and std == .005
            return np.array([[[-.02, .02, .02]]], dtype=np.float32).reshape(size)

        def poisson(self, rates):
            np.testing.assert_allclose(rates, [[[0.,104.,200.]]], atol=1e-5)
            # Check both final clamping and conversion from count scale.
            return np.array([[[0,100,205]]])

    image = torch.tensor([[[0., .5, 1.]]])
    augmented, operations = noise(image, ForcedNoise())
    assert operations == ('Gaussian','Poisson')
    assert torch.equal(augmented, image)
    assert augmented.shape[0] == 1 and augmented.dtype == torch.float32


def test_exact_orientation_permutation():
    class ForcedRotation:
        def __init__(self, degrees):
            self.degrees = degrees
        def choice(self, options):
            assert list(options) == [0,90,180,270]
            return self.degrees

    gray = torch.arange(16).reshape(1,4,4)
    for degrees in (0, 90, 180, 270):
        transformed, applied = orientation(gray, ForcedRotation(degrees))
        assert applied == degrees
        assert torch.equal(transformed, gray.rot90(degrees//90, (-2,-1)))


def test_factorial_extras_share_crops_noise_masks_and_record_order(fixture, tmp_path):
    cfg, data, collate, _, _ = fixture
    baseline = SEMTrainDataset(tmp_path, [dict(row, image_id=row['path']) for row in data.records], SEMViews(cfg))
    # The first microbatch includes a gated row and a row with geometry forbidden.
    a = next(iter(loader(cfg, baseline, collate, 'cpu', 0, 1)))
    allowed = {baseline.records[a['record_indices'][0]]['image_id']}
    b, c, d = (
        next(iter(loader(cfg, training_dataset(cfg, baseline, add_noise=n, geometry_ids=ids),
                         collate, 'cpu', 0, 1)))
        for n,ids in ((True,set()), (False,allowed), (True,allowed))
    )
    for batch in (b,c,d):
        assert batch['draw_ids'] == a['draw_ids']
        assert batch['record_indices'] == a['record_indices']
        assert torch.equal(batch['collated_masks'], a['collated_masks'])
    size = len(a['draw_ids'])
    for crop in range(2+cfg.crops.local_crops_number):
        key = 'collated_global_crops' if crop < 2 else 'collated_local_crops'
        index = (crop if crop < 2 else crop-2) * size
        draw = a['draw_ids'][0]
        seed = draw_seed(draw_seed(cfg.runtime.seed, 'image', draw), 'sem-augmentation-rotation-v1', crop)
        _, deg = orientation(torch.zeros(1,2,2), np.random.default_rng(seed))
        for original, augmented in ((a,c),(b,d)):
            expected = original[key][index]
            assert torch.equal(augmented[key][index], expected.rot90(deg//90, (-2,-1)))
            assert torch.equal(augmented[key][index+1], original[key][index+1])


def test_augmented_checkpoint_resume_matches_uninterrupted(fixture, tmp_path):
    cfg, data, collate, identity, init = fixture
    baseline = SEMTrainDataset(tmp_path, [dict(row,image_id=row['path']) for row in data.records], SEMViews(cfg))
    augmented = training_dataset(cfg, baseline, add_noise=True, geometry_ids={baseline.records[0]['image_id']})
    full, _ = execute(cfg, augmented, collate, identity, init, tmp_path/'full', device='cpu')
    execute(cfg, augmented, collate, identity, init, tmp_path/'resume', device='cpu', interrupt_after=2)
    resumed, _ = execute(cfg, augmented, collate, identity, init, tmp_path/'resume', device='cpu', resume=True)
    assert all(torch.equal(value,resumed.state_dict()[key]) for key,value in full.state_dict().items())


def test_policy_refuses_heldout_ids_and_contract_refuses_regularization(tmp_path):
    target = tmp_path/'policy.json'
    policy = dict(schema='nffa_split_a_geometry_allowlist_v1', source='nffa_sem_100',
                  evidence=['source'], rationale='reviewed', allow=[])
    target.write_text(json.dumps(policy))
    assert not geometry_policy(target, {'nffa/train'})[1]
    policy['allow'] = [dict(source_id='test', evidence=['source'], rationale='allowed', image_ids=['nffa/test'])]
    target.write_text(json.dumps(policy))
    with pytest.raises(ValueError, match='approved Split A train'):
        geometry_policy(target, {'nffa/train'})
    cfg = dict(configuration=dict(runtime=dict(output='old'), regularization=dict(l2_sp=True)))
    with pytest.raises(ValueError, match='unregularized SSL'):
        scientific_identity(cfg)


def test_rotation_trial_scope_is_exactly_the_verified_train_ids(tmp_path):
    ids = {'nffa/train_1', 'nffa/train_2'}
    policy, approved = geometry_policy(POLICY, ids)
    assert approved == ids
    policy['scope'] = 'all_images'
    candidate = tmp_path/'rotation.json'
    candidate.write_text(json.dumps(policy))
    with pytest.raises(ValueError, match='verified Split A training inventory'):
        geometry_policy(candidate, ids)


def test_baseline_default_and_explicit_factorial_entrypoint():
    assert parse_args([]).stage == 'baseline'
    assert parse_args(['--stage', 'run']).stage == 'run'


def test_existing_factorial_plan_accepts_runner_default_change_only():
    previous = dict(mode='run', runner_sha256=PREVIOUS_RUNNER_SHA256,
                    arms=['Aug-A', 'Aug-B', 'Aug-C', 'Aug-D'], augmentation_sha256='augmentation')
    current = previous | {'runner_sha256': 'new runner'}
    assert frozen_plan(previous, current) == previous
    with pytest.raises(ValueError, match='different frozen study plan'):
        frozen_plan(previous, current | {'augmentation_sha256': 'changed'})
