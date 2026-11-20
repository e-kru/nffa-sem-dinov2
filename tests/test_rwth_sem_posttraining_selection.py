from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from scripts import select_rwth_sem_posttraining_v0 as rwth


def test_selection_balances_frames_and_filters_hr_paths():
    chosen = rwth.selected_ids()
    steel = chosen['16MnCrS5']
    assert Counter(tile // 900 for tile in steel) == {frame: 100 for frame in range(160)}
    assert {divmod(tile, 30) for tile in steel if tile < 900} == {
        (y, x) for y in range(1, 30, 3) for x in range(1, 30, 3)
    }
    assert len(chosen['DP/10mu']) == 10000
    assert min(chosen['DP/10mu']) >= 0 and max(chosen['DP/10mu']) < 25281
    assert chosen == rwth.selected_ids() and not chosen['DP/100mu']
    assert rwth.hr_key(rwth.HR_ROOT / 'DP/10mu/hr/27.png') == ('DP/10mu', 27)
    for path in ('DP/10mu/lr/27.png', 'DP/10mu/ref/27.png', 'DP/10mu/hr/027.png'):
        with pytest.raises(ValueError, match='Unexpected RWTH manifest path'):
            rwth.hr_key(rwth.HR_ROOT / path)
    with pytest.raises(ValueError):
        rwth.hr_key(Path('/unrelated/DP/10mu/hr/27.png'))


def test_overlap_finds_drift_but_does_not_accept_unrelated_tiles():
    generator = np.random.default_rng(42)
    frame = generator.integers(0, 256, (224, 228), dtype=np.uint8)
    first = frame[1:129, :128]
    neighbor = frame[:128, 97:225]
    assert rwth.overlap_shifts(first, neighbor, (0, 96)) == [[-1, 97]]
    assert rwth.overlap_shifts(first, generator.integers(0, 256, (128, 128), dtype=np.uint8), (0, 96)) == []
    uniform = np.zeros((128, 128), dtype=np.uint8)
    assert len(rwth.overlap_shifts(uniform, uniform, (0, 96))) > 1
