"""Seekable image-presentation stream around Meta's unchanged augment/collate.

Seeds depend on consumed draw IDs, not worker scheduling or prefetched queues.
Model RNG (including DropPath) is independent of this scoped CPU data RNG.
"""
from contextlib import contextmanager
import hashlib
import random
import time

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


def draw_seed(seed, domain, index):
    return int.from_bytes(hashlib.sha256(f"{seed}:{domain}:{index}".encode()).digest()[:8], "little") % (2**63)


@contextmanager
def data_rng(seed):
    python, numpy, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    try:
        random.seed(seed)
        np.random.seed(seed % 2**32)
        torch.set_rng_state(torch.Generator().manual_seed(seed).get_state())
        yield
    finally:
        random.setstate(python)
        np.random.set_state(numpy)
        torch.set_rng_state(cpu)


class DrawDataset(Dataset):
    def __init__(self, dataset, seed):
        self.dataset, self.seed = dataset, seed

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        index, draw = key
        with data_rng(draw_seed(self.seed, "image", draw)):
            sample = self.dataset[index]
        return sample, index, draw


class DrawCollate:
    def __init__(self, collate, seed):
        self.collate, self.seed = collate, seed

    def __call__(self, samples):
        draws = [item[2] for item in samples]
        begun = time.perf_counter()
        with data_rng(draw_seed(self.seed, "mask", draws[0])):
            batch = self.collate([item[0] for item in samples])
        profiles = [item[0][0].get('_loader_profile') for item in samples]
        if all(profile is not None for profile in profiles):
            batch['loader_profile'] = {key:sum(profile.get(key, 0.) for profile in profiles)
                                      for key in set().union(*profiles)}
            batch['loader_profile'].update(images=len(samples), collate_seconds=time.perf_counter()-begun)
        batch["draw_ids"] = draws
        batch["record_indices"] = [item[1] for item in samples]
        return batch


class CursorBatches(Sampler):
    """Epoch permutations with drop_last; start/stop count optimizer updates."""
    def __init__(self, size, batch_size, seed, start, stop):
        self.size, self.batch_size, self.seed = size, batch_size, seed
        self.start, self.stop = start, stop
        self.per_epoch = size // batch_size
        if not self.per_epoch or not 0 <= start <= stop:
            raise ValueError("Invalid dataset size or stream cursor")

    def __len__(self):
        return self.stop - self.start

    def __iter__(self):
        previous, order = None, None
        for step in range(self.start, self.stop):
            epoch, offset = divmod(step, self.per_epoch)
            if epoch != previous:
                generator = torch.Generator().manual_seed(draw_seed(self.seed, "shuffle", epoch))
                order = torch.randperm(self.size, generator=generator).tolist()
                previous = epoch
            indices = order[offset * self.batch_size:(offset + 1) * self.batch_size]
            yield [(index, step * self.batch_size + pos) for pos, index in enumerate(indices)]
