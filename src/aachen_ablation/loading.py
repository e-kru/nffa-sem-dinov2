"""Persistent original-image loaders with deterministic, epoch-addressable order."""
import torch
from torch.utils.data import DataLoader, Sampler


def pack_originals(samples):
    targets = [s["target"] for s in samples]
    if len({tuple(y.shape) for y in targets}) == 1:
        targets = torch.stack(targets)
    return {"features": torch.stack([s["features"] for s in samples]),
            "targets": targets,
            "records": [s["record"] for s in samples]}


def slice_batch(batch, start, stop):
    return {"features": batch["features"][start:stop],
            "targets": batch["targets"][start:stop], "records": batch["records"][start:stop]}


class EpochSampler(Sampler):
    def __init__(self, dataset, seed):
        self.count, self.seed, self.epoch = len(dataset), seed, 1

    def __len__(self):
        return self.count

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        # Match the previous loader's base_seed draw before its RandomSampler.
        torch.empty((), dtype=torch.int64).random_(generator=generator)
        return iter(torch.randperm(self.count, generator=generator).tolist())


def initialize_worker(_):
    torch.set_num_threads(1)


def make_loader(dataset, batch_size, workers, device, shuffle=False, seed=42):
    return DataLoader(dataset, batch_size=batch_size,
                      sampler=EpochSampler(dataset, seed) if shuffle else None,
                      generator=torch.Generator().manual_seed(seed),
                      collate_fn=pack_originals, num_workers=workers,
                      pin_memory=device.type == "cuda", persistent_workers=workers > 0,
                      worker_init_fn=initialize_worker,
                      **({"multiprocessing_context": "spawn", "timeout": 120} if workers else {}))
