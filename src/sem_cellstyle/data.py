"""A deliberately small manifest boundary; curation/audit remains external."""
import csv
from functools import partial
import json
import math
from pathlib import Path
import random

from PIL import Image, ImageEnhance, ImageOps
import torch
from torch.utils.data import Dataset, Sampler
from torchvision.transforms import functional as TF

from src.ssl.data import sha256
from src.ssl.loading import load_image
from src.ssl.stream import draw_seed
from src.ssl.upstream import activate


class SEMViews:
    """Square area sampling relative to min(W,H)^2, with source-gated orientation."""
    def __init__(self, cfg):
        self.global_size, self.local_size = cfg.crops.global_crops_size, cfg.crops.local_crops_size
        self.global_scale, self.local_scale = cfg.crops.global_crops_scale, cfg.crops.local_crops_scale
        self.locals = cfg.crops.local_crops_number

    @staticmethod
    def square_box(size, scale):
        width, height = size
        side = max(1, min(min(size), round(min(size) * math.sqrt(random.uniform(*scale)))))
        left, top = random.randint(0, width-side), random.randint(0, height-side)
        return left, top, left+side, top+side

    def view(self, image, size, scale, orientation):
        image = image.crop(self.square_box(image.size, scale)).resize((size, size), Image.Resampling.BICUBIC)
        if orientation:
            if random.random() < .5:
                image = ImageOps.mirror(image)
            if random.random() < .5:
                image = ImageOps.flip(image)
            image = image.rotate(random.choice((0, 90, 180, 270)))
        if random.random() < .8:
            # Random order, both on the one grayscale channel, before RGB replication.
            operations = [(ImageEnhance.Brightness, random.uniform(.9, 1.1)),
                          (ImageEnhance.Contrast, random.uniform(.9, 1.1))]
            random.shuffle(operations)
            for operation, factor in operations:
                image = operation(image).enhance(factor)
        tensor = TF.to_tensor(image).repeat(3, 1, 1)
        return TF.normalize(tensor, [.485, .456, .406], [.229, .224, .225])

    def __call__(self, image, orientation=False):
        image = image.convert('L')
        globals_ = [self.view(image, self.global_size, self.global_scale, orientation) for _ in range(2)]
        return dict(global_crops=globals_, global_crops_teacher=globals_,
                    local_crops=[self.view(image, self.local_size, self.local_scale, orientation) for _ in range(self.locals)],
                    offsets=())


class CorpusDataset(Dataset):
    def __init__(self, records, sources, transform):
        self.records, self.sources, self.transform = records, sources, transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        # Existing deterministic metadata-strip removal before any stochastic crop.
        image = load_image(Path(record['path']))
        allowed = self.sources[record['source']].get('orientation_invariant', False) is True
        return self.transform(image, allowed), ()


def collate_for(cfg):
    activate()
    from dinov2.data import MaskingGenerator, collate_data_and_cast
    grid = cfg.crops.global_crops_size // cfg.student.patch_size
    return partial(collate_data_and_cast, mask_ratio_tuple=tuple(cfg.ibot.mask_ratio_min_max),
                   mask_probability=cfg.ibot.mask_sample_probability, n_tokens=grid**2,
                   mask_generator=MaskingGenerator(input_size=(grid, grid), max_num_patches=int(grid**2 * .5)),
                   dtype=torch.float32)


def provider(cfg):
    if cfg.dataset.provider == 'verified_nffa_inventory_v1':
        return nffa_provider(cfg)
    if cfg.dataset.provider != 'verified_sem_manifest_v1':
        raise ValueError('Expected an explicitly curated SEM manifest')
    if not cfg.dataset.manifest or not cfg.dataset.receipt:
        raise ValueError('Dataset not ready: provide dataset.manifest and dataset.receipt; no inventory is selected automatically')
    manifest, receipt_path = Path(cfg.dataset.manifest).resolve(), Path(cfg.dataset.receipt).resolve()
    receipt = json.loads(receipt_path.read_text())
    if receipt.get('schema') != 'verified_sem_manifest_v1' or receipt.get('status') != 'verified':
        raise ValueError('Dataset receipt is a placeholder or has not been verified')
    if not all(receipt.get(key) for key in ('corpus_version', 'content_snapshot', 'curation_record', 'leakage_review_record')):
        raise ValueError('Dataset receipt needs curation and leakage-review provenance')
    exclusions = ('nffa_A_B_val_test_and_groups', 'all_aachen', 'nasa_benchmarks_if_evaluated')
    if not all(receipt.get('exclusions', {}).get(key) is True for key in exclusions):
        raise ValueError('Explicit held-out benchmark/group exclusions are required')
    if sha256(manifest) != receipt.get('manifest_sha256'):
        raise ValueError('Manifest differs from the curated receipt')
    with manifest.open(newline='') as stream:
        records = list(csv.DictReader(stream))
    n = receipt.get('verified_train_originals')
    if type(n) is not int or n != len(records) or n < 2:
        raise ValueError('verified_train_originals does not match the manifest')
    ids, paths = set(), set()
    sources = receipt.get('sources', {})
    for record in records:
        if any(not record.get(key) for key in ('image_id', 'path', 'source', 'original_id', 'specimen_id')):
            raise ValueError('Manifest needs image/source/original/specimen identities and absolute paths')
        path = Path(record['path'])
        if record.get('split') != 'train' or not path.is_absolute():
            raise ValueError('Only explicit train rows with absolute paths are allowed')
        if record['image_id'] in ids or str(path) in paths:
            raise ValueError('Repeated image ID/path in training manifest')
        ids.add(record['image_id']); paths.add(str(path))
        if record['source'] not in sources:
            raise ValueError('Each source needs an explicit policy (orientation defaults off)')
        if type(sources[record['source']].get('orientation_invariant', False)) is not bool:
            raise ValueError('Orientation permission must be a boolean')
    identity = dict(provider='verified_sem_manifest_v1', count=n, manifest=str(manifest),
                    manifest_sha256=receipt['manifest_sha256'], receipt=receipt,
                    receipt_sha256=sha256(receipt_path), image_contents_rehashed=False)
    # The receipt attests curation/holdout checks; this adapter does not repeat them.
    return CorpusDataset(records, sources, SEMViews(cfg)), collate_for(cfg), identity


def nffa_provider(cfg):
    """Reuse the audited NFFA training selection with the Large recipe's views.

    This is a separate pilot boundary, not a receipt for the future mixed corpus.
    Only inventory/manifests are fingerprinted; original image bytes are not
    rehashed and held-out images are never opened.
    """
    from omegaconf import OmegaConf
    from src.ssl_benchmark import dataset_from_inventory

    if not cfg.dataset.get('inventory'):
        raise ValueError('NFFA pilot requires an existing verified SSL inventory')
    path = Path(cfg.dataset.inventory).expanduser().resolve()
    saved = json.loads(path.read_text())
    inventory = saved.get('dataset', saved)  # Original run.json or standalone inventory.
    if not cfg.dataset.get('primary_split'):
        raise ValueError('NFFA pilot requires an explicit primary split')
    protocol = cfg.dataset.get('selection_protocol', 'A_B_train_intersection_v1')
    if protocol == 'split_A_train_v1':
        if cfg.dataset.get('holdout_splits'):
            raise ValueError('Split A protocol does not intersect additional splits')
        if list(cfg.evaluation.get('classification_splits', ['A'])) != ['A']:
            raise ValueError('Split A SSL training requires Split A downstream validation; B holdouts overlap training')
    elif protocol != 'A_B_train_intersection_v1' or not cfg.dataset.get('holdout_splits'):
        raise ValueError('Historical intersection protocol requires additional A/B holdout protection')
    legacy = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    legacy.dataset.provider = 'nffa_v3'
    activate()
    dataset, _ = dataset_from_inventory(legacy, inventory)
    # Existing dataset removes the metadata strip before applying this transform.
    # Orientation is unapproved for NFFA: SEMViews defaults to False.
    dataset.transform = SEMViews(cfg)
    identity = dict(provider='verified_nffa_inventory_v1', count=len(dataset),
                    inventory_path=str(path), inventory_sha256=sha256(path),
                    audited_inventory=inventory, image_contents_rehashed=False,
                    orientation_invariant=False, selection_protocol=protocol,
                    scope='NFFA train-only pilot; not the mixed SEM corpus')
    return dataset, collate_for(cfg), identity


class PresentationBatches(Sampler):
    """Seekable uniform permutations, including tails; no originals dropped at epochs."""
    def __init__(self, size, batch, seed, start_microbatch, stop_microbatch):
        if size < 2 or batch < 2 or not 0 <= start_microbatch <= stop_microbatch:
            raise ValueError('Invalid presentation cursor')
        self.size, self.batch, self.seed = size, batch, seed
        self.start, self.stop = start_microbatch, stop_microbatch

    def __len__(self):
        return self.stop-self.start

    def __iter__(self):
        previous, order = None, None
        for micro in range(self.start, self.stop):
            rows = []
            for draw in range(micro*self.batch, (micro+1)*self.batch):
                epoch, offset = divmod(draw, self.size)
                if previous != epoch:
                    order = torch.randperm(self.size, generator=torch.Generator().manual_seed(
                        draw_seed(self.seed, 'corpus-permutation-v1', epoch))).tolist()
                    previous = epoch
                rows.append((order[offset], draw))
            yield rows
