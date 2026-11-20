"""Opt-in SEM crop perturbations with draw-scoped, independent random streams."""

from functools import partial
import random

import numpy as np
from PIL import Image, ImageEnhance
import torch
from torchvision.transforms import functional as TF

from src.sem_cellstyle.data import CorpusDataset, SEMViews
from src.ssl.stream import draw_seed


MEAN = (.485, .456, .406)
STD = (.229, .224, .225)
NOISE = dict(gaussian_probability=.20, gaussian_std_range=[.005, .02],
             poisson_probability=.20, poisson_scale_range=[200, 1000])
GEOMETRY = dict(rotations_degrees=[0, 90, 180, 270])


def noise(gray, generator):
    gaussian, poisson = generator.random(2) < [NOISE['gaussian_probability'], NOISE['poisson_probability']]
    if not (gaussian or poisson):
        return gray, ()
    pixels = gray.numpy().copy()
    operations = []
    if gaussian:
        std = generator.uniform(*NOISE['gaussian_std_range'])
        pixels += generator.normal(0., std, size=pixels.shape).astype(np.float32)
        operations.append('Gaussian')
    if poisson:
        scale = generator.uniform(*NOISE['poisson_scale_range'])
        # Poisson rates must derive from valid [0, 1] intensities even when
        # Gaussian noise also fired on the same view.
        pixels = (generator.poisson(np.clip(pixels, 0., 1.) * scale) / scale).astype(np.float32)
        operations.append('Poisson')
    return torch.from_numpy(pixels).clamp_(0., 1.), tuple(operations)


def orientation(gray, generator):
    degrees = int(generator.choice(GEOMETRY['rotations_degrees']))
    return gray.rot90(degrees // 90, (-2, -1)), degrees


class FactorialViews(SEMViews):
    """Use SEMViews' crop routing and geometry, with independent extras before normalization."""

    def __init__(self, cfg, add_noise):
        super().__init__(cfg)
        self.add_noise = add_noise

    def gray_view(self, image, size, scale):
        # Match the original SEMViews operations and the Python RNG draw order.
        # Keeping this boundary local preserves old checkpoint/source fingerprints.
        image = image.crop(self.square_box(image.size, scale)).resize((size, size), Image.Resampling.BICUBIC)
        if random.random() < .8:
            operations = [(ImageEnhance.Brightness, random.uniform(.9, 1.1)),
                          (ImageEnhance.Contrast, random.uniform(.9, 1.1))]
            random.shuffle(operations)
            for operation, factor in operations:
                image = operation(image).enhance(factor)
        return TF.to_tensor(image)

    def augment_gray(self, gray, seed, crop, allow_geometry):
        changes = ()
        transform = 0
        if self.add_noise:
            gray, changes = noise(gray, np.random.default_rng(draw_seed(seed, 'sem-augmentation-noise-v1', crop)))
        if allow_geometry:
            gray, transform = orientation(gray, np.random.default_rng(draw_seed(seed, 'sem-augmentation-rotation-v1', crop)))
        return gray, changes, transform

    def __call__(self, image, orientation=False):
        if not (self.add_noise or orientation):
            return super().__call__(image)
        # DrawDataset sets this seed per consumed image, independent of worker,
        # resume cursor, collation RNG and model RNG.
        generate = partial(self.augment_gray, seed=torch.initial_seed(), allow_geometry=orientation)
        gray = image.convert('L')
        crops = [(self.global_size, self.global_scale)] * 2 + [(self.local_size, self.local_scale)] * self.locals
        views = []
        for index, (size, scale) in enumerate(crops):
            plane, _, _ = generate(self.gray_view(gray, size, scale), crop=index)
            views.append(TF.normalize(plane.repeat(3, 1, 1), MEAN, STD))
        return dict(global_crops=views[:2], global_crops_teacher=views[:2], local_crops=views[2:], offsets=())


def training_dataset(cfg, baseline, *, add_noise, geometry_ids):
    if not add_noise and not geometry_ids:
        return baseline
    records = [
        dict(row, path=str(baseline.root / row['image_id']), source=row['image_id'] in geometry_ids)
        for row in baseline.records
    ]
    sources = {allowed: dict(orientation_invariant=allowed) for allowed in (False, True)}
    return CorpusDataset(records, sources, FactorialViews(cfg, add_noise))
