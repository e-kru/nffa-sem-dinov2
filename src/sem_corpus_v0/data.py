"""Resolution-aware, load-time views using the existing SEM photometric pipeline."""
import csv
from pathlib import Path

from omegaconf import OmegaConf

from src.sem_cellstyle.data import CorpusDataset, SEMViews, collate_for
from src.ssl.data import sha256

SUPPORTED_RESOLUTIONS = (224, 518)


class ResolutionAwareSEMViews(SEMViews):
    """Keep >=518-source distributions; protect smaller source content from zoom."""
    @staticmethod
    def square_box(size, scale):
        shortest = min(size)
        if shortest >= 518:
            return SEMViews.square_box(size, scale)
        # Below 224 source pixels retain the entire short side. For 224..517
        # retain at least 224 source pixels in every global/local crop. Applied
        # after the existing metadata-strip removal, before load-time resizing.
        floor = (min(shortest, 224)/shortest)**2
        lower = max(float(scale[0]), floor)
        upper = max(float(scale[1]), lower)
        return SEMViews.square_box(size, (lower, upper))


def prepare_training_data(cfg, manifest, *, training_resolution=224):
    """Return opt-in config/dataset/Meta collate/identity; no model or training.

    This is a candidate corpus, not a verified mixed-corpus curation receipt.
    included=0 is always honored; LITHOS is forbidden even after manual edits.
    """
    if training_resolution not in SUPPORTED_RESOLUTIONS:
        raise ValueError('V0 training resolution must be 224 or 518')
    manifest = Path(manifest).absolute()
    records, ids, paths = [], set(), set()
    with manifest.open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row.get('included') not in ('0', '1'):
                raise ValueError('included must be 0 or 1')
            if row['included'] == '0':
                continue
            if row['source'] == 'lithos':
                raise ValueError('LITHOS is excluded from V0')
            if row.get('role') not in ('microscopy_image', 'derived_image'):
                raise ValueError('V0 requires a microscopy image role, never a mask or unknown target')
            if row.get('split') != 'train' or not Path(row['path']).is_absolute():
                raise ValueError('V0 requires train rows and absolute paths')
            if row.get('header_status') != 'ok' or min(int(row['width']), int(row['height'])) < 128:
                raise ValueError('V0 requires readable original headers and minimum 128 per side')
            if row['source'] in ('nffa_sem_100', 'aachen_heerlen') and row.get('original_split') != 'train':
                raise ValueError('NFFA/Aachen held-out rows cannot enter V0')
            if row['image_id'] in ids or row['path'] in paths:
                raise ValueError('Repeated image ID/path in V0')
            ids.add(row['image_id']); paths.add(row['path']); records.append(row)
    if len(records) < 2:
        raise ValueError('V0 needs at least two included images')
    configured = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    configured.crops.global_crops_size = training_resolution
    configured.crops.local_crops_size = 98 if training_resolution == 224 else 224
    sources = {row['source']: dict(orientation_invariant=False) for row in records}
    transform = ResolutionAwareSEMViews(configured)
    identity = dict(provider='candidate_sem_corpus_v0', count=len(records),
                    manifest=str(manifest), manifest_sha256=sha256(manifest),
                    training_resolution=training_resolution, local_resolution=configured.crops.local_crops_size,
                    original_minimum_side=128, resolution_basis='original_header_before_metadata_strip',
                    load_time_crop_policy='source_min_side_ge518: existing_scales; otherwise crop_side>=min(224,post_strip_short_side)',
                    original_files_modified=False, color_filter_applied=False, cleaning_status='pending',
                    source_sha256=sha256(Path(__file__)), sources_orientation=sources)
    return configured, CorpusDataset(records, sources, transform), collate_for(configured), identity
