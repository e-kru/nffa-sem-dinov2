"""Optional loader tuning and lossless, shared decoded-image storage.

The cache contains only deterministic RGB + crop_metadata_strip pixels.
Every presentation still executes the original Meta augmentation and mask RNG.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import mmap
import os
from pathlib import Path
import shutil
import time
import uuid

import numpy as np
from PIL import Image

from src.preprocessing_v2 import crop_metadata_strip
from src.ssl.data import sha256


@dataclass
class LoaderOptions:
    persistent_workers: bool = False
    prefetch_factor: int = 2
    pin_memory: bool = True
    non_blocking: bool = True
    decoded_cache: str | None = None
    profile: bool = False


def options(cfg):
    values = dict(cfg.get('loader', {}))
    result = LoaderOptions(**values)
    if type(result.prefetch_factor) is not int or result.prefetch_factor < 1:
        raise ValueError('prefetch_factor must be a positive integer')
    for field in ('persistent_workers', 'pin_memory', 'non_blocking', 'profile'):
        if type(getattr(result, field)) is not bool:
            raise ValueError(f'{field} must be boolean')
    return result


def add_arguments(parser):
    import argparse
    for name in ('persistent-workers', 'pin-memory', 'non-blocking'):
        parser.add_argument('--' + name, action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--prefetch-factor', type=int)
    parser.add_argument('--decoded-cache', type=Path, help='Completed lossless cache on node-local disk or tmpfs')
    parser.add_argument('--profile-loader', action=argparse.BooleanOptionalAction, default=None,
                        help='Diagnostic stage timings; use a separate run for final uninstrumented throughput')


def apply_arguments(cfg, args):
    names = {'persistent_workers':'persistent_workers', 'pin_memory':'pin_memory',
             'non_blocking':'non_blocking', 'prefetch_factor':'prefetch_factor',
             'decoded_cache':'decoded_cache', 'profile_loader':'profile'}
    values = dict(cfg.get('loader', {}))
    for arg, key in names.items():
        value = getattr(args, arg, None)
        if value is not None:
            if arg == 'decoded_cache':
                values[key] = None if str(value).lower() == 'none' else str(Path(value).resolve())
            else:
                values[key] = value
    if values:
        cfg.loader = values
    return options(cfg)


def loader_kwargs(cfg, device):
    opt = options(cfg)
    workers = cfg.train.num_workers
    return dict(num_workers=workers, pin_memory=opt.pin_memory and str(device).startswith('cuda'),
                **(dict(multiprocessing_context='spawn', persistent_workers=opt.persistent_workers,
                        prefetch_factor=opt.prefetch_factor) if workers else {}))


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def cache_identity(identity):
    return canonical({k:identity[k] for k in ('provider', 'root', 'manifests', 'records')})


def preprocessing_sha():
    return sha256(Path(__file__).resolve().parents[1] / 'preprocessing_v2.py')


def load_image(path, timings=None):
    if timings is None:
        with Image.open(path) as source:
            return crop_metadata_strip(source.convert('RGB'))
    begun = time.perf_counter()
    encoded = path.read_bytes()
    timings['read_seconds'] = time.perf_counter() - begun
    begun = time.perf_counter()
    with Image.open(io.BytesIO(encoded)) as source:
        image = source.convert('RGB')  # Forces lazy decoding; RGB conversion unchanged.
    timings['decode_rgb_seconds'] = time.perf_counter() - begun
    begun = time.perf_counter()
    image = crop_metadata_strip(image)
    timings['metadata_seconds'] = time.perf_counter() - begun
    return image


class TimedTransform:
    def __init__(self, transform, measurements, name):
        self.transform, self.measurements, self.name = transform, measurements, name

    def __call__(self, image):
        begun = time.perf_counter()
        result = self.transform(image)
        self.measurements[self.name] = self.measurements.get(self.name, 0.) + time.perf_counter() - begun
        return result


class ProfiledAugmentation:
    """Time calls to original instance components, without copying their logic."""
    def __init__(self, original):
        self.original, self.measurements = original, {}
        for name in ('geometric_augmentation_global', 'geometric_augmentation_local'):
            setattr(original, name, TimedTransform(getattr(original, name), self.measurements,
                    'crop_global_seconds' if name.endswith('global') else 'crop_local_seconds'))
        for name in ('global_transfo1', 'global_transfo2', 'local_transfo'):
            transform = getattr(original, name)
            transform.transforms = [TimedTransform(child, self.measurements, key) for child, key in zip(
                transform.transforms, ('color_seconds', 'blur_solarize_seconds', 'normalize_seconds'))]

    def __call__(self, image):
        self.measurements.clear()
        begun = time.perf_counter()
        result = self.original(image)
        result['_loader_profile'] = {**self.measurements, 'augmentation_seconds':time.perf_counter()-begun}
        return result


class DecodedCache:
    """Spawn-safe read-only mmap. Workers share OS pages, not Python image dicts."""
    def __init__(self, directory, identity, *, verify_blob=False):
        self.directory = Path(directory).resolve()
        self.metadata = json.loads((self.directory / 'complete.json').read_text())
        meta = self.metadata
        if (meta['schema'] != 'sem-rgb-cache-v1' or meta['data_sha256'] != cache_identity(identity)
                or meta['preprocessing_sha256'] != preprocessing_sha()):
            raise ValueError('Decoded cache data/preprocessing identity mismatch')
        self.entries = {row['image_id']:row for row in meta['images']}
        if list(self.entries) != [r['image_id'] for r in identity['records']]:
            raise ValueError('Decoded cache coverage/order mismatch')
        offset = 0
        for row in meta['images']:
            height, width, channels = row['shape']
            if channels != 3 or min(height, width) < 1 or row['offset'] != offset:
                raise ValueError('Invalid decoded cache index')
            offset += height * width * channels
        blob = self.directory / 'pixels.bin'
        if offset != meta['bytes'] or blob.stat().st_size != offset:
            raise ValueError('Truncated decoded cache')
        if verify_blob and sha256(blob) != meta['pixels_sha256']:
            raise ValueError('Decoded cache checksum mismatch')
        self._mapping = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state['_mapping'] = None
        return state

    def image(self, image_id):
        if self._mapping is None:
            with (self.directory / 'pixels.bin').open('rb') as stream:
                self._mapping = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
        row = self.entries[image_id]
        pixels = np.ndarray(row['shape'], dtype=np.uint8, buffer=self._mapping, offset=row['offset'])
        return Image.fromarray(pixels)  # Transient PIL RGB copy; mapped corpus stays shared.


def configure_dataset(cfg, dataset, identity):
    opt = options(cfg)
    dataset.profile = opt.profile
    if opt.profile and not isinstance(dataset.transform, ProfiledAugmentation):
        dataset.transform = ProfiledAugmentation(dataset.transform)
    if opt.decoded_cache:
        dataset.decoded_cache = DecodedCache(opt.decoded_cache, identity)
    return dataset


def memory_info():
    info = {line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
            if line.startswith(('MemTotal:', 'MemAvailable:', 'Shmem:'))}
    for limit_path, used_path in (('/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory.current'),
                                   ('/sys/fs/cgroup/memory/memory.limit_in_bytes','/sys/fs/cgroup/memory/memory.usage_in_bytes')):
        limit_file, used_file = Path(limit_path), Path(used_path)
        if limit_file.is_file() and used_file.is_file():
            raw = limit_file.read_text().strip()
            if raw != 'max' and int(raw)>0:
                info['MemAvailable'] = min(info['MemAvailable'],max(0,int(raw)-int(used_file.read_text())))
    return info


def effective_cpus():
    available = len(os.sched_getaffinity(0))
    # Common cgroup v2 and v1 limits; affinity remains a hard upper bound.
    for path in (Path('/sys/fs/cgroup/cpu.max'),):
        if path.exists():
            quota, period = path.read_text().split()
            if quota != 'max':
                available = min(available, max(1, int(quota)//int(period)))
    quota_path = Path('/sys/fs/cgroup/cpu/cpu.cfs_quota_us')
    if quota_path.exists() and int(quota_path.read_text()) > 0:
        available = min(available, max(1, int(quota_path.read_text())//int(quota_path.with_name('cpu.cfs_period_us').read_text())))
    return available


def process_memory(pids):
    result = dict(rss_bytes=0, pss_bytes=0)
    for pid in pids:
        try:
            lines = Path(f'/proc/{pid}/smaps_rollup').read_text().splitlines()
            values = {line.split(':')[0]:int(line.split()[1])*1024 for line in lines if line.startswith(('Rss:', 'Pss:'))}
            result['rss_bytes'] += values.get('Rss', 0)
            result['pss_bytes'] += values.get('Pss', 0)
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    return result


def filesystem(path):
    path = Path(path).resolve()
    matches = []
    for line in Path('/proc/mounts').read_text().splitlines():
        fields = line.split()
        mount = Path(fields[1].replace('\\040', ' '))
        if path.is_relative_to(mount):
            matches.append((len(str(mount)), fields[2]))
    return max(matches)[1] if matches else 'unknown'


def estimate_cache(root, records, workers=8):
    def inspect(row):
        path=(Path(root)/row['image_id']).resolve()
        if not path.is_relative_to(Path(root).resolve()):
            raise ValueError('Image path outside the audited source root')
        with Image.open(path) as image:
            width, height = image.size
        return width * height * 3
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        sizes = list(pool.map(inspect, records))
    return dict(images=len(sizes), upper_bound_bytes=sum(sizes), max_image_bytes=max(sizes),
                mean_image_bytes=sum(sizes)/len(sizes))


def build_cache(directory, identity, *, max_bytes, workers=8):
    """Bounded concurrent decode; one packed file avoids thousands of cache opens."""
    from src.ssl.checkpoints import atomic_json
    directory = Path(directory).resolve()
    if directory.exists():
        raise FileExistsError(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    fs = filesystem(directory.parent)
    if fs.startswith(('nfs', 'cifs', 'smb')):
        raise ValueError('Use node-local disk or tmpfs for a decoded cache, not the shared filesystem')
    estimate = estimate_cache(identity['root'], identity['records'], workers)
    available = shutil.disk_usage(directory.parent).free
    if fs in ('tmpfs', 'ramfs'):
        available = min(available, int(memory_info()['MemAvailable'] * .6))
    if estimate['upper_bound_bytes'] > min(max_bytes, available):
        raise MemoryError(f"Decoded RGB upper bound {estimate['upper_bound_bytes']} bytes exceeds budget/free memory {min(max_bytes, available)}")
    temporary = directory.with_name('.pending_' + directory.name + '_' + uuid.uuid4().hex)
    temporary.mkdir()
    images, offset, checksum = [], 0, hashlib.sha256()
    begun = time.perf_counter()

    def decode(row):
        path = (Path(identity['root']) / row['image_id']).resolve()
        if not path.is_relative_to(Path(identity['root']).resolve()) or sha256(path) != row['sha256']:
            raise ValueError(f"Changed or unsafe source image: {row['image_id']}")
        return np.asarray(load_image(path), dtype=np.uint8)

    try:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool, (temporary / 'pixels.bin').open('wb') as stream:
            records = iter(identity['records'])
            # Unlike an eager map, at most workers decoded full images are queued.
            pending = deque((row, pool.submit(decode, row)) for row in [next(records, None) for _ in range(max(1, workers))] if row is not None)
            while pending:
                row, future = pending.popleft()
                pixels = future.result()
                payload = pixels.tobytes()
                if offset + len(payload) > max_bytes:
                    raise MemoryError('Decoded cache exceeds explicit byte budget')
                stream.write(payload)
                checksum.update(payload)
                images.append(dict(image_id=row['image_id'], offset=offset, shape=list(pixels.shape)))
                offset += len(payload)
                row = next(records, None)
                if row is not None:
                    pending.append((row, pool.submit(decode, row)))
            stream.flush()
            os.fsync(stream.fileno())
        metadata = dict(schema='sem-rgb-cache-v1', data_sha256=cache_identity(identity),
                        preprocessing_sha256=preprocessing_sha(), bytes=offset, images=images,
                        pixels_sha256=checksum.hexdigest(), filesystem=fs, build_seconds=time.perf_counter()-begun,
                        estimate=estimate, source_count=identity['count'])
        atomic_json(temporary / 'complete.json', metadata)
        os.rename(temporary, directory)
        return metadata
    except BaseException:
        shutil.rmtree(temporary)
        raise
