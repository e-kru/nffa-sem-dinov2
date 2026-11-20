"""Transactional training snapshots; immutable milestones and bounded recovery."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import random
import shutil
import uuid

import numpy as np
import torch

from src.ssl.data import sha256
from src.ssl.experiment import digest


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
    with temp.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


@contextmanager
def run_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.writer.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another process owns this run') from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:
        if not torch.cuda.is_available():
            raise ValueError('CUDA RNG requires the original CUDA runtime')
        torch.cuda.set_rng_state_all([item.cpu() for item in state['cuda']])


def validate_snapshot(path):
    metadata = json.loads((path / 'complete.json').read_text())
    if metadata['step'] != int(path.name.split('_')[-1]):
        raise ValueError('Snapshot step/path mismatch')
    for name in ('training.pth', 'teacher_backbone.pth'):
        if sha256(path / name) != metadata['files'][name]:
            raise ValueError(f'Corrupt snapshot: {path / name}')
    return metadata


def find_latest(root, expected_contract=None):
    candidates = list((root / 'checkpoints').glob('step_*')) + list((root / 'recovery').glob('step_*'))
    rejected = []
    for path in sorted(candidates, key=lambda p: p.name, reverse=True):
        try:
            meta = validate_snapshot(path)
        except (OSError, ValueError, KeyError, TypeError) as error:
            rejected.append(f'{path}: {error}')
            continue
        # A valid incompatible snapshot is an error, never a reason to rewind.
        if expected_contract is not None and meta['contract_sha256'] != digest(expected_contract):
            raise ValueError('Resume configuration/data/source/software differs from original run')
        try:
            payload = torch.load(path / 'training.pth', map_location='cpu', weights_only=False)
            if payload['step'] != meta['step'] or digest(payload['contract']) != meta['contract_sha256']:
                raise ValueError('Payload metadata mismatch')
        except (OSError, ValueError, KeyError, RuntimeError, EOFError) as error:
            rejected.append(f'{path}: {error}')
            continue
        return path, payload, rejected
    raise ValueError('No valid checkpoint to resume: ' + '; '.join(rejected))


def save_snapshot(root, model, optimizer, step, cfg_dict, run_contract, schedules, initialization, *, milestone,
                  data_cursor=None):
    parent = root / ('checkpoints' if milestone else 'recovery')
    parent.mkdir(exist_ok=True)
    destination = parent / f'step_{step:06d}'
    if destination.exists():
        validate_snapshot(destination)
        raise FileExistsError(f'Refusing to overwrite snapshot: {destination}')
    temporary = parent / ('.pending_' + uuid.uuid4().hex)
    temporary.mkdir()
    # Meta delays center updates until the next forward. Flush the pending
    # reductions now so state_dict includes them; the next forward is equivalent.
    model.dino_loss.apply_center_update()
    model.ibot_patch_loss.apply_center_update()
    payload = dict(schema=1, step=step, model=model.state_dict(), optimizer=optimizer.state_dict(),
                   config=cfg_dict, contract=run_contract, rng=rng_state(),
                   schedule={k: v.schedule.tolist() for k, v in schedules.items()},
                   data_cursor=dict(completed_steps=step, image_presentations=step * cfg_dict['train']['batch_size_per_gpu'],
                                    policy='draw_seed_v1; epoch_randperm; drop_last'))
    if data_cursor is not None:
        payload['data_cursor'] = data_cursor
    torch.save(payload, temporary / 'training.pth')
    export = temporary / 'teacher_backbone.pth'
    if step == 0:
        # Verified identical tensors; preserve exact release bytes to reuse the
        # reproduced frozen baseline (whose identity is a file SHA256).
        shutil.copyfile(initialization, export)
    else:
        torch.save({k: v.detach().cpu() for k, v in model.teacher.backbone.state_dict().items()}, export)
    files = {}
    for name in ('training.pth', 'teacher_backbone.pth'):
        with (temporary / name).open('rb') as stream:
            os.fsync(stream.fileno())
        files[name] = sha256(temporary / name)
    atomic_json(temporary / 'complete.json', dict(schema=1, step=step, files=files,
                contract_sha256=digest(run_contract), milestone=milestone))
    os.rename(temporary, destination)
    # Flush the rename where the filesystem supports directory fsync.
    directory_fd = os.open(parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    atomic_json(root / 'latest.json', dict(path=str(destination.relative_to(root)), step=step))
    return destination


def trim_recovery(root, keep):
    valid = []
    for path in sorted((root / 'recovery').glob('step_*'), reverse=True):
        try:
            validate_snapshot(path)
            valid.append(path)
        except (OSError, ValueError, KeyError):
            pass
    for path in valid[keep:]:
        shutil.rmtree(path)


def reconcile_metrics(root, step):
    """Keep exactly the committed prefix; preserve discarded crash-tail evidence."""
    path = root / 'metrics.jsonl'
    if not path.exists():
        if step:
            raise ValueError('Missing metrics prefix for resumed checkpoint')
        return
    lines = path.read_text().splitlines(keepends=True)
    prefix = []
    for line in lines:
        if len(prefix) == step:
            break
        row = json.loads(line)
        if row['step'] != len(prefix) + 1:
            raise ValueError('Noncontiguous metrics prefix')
        prefix.append(line)
    if len(prefix) != step:
        raise ValueError('Metrics end before saved checkpoint')
    if len(prefix) != len(lines):
        shutil.copyfile(path, root / ('metrics_before_resume_' + uuid.uuid4().hex + '.jsonl'))
        temp = path.with_suffix('.pending')
        temp.write_text(''.join(prefix))
        os.replace(temp, path)
