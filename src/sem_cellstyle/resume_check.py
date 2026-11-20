"""Strict resume diagnostics: state, RNG and real augmented batches stay visible."""
from copy import deepcopy
import hashlib
import math
import pickle
from pathlib import Path

import torch

from src.ssl.checkpoints import atomic_json, restore_rng, rng_state
from .runner import loader
from .training import accumulated_step


class ResumeCheckError(ValueError):
    def __init__(self, message, report):
        super().__init__(message)
        self.report = report
        self.result = None


def restore_probe_state(model, optimizer, payload):
    model.load_state_dict(payload['model'], strict=True)
    # In particular AdamW's CPU step tensors must not alias the checkpoint.
    optimizer.load_state_dict(deepcopy(payload['optimizer']))
    restore_rng(payload['rng'])


def optimizer_tensors(state):
    return {str(key)+'.'+name: value for key, entry in state['state'].items()
            for name, value in entry.items() if isinstance(value, torch.Tensor)}


def differences(expected, actual, *, may_differ=None):
    details, bad_keys, bad_elements, maximum = [], 0, 0, 0.
    allowed_finite_changes_only = True
    for name in sorted(set(expected) | set(actual)):
        a, b = expected.get(name), actual.get(name)
        if a is None or b is None or a.shape != b.shape or a.dtype != b.dtype:
            allowed_finite_changes_only = False
            bad_keys += 1
            if len(details) < 12:
                details.append(dict(name=name, reason='missing_tensor_or_shape_dtype_mismatch'))
            continue
        a, b = a.detach().cpu(), b.detach().cpu()
        if may_differ is not None and a.is_floating_point():
            if not (bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())):
                allowed_finite_changes_only = False
                bad_keys += 1
                if len(details) < 12:
                    details.append(dict(name=name, reason='nonfinite_tensor'))
                continue
        if torch.equal(a, b):
            continue
        if may_differ is None or not a.is_floating_point() or not may_differ(name):
            allowed_finite_changes_only = False
        bad_keys += 1
        count = int(torch.count_nonzero(a != b))
        error = float((a.float()-b.float()).abs().max())
        bad_elements += count
        if math.isfinite(error):
            maximum = max(maximum, error)
        else:
            allowed_finite_changes_only = False
        if len(details) < 12:
            reference_max = float(a.float().abs().max())
            details.append(dict(name=name, different_elements=count,
                                max_absolute_error=error if math.isfinite(error) else None,
                                reference_max_absolute_value=reference_max if math.isfinite(reference_max) else None))
    return dict(exact=bad_keys == 0, different_tensor_count=bad_keys,
                different_element_count=bad_elements, max_absolute_error=maximum, first_differences=details,
                allowed_finite_changes_only=allowed_finite_changes_only)


def resume_verdict(checks, model_difference, optimizer_difference, *, allow_cuda_update_drift):
    update_keys = ('model_exact', 'optimizer_exact')
    exact = all(checks[key] for key in update_keys)
    required_checks_pass = all(value for key, value in checks.items() if key not in update_keys)
    drift_accepted = (allow_cuda_update_drift and not exact and required_checks_pass
                      and model_difference['allowed_finite_changes_only']
                      and optimizer_difference['allowed_finite_changes_only'])
    return dict(passed=required_checks_pass and (exact or drift_accepted),
                update_bitwise_exact=exact, cuda_update_drift_accepted=drift_accepted)


def tensor_bytes(tensor):
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def rng_signature(state):
    return dict(cpu=hashlib.sha256(tensor_bytes(state['torch'])).hexdigest(),
                cuda=[hashlib.sha256(tensor_bytes(value)).hexdigest() for value in state['cuda'] or []],
                python_numpy=hashlib.sha256(pickle.dumps((state['python'], state['numpy']))).hexdigest())


class BatchTrace:
    def __init__(self, batches):
        self.batches, self.signatures = iter(batches), []

    def __iter__(self):
        return self

    def __next__(self):
        batch = next(self.batches)
        digest = hashlib.sha256()
        for key, value in sorted(batch.items()):
            if isinstance(value, torch.Tensor):
                if value.is_cuda:
                    raise ValueError('Resume input tracing expects CPU DataLoader batches')
                digest.update((key+str(value.dtype)+str(list(value.shape))).encode())
                digest.update(tensor_bytes(value))
        self.signatures.append(dict(draw_ids=batch.get('draw_ids'), record_indices=batch.get('record_indices'),
                                    augmented_tensors_sha256=digest.hexdigest()))
        return batch

    def close(self):
        # Release the exhausted iterator before creating another worker pool.
        self.batches = None


def modes(model):
    return {name: module.training for name, module in model.named_modules()}


def verify_resume_update(model, optimizer, dataset, collate, sched, update, payload, report_path, *,
                         precision='bf16', allow_cuda_update_drift=False):
    """Verify exact restore and compare both updates. Default to bitwise equality.

    The approved CUDA exception applies only to finite post-update Student/Teacher
    floating tensors and AdamW moments. Everything else must compare exactly.
    """
    device = next(model.parameters()).device
    if allow_cuda_update_drift and (device.type != 'cuda' or torch.are_deterministic_algorithms_enabled()):
        raise ValueError('Update drift exception requires the nondeterministic CUDA recipe')
    cfg = model.cfg
    restore_rng(payload['rng'])
    first_pre = dict(model=differences(payload['model'], model.state_dict()),
                     optimizer=differences(optimizer_tensors(payload['optimizer']), optimizer_tensors(optimizer.state_dict())),
                     optimizer_groups_exact=payload['optimizer']['param_groups'] == optimizer.state_dict()['param_groups'])
    first_batches = BatchTrace(loader(cfg, dataset, collate, device, update, update+1))
    before = rng_state()
    first_modes = modes(model)
    first = accumulated_step(model, optimizer, first_batches, sched, update, precision=precision)
    first_batches.close()
    first_rng_after = rng_signature(rng_state())
    first_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    first_optimizer = optimizer.state_dict()
    first_opt_tensors = {key: value.detach().cpu().clone() for key, value in optimizer_tensors(first_optimizer).items()}
    first_opt_groups = deepcopy(first_optimizer['param_groups'])
    del first_optimizer
    restore_probe_state(model, optimizer, payload)
    second_pre = dict(model=differences(payload['model'], model.state_dict()),
                      optimizer=differences(optimizer_tensors(payload['optimizer']), optimizer_tensors(optimizer.state_dict())),
                      optimizer_groups_exact=payload['optimizer']['param_groups'] == optimizer.state_dict()['param_groups'])
    second_batches = BatchTrace(loader(cfg, dataset, collate, device, update, update+1))
    # Recreating a loader must not alter the model RNG. Compare it explicitly.
    second_before = rng_state()
    second_modes = modes(model)
    second = accumulated_step(model, optimizer, second_batches, sched, update, precision=precision)
    second_batches.close()
    second_rng_after = rng_signature(rng_state())
    model_may_differ = (lambda key: key.startswith(('student.', 'teacher.'))) if allow_cuda_update_drift else None
    optimizer_may_differ = (lambda key: key.endswith(('.exp_avg', '.exp_avg_sq'))) if allow_cuda_update_drift else None
    model_difference = differences(first_state, model.state_dict(), may_differ=model_may_differ)
    optimizer_difference = differences(first_opt_tensors, optimizer_tensors(optimizer.state_dict()),
                                       may_differ=optimizer_may_differ)
    checks = dict(first_checkpoint_state_exact=all(first_pre[k]['exact'] for k in ('model', 'optimizer')) and first_pre['optimizer_groups_exact'],
                  restored_checkpoint_state_exact=all(second_pre[k]['exact'] for k in ('model', 'optimizer')) and second_pre['optimizer_groups_exact'],
                  rng_before_equal=rng_signature(before) == rng_signature(second_before),
                  rng_after_equal=first_rng_after == second_rng_after,
                  module_modes_equal=first_modes == second_modes,
                  batch_inputs_equal=first_batches.signatures == second_batches.signatures,
                  total_loss_exact=math.isfinite(first['total_loss']) and first['total_loss'] == second['total_loss'],
                  model_exact=model_difference['exact'], optimizer_exact=optimizer_difference['exact'],
                  optimizer_groups_exact=first_opt_groups == optimizer.state_dict()['param_groups'])
    verdict = resume_verdict(checks, model_difference, optimizer_difference,
                             allow_cuda_update_drift=allow_cuda_update_drift)
    report = dict(schema='sem-cellstyle-resume-diagnostics-v2', **verdict, checks=checks,
                  update=update, physical_batch=cfg.runtime.physical_batch, precision=precision,
                  torch=str(torch.__version__), device=str(device),
                  deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                  first_pre_state=first_pre, restored_pre_state=second_pre,
                  model_difference=model_difference, optimizer_difference=optimizer_difference,
                  first_metrics={k:v for k,v in first.items() if isinstance(v, (int, float, bool))},
                  restored_metrics={k:v for k,v in second.items() if isinstance(v, (int, float, bool))},
                  first_batch_signatures=first_batches.signatures, restored_batch_signatures=second_batches.signatures,
                  rng_before=rng_signature(before), restored_rng_before=rng_signature(second_before),
                  first_rng_after=first_rng_after, restored_rng_after=second_rng_after,
                  zero_tolerance=not allow_cuda_update_drift, scientific_training_started=False)
    atomic_json(report_path, report)
    if not report['passed']:
        failed = ', '.join(key for key, value in checks.items() if not value)
        raise ResumeCheckError('Resume comparison failed ('+failed+'); details: '+str(report_path), report)
    return report
