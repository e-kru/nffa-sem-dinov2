"""Verify immutable milestone files, full state and frozen original Meta exports."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ['XFORMERS_DISABLED'] = '1'
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from omegaconf import OmegaConf

from src.ssl.adapter import SingleGPUAdapter
from src.ssl.checkpoints import atomic_json, validate_snapshot
from src.ssl.experiment import REFERENCE, digest
from src.ssl.upstream import PIN, activate


def verify(path, *, forward=True, initialization=None):
    activate()
    metadata = validate_snapshot(path)
    payload = torch.load(path / 'training.pth', map_location='cpu', weights_only=False)
    if payload['step'] != metadata['step'] or digest(payload['contract']) != metadata['contract_sha256']:
        raise ValueError('Training payload/metadata mismatch')
    cfg = OmegaConf.create(payload['config'])
    if payload['contract']['upstream_commit'] != PIN['commit']:
        raise ValueError('Wrong upstream revision')
    model = SingleGPUAdapter(cfg, None, allow_random_init=True)
    model.load_state_dict(payload['model'], strict=True)
    export = torch.load(path / 'teacher_backbone.pth', map_location='cpu', weights_only=True)
    if any(not torch.equal(v, export[k]) for k, v in model.teacher.backbone.state_dict().items()):
        raise ValueError('Export differs from checkpoint teacher')
    if any(not torch.isfinite(v).all() for v in model.state_dict().values()):
        raise ValueError('Nonfinite model/center state')
    if model.teacher.training or any(p.requires_grad for p in model.teacher.parameters()):
        raise ValueError('Teacher freeze/eval invariant failed')
    if payload['step'] == 0 and metadata['files']['teacher_backbone.pth'] != PIN['backbone_sha256']:
        raise ValueError('Step 0 is not the original release')
    if payload['step'] == 0 and any(not torch.equal(v, model.teacher.state_dict()[k])
                                  for k, v in model.student.state_dict().items()):
        raise ValueError('Step-0 student/teacher backbone or projection-head mismatch')
    frozen = cfg.student.get('freeze_first_blocks', 0)
    if frozen:
        initial = torch.load(initialization or REFERENCE, map_location='cpu', weights_only=True)
        prefixes = tuple(f'blocks.{i}.' for i in range(frozen))
        for module in (model.student.backbone, model.teacher.backbone):
            if any(not torch.equal(v, initial[k]) for k, v in module.state_dict().items() if k.startswith(prefixes)):
                raise ValueError('Frozen transformer blocks changed from pretrained initialization')
        if any(p.requires_grad or p.grad is not None for block in model.student.backbone.blocks[:frozen]
               for p in block.parameters()):
            raise ValueError('Frozen student blocks must not receive gradients')
    optimizer = torch.optim.AdamW(model.parameter_groups())
    optimizer.load_state_dict(payload['optimizer'])
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                raise ValueError('Nonfinite optimizer state')
    if payload['data_cursor']['completed_steps'] != payload['step']:
        raise ValueError('Data cursor mismatch')
    error = None
    if forward:
        from dinov2.hub.backbones import dinov2_vits14
        fresh = dinov2_vits14(pretrained=False).eval()
        fresh.load_state_dict(export, strict=True)
        with torch.inference_mode():
            images = torch.randn(2, 3, 224, 224)
            expected = model.teacher.backbone.forward_features(images)
            actual = fresh.forward_features(images)
            error = max(float((actual[k] - expected[k]).abs().max())
                        for k in ('x_norm_clstoken', 'x_norm_patchtokens'))
        if error != 0:
            raise ValueError(f'Frozen backbone reload mismatch: {error}')
    return dict(step=payload['step'], passed=True, teacher_sha256=metadata['files']['teacher_backbone.pth'],
                forward_max_abs_error=error, contract_sha256=metadata['contract_sha256'],
                frozen_blocks=frozen, frozen_blocks_unchanged=True if frozen else None,
                teacher_frozen=True, teacher_eval=True, export_matches_teacher=True,
                step0_student_teacher_equal=True if payload['step'] == 0 else None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--steps', nargs='+', type=int, default=[0, 50, 100, 200])
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    results = [verify(args.run / 'checkpoints' / f'step_{step:06d}') for step in args.steps]
    if len({item['contract_sha256'] for item in results}) != 1:
        raise ValueError('Milestones originate from different experiment contracts')
    atomic_json(args.output, dict(passed=True, checkpoints=results))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
