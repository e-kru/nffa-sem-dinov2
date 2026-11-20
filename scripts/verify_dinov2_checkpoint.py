"""Reload a trusted local pilot checkpoint and verify its raw backbone export.

The full checkpoint contains Python/NumPy RNG state and is loaded with
weights_only=False. Use this only for checkpoints created by our own pilot.
The raw backbone export is loaded with weights_only=True.
"""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["XFORMERS_DISABLED"] = "1"
    from src.ssl.upstream import PIN, activate
    activate()
    import torch
    from dinov2.hub.backbones import dinov2_vits14
    from src.ssl.adapter import SingleGPUAdapter, load_config

    torch.set_num_threads(1)
    output = args.output or args.run / "checkpoint_verification.json"
    if output.exists():
        raise FileExistsError(output)
    report = json.loads((args.run / "report.json").read_text())
    if report["status"] != "passed" or report["upstream_commit"] != PIN["commit"]:
        raise ValueError("Expected a successful pilot with the pinned Meta revision")
    if report["checkpoint_sha256"] != PIN["backbone_sha256"]:
        raise ValueError("Pilot did not use the pinned released initialization")
    cfg = load_config(args.run / "config.yaml")
    # Batch-size override is recorded in config, including its scaled LR.
    model = SingleGPUAdapter(cfg, None, allow_random_init=True)
    saved = torch.load(args.run / "pilot_checkpoint.pth", map_location="cpu", weights_only=False)
    if saved["upstream_commit"] != PIN["commit"] or saved["iteration"] + 1 != report["steps_completed"]:
        raise ValueError("Checkpoint identity or iteration disagrees with the pilot report")
    model.load_state_dict(saved["model"], strict=True)
    for name, value in model.state_dict().items():
        if not torch.equal(value, saved["model"][name]) or not torch.isfinite(value).all():
            raise RuntimeError(f"State round trip failed: {name}")
    optimizer = torch.optim.AdamW(model.parameter_groups(),
                                 betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2))
    optimizer.load_state_dict(saved["optimizer"])
    if not optimizer.state:
        raise RuntimeError("No optimizer state was restored")
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                raise RuntimeError("Nonfinite optimizer state")
    exported = torch.load(args.run / "teacher_backbone.pth", map_location="cpu", weights_only=True)
    teacher_state = model.teacher.backbone.state_dict()
    if exported.keys() != teacher_state.keys() or not all(torch.equal(v, teacher_state[k]) for k, v in exported.items()):
        raise RuntimeError("Backbone export differs from the checkpoint teacher")
    reference = dinov2_vits14(pretrained=False).eval()
    reference.load_state_dict(exported, strict=True)
    generator = torch.Generator().manual_seed(cfg.train.seed)
    images = torch.randn(2, 3, cfg.crops.global_crops_size, cfg.crops.global_crops_size, generator=generator)
    with torch.no_grad():
        expected = reference(images, is_training=True)
        actual = model.teacher.backbone(images, is_training=True)
    errors = {}
    for name in ("x_norm_clstoken", "x_norm_patchtokens"):
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
        errors[name] = float((actual[name] - expected[name]).abs().max())
    result = {"status": "passed", "torch": torch.__version__, "upstream_commit": PIN["commit"],
              "iteration": saved["iteration"], "model_state_strict_equal": True,
              "optimizer_state_restored": True, "export_strict_equal": True,
              "export_features_max_abs_error": errors,
              "teacher_eval": not model.teacher.training,
              "teacher_frozen": all(not p.requires_grad for p in model.teacher.parameters())}
    with output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
