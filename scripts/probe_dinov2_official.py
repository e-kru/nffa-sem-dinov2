"""Bounded test of Meta's complete SSLMetaArch + do_train, on one visible GPU.

Only data injection and strict backbone initialization are supplied locally.
No upstream training code, loss, FSDP method, or EMA method is patched.
"""

import argparse
import json
import math
import os
from pathlib import Path
import platform
import traceback

from src.ssl.upstream import PIN, activate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2, choices=range(1, 4))
    parser.add_argument("--sem", action="store_true", help="Use curated SEM train images instead of synthetic images")
    parser.add_argument("--manifest", type=Path, default=Path("outputs/dataset_v3/manifest.csv"))
    parser.add_argument("--split", type=Path, default=Path("outputs/dataset_v3/splits/split_B_full.csv"))
    parser.add_argument("--holdout-split", type=Path, action="append")
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ.pop("XFORMERS_DISABLED", None)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"upstream_commit": PIN["commit"], "requested_steps": args.steps,
              "python": platform.python_version(), "status": "failed", "stage": "imports",
              "mode": "SEM_train_only" if args.sem else "synthetic"}
    try:
        root = activate()
        import torch
        torch.set_num_threads(1)
        report.update(torch=torch.__version__, cuda=torch.version.cuda,
                      cuda_available=torch.cuda.is_available(),
                      gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
        # Import the actual complete trainer even on CPU to expose dependencies.
        from dinov2.train import train
        from dinov2.train.ssl_meta_arch import SSLMetaArch
        from omegaconf import OmegaConf
        import dinov2.distributed as distributed
        from dinov2.utils.utils import fix_random_seeds
        from torchvision.datasets import FakeData
        report["stage"] = "single_gpu_setup"
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Expose exactly one CUDA GPU with CUDA_VISIBLE_DEVICES")
        if ("A100" not in torch.cuda.get_device_name(0)
                or torch.cuda.get_device_properties(0).total_memory < 75 * 1024**3):
            raise RuntimeError("Use one A100 80 GB allocation for this pilot")
        report["gpu"] = torch.cuda.get_device_name(0)
        distributed.enable(overwrite=True)
        fix_random_seeds(42)
        cfg = OmegaConf.load(root / "dinov2/configs/ssl_default_config.yaml")
        # 518 preserves the released positional grid for strict loading. The
        # smaller 224 SEM adapter interpolates that grid in Meta's own backbone.
        cfg = OmegaConf.merge(cfg, {
            "student": {"arch": "vit_small", "patch_size": 14},
            "train": {"batch_size_per_gpu": 2, "num_workers": 0,
                      "OFFICIAL_EPOCH_LENGTH": args.steps, "output_dir": str(args.output.resolve())},
            "crops": {"global_crops_size": 518, "local_crops_size": 98, "local_crops_number": 2},
            "optim": {"epochs": 1, "warmup_epochs": 0, "freeze_last_layer_epochs": 0,
                      "lr": 0.004 * math.sqrt(2 / 1024)},
            "teacher": {"warmup_teacher_temp_epochs": 1},
            "evaluation": {"eval_period_iterations": 0},
        })
        OmegaConf.save(cfg, args.output / "config.yaml")
        report["stage"] = "construct_and_strict_initialize"
        from src.ssl.data import sha256
        if sha256(args.checkpoint) != PIN["backbone_sha256"]:
            raise ValueError("Expected the pinned released Meta ViT-S/14 backbone")
        report["checkpoint_sha256"] = sha256(args.checkpoint)
        model = SSLMetaArch(cfg)
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.student.backbone.load_state_dict(state, strict=True)
        report["checkpoint_strict_loaded"] = True
        model.cuda()
        report["stage"] = "prepare_fsdp"
        model.prepare_for_distributed_training()
        # Meta's full loop creates its own augmentation, collator, optimizer,
        # schedules, checkpointer and sharded infinite sampler.
        if args.sem:
            from src.config import DATASET_ROOT
            from src.ssl.data import SEMTrainDataset, train_records
            holdouts = args.holdout_split or [Path("outputs/dataset_v3/splits/split_A_exact.csv")]
            records = train_records(args.manifest, args.split, holdouts)
            (args.output / "train_records.json").write_text(json.dumps(records, indent=2) + "\n")
            report.update(train_images=len(records), manifest_sha256=sha256(args.manifest),
                          split_sha256=sha256(args.split),
                          holdout_splits=[{"path": str(p.resolve()), "sha256": sha256(p)} for p in holdouts])
            train.make_dataset = lambda **kw: SEMTrainDataset(args.data_root or DATASET_ROOT, records, kw["transform"])
        else:
            train.make_dataset = lambda **kw: FakeData(
                size=16, image_size=(3, 560, 560), transform=kw["transform"],
                target_transform=kw["target_transform"], random_offset=42,
            )
        report["stage"] = "official_do_train"
        metrics = train.do_train(cfg, model, resume=False)
        torch.cuda.synchronize()
        if not metrics or not all(math.isfinite(float(v)) for v in metrics.values()):
            raise RuntimeError(f"Nonfinite or absent official metrics: {metrics}")
        report.update(status="passed", stage="complete", metrics=metrics,
                      peak_memory_bytes=torch.cuda.max_memory_allocated())
    except Exception as exc:
        report.update(error_type=type(exc).__name__, error=str(exc))
        (args.output / "traceback.txt").write_text(traceback.format_exc())
    finally:
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        import sys
        sys.exit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
