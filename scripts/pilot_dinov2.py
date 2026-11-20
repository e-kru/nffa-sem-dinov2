"""At most 20 SEM SSL steps with Meta's imported objectives; never a full run."""

import argparse
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import time
import traceback

from src.ssl.upstream import PIN, ROOT, activate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/sem_ssl_pilot.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3, choices=range(2, 21))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--manifest", type=Path, default=ROOT / "outputs/dataset_v3/manifest.csv")
    parser.add_argument("--split", type=Path, default=ROOT / "outputs/dataset_v3/splits/split_B_full.csv")
    parser.add_argument("--holdout-split", type=Path, action="append",
                        help="Additional split whose val/test images are excluded, default split A")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--official-report", type=Path,
                        help="Required on CUDA: failure report from the official probe on this GPU stack")
    args = parser.parse_args()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    # The no-xFormers path uses Meta's native PyTorch attention and loss fallback.
    # Must be chosen before importing the upstream layers. No monkey patches.
    os.environ["XFORMERS_DISABLED"] = "1"
    upstream = activate()
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from torchvision.datasets import FakeData
    from src.config import DATASET_ROOT
    from src.ssl.adapter import SingleGPUAdapter, load_config, schedules, train_step
    from src.ssl.data import SEMTrainDataset, augmentation_and_collate, seed_worker, sha256, train_records

    torch.set_num_threads(1)
    if args.device == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Expose exactly one A100 with CUDA_VISIBLE_DEVICES")
        if "A100" not in torch.cuda.get_device_name(0):
            raise RuntimeError("This pilot is configured for one A100")
        if torch.cuda.get_device_properties(0).total_memory < 75 * 1024**3:
            raise RuntimeError("This pilot requires the project's A100 80 GB allocation")
        if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 is not supported on the selected GPU")
        if not args.official_report:
            raise ValueError("Run probe_dinov2_official first and supply its report.json")
        official = json.loads(args.official_report.read_text())
        if (official["upstream_commit"] != PIN["commit"] or official["status"] != "failed"
                or official.get("torch") != torch.__version__
                or official.get("gpu") != torch.cuda.get_device_name(0)):
            raise ValueError("Fallback requires an official failure on this revision and GPU software stack")
    elif args.precision != "fp32":
        raise ValueError("CPU verification requires --precision fp32")
    cfg = load_config(args.config)
    cfg.train.num_workers = args.num_workers
    if args.batch_size is not None:
        if args.batch_size < 2:
            raise ValueError("KoLeo requires at least two images")
        cfg.train.batch_size_per_gpu = args.batch_size
        cfg.optim.lr = cfg.optim.base_lr * (args.batch_size / 1024) ** 0.5
    if args.steps > cfg.optim.epochs * cfg.train.OFFICIAL_EPOCH_LENGTH:
        raise ValueError("Pilot steps exceed the configured schedule")
    if sha256(args.checkpoint) != PIN["backbone_sha256"]:
        raise ValueError("Checkpoint differs from the pinned released Meta ViT-S/14 artifact")
    seed = cfg.train.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "mode": "synthetic" if args.synthetic else "SEM_train_only",
              "upstream_commit": PIN["commit"], "upstream_directory": str(upstream),
              "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha256(args.checkpoint),
              "torch": torch.__version__, "torchvision": __import__("torchvision").__version__,
              "python": platform.python_version(), "cuda": torch.version.cuda,
              "device": args.device, "precision": args.precision,
              "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
              "steps_requested": args.steps, "steps_completed": 0,
              "official_report": str(args.official_report) if args.official_report else None,
              "deviations_document": "docs/sem_ssl_pilot.md"}
    try:
        report["project_git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        report["project_git_status"] = subprocess.check_output(["git", "status", "--short"], text=True).strip()
        source_files = [*sorted((ROOT / "src/ssl").glob("*.py")), Path(__file__), args.config,
                        ROOT / "dinov2-upstream.json", ROOT / "src/preprocessing_v2.py"]
        report["source_sha256"] = {
            str(p.resolve().relative_to(ROOT) if p.resolve().is_relative_to(ROOT) else p.resolve()): sha256(p)
            for p in source_files
        }
        (args.output / "packages.txt").write_text(subprocess.check_output(
            [os.sys.executable, "-m", "pip", "freeze"], text=True))
        OmegaConf.save(cfg, args.output / "config.yaml")
        transform, collate = augmentation_and_collate(cfg)
        batch_size = cfg.train.batch_size_per_gpu
        if args.synthetic:
            dataset = FakeData(size=max(16, batch_size * args.steps), image_size=(3, 256, 256),
                               transform=transform, random_offset=seed)
        else:
            holdouts = args.holdout_split or [ROOT / "outputs/dataset_v3/splits/split_A_exact.csv"]
            records = train_records(args.manifest, args.split, holdouts)
            root = args.data_root or DATASET_ROOT
            dataset = SEMTrainDataset(root, records, transform)
            (args.output / "train_records.json").write_text(json.dumps(records, indent=2) + "\n")
            report.update(manifest=str(args.manifest.resolve()), manifest_sha256=sha256(args.manifest),
                          split=str(args.split.resolve()), split_sha256=sha256(args.split),
                          holdout_splits=[{"path": str(p.resolve()), "sha256": sha256(p)} for p in holdouts],
                          data_root=str(root.resolve()), train_images=len(records))
        if len(dataset) < batch_size:
            raise ValueError("Fewer training images than one complete batch")
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                            num_workers=args.num_workers, worker_init_fn=seed_worker,
                            pin_memory=args.device == "cuda", collate_fn=collate, generator=generator)
        model = SingleGPUAdapter(cfg, args.checkpoint).to(args.device)
        report["pretrained_tensors_equal"] = model.pretrained_verified
        report["teacher_student_initial_equal"] = all(
            torch.equal(v, model.teacher.state_dict()[k]) for k, v in model.student.state_dict().items())
        report["components"] = {name: type(getattr(model, name)).__module__
                                for name in ("dino_loss", "ibot_patch_loss", "koleo_loss")}
        report["components"].update(backbone=type(model.student.backbone).__module__,
                                     dino_head=type(model.student.dino_head).__module__)
        if not report["teacher_student_initial_equal"]:
            raise RuntimeError("Student and teacher must begin with identical state")
        initial_patch = model.student.backbone.patch_embed.proj.weight.detach().clone()
        optimizer = torch.optim.AdamW(model.parameter_groups(),
                                     betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2))
        sched = schedules(cfg)
        started = time.monotonic()
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        iterator = iter(loader)
        with (args.output / "metrics.jsonl").open("x") as metrics:
            for iteration in range(args.steps):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    batch = next(iterator)
                values = train_step(model, optimizer, batch, sched, iteration, precision=args.precision)
                metrics.write(json.dumps(values) + "\n")
                metrics.flush()
                report["steps_completed"] += 1
                print(json.dumps(values), flush=True)
        if args.device == "cuda":
            torch.cuda.synchronize()
        report.update(elapsed_seconds=time.monotonic() - started,
                      peak_memory_bytes=torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
                      backbone_updated=not torch.equal(initial_patch, model.student.backbone.patch_embed.proj.weight),
                      teacher_has_no_grad=all(p.grad is None for p in model.teacher.parameters()))
        if not report["backbone_updated"] or not report["teacher_has_no_grad"]:
            raise RuntimeError("Backbone update / teacher gradient invariant failed")
        # Flush delayed centers once before serialization. The next softmax
        # would do this same operation; pending async Python fields are not in
        # state_dict. No checkpoint silently loses the final center update.
        model.dino_loss.apply_center_update()
        model.ibot_patch_loss.apply_center_update()
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "iteration": args.steps - 1, "config": OmegaConf.to_container(cfg),
                    "upstream_commit": PIN["commit"], "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if args.device == "cuda" else [],
                    "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
                    "loader_rng": generator.get_state()}, args.output / "pilot_checkpoint.pth")
        torch.save(model.teacher.backbone.state_dict(), args.output / "teacher_backbone.pth")
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        (args.output / "traceback.txt").write_text(traceback.format_exc())
        raise
    finally:
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
