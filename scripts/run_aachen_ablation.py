"""Run resolution A and quadrant B ablations without ever opening test files."""
import argparse
import json
import os
from pathlib import Path

import torch

from src.aachen_ablation.cache import (
    OriginalCachedDataset, atomic_json, cache_contract, extract_cache,
    encoder_fingerprint, find_cache, git_info, make_encoder, make_pinned_encoder, source_hashes, validate_cache,
)
from src.aachen_ablation.data import digest, inspect_data, load_mask, manifest_rows
from src.aachen_ablation.devices import CudaMemoryProfiler, select_device
from src.aachen_ablation.engine import (benchmark_end_to_end, evaluate_run, make_decoder, train_run)
from src.aachen_ablation.geometry import VARIANTS
from src.aachen_ablation.metrics import training_size_bins

DEFAULT_ROOT = Path("outputs/aachen_heerlen/resolution_cropping_v1")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=Path("outputs/aachen_heerlen/splits/group_split_seed42.csv"))
    p.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--cache-root", type=Path, help="Existing cache directory root; defaults to output-root/caches")
    p.add_argument("--experiment", choices=("A", "B", "both"), default="both")
    p.add_argument("--variants", nargs="+", choices=tuple(VARIANTS))
    p.add_argument("--heads", nargs="+", choices=("linear", "cnn8"), default=["linear", "cnn8"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--stage", choices=("preflight", "extract", "train", "evaluate", "report", "all"), default="all")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--determinism", choices=("off", "warn", "strict"), default="off",
                   help="off: fixed seeds without deterministic CUDA kernels (default); warn/strict opt in")
    p.add_argument("--encoder-checkpoint", type=Path)
    p.add_argument("--encoder-name", choices=("dinov2_vits14", "dinov2_vitl14"), default="dinov2_vits14")
    p.add_argument("--pinned-encoder", action="store_true",
                   help="Load an explicit backbone checkpoint using the pinned local Meta checkout")
    p.add_argument("--cache-dtype", choices=("float32", "float16"), default="float32")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8, help="Effective batch of ORIGINALS")
    p.add_argument("--micro-batch-size", type=int, default=8, help="Originals per forward/backward")
    p.add_argument("--validation-batch-size", type=int, default=8)
    p.add_argument("--extraction-batch-size", type=int, default=4, help="Crops per encoder forward")
    p.add_argument("--workers", "--num-workers", dest="workers", type=int, default=8, help="CPU data workers per stage; smoke tests use 0")
    p.add_argument("--linear-lr", type=float, default=1e-3)
    p.add_argument("--cnn8-lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--timing-images", type=int, default=5)
    p.add_argument("--example-count", type=int, default=3)
    p.add_argument("--resume", action="store_true", help="Resume matching incomplete caches/runs at original/epoch boundaries")
    p.add_argument("--verified-data", type=Path, help="Reuse audited records from an existing A3 config.json; no image/mask rehashing")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--smoke-test", action="store_true", help="FAKE encoder, max 2 train/val originals, isolated output")
    return p


def selected_variants(args):
    allowed = [k for k in VARIANTS if args.experiment == "both" or k.startswith(args.experiment)]
    names = args.variants or allowed
    if set(names) - set(allowed) or len(set(names)) != len(names):
        raise ValueError("Variants must be unique and belong to --experiment")
    return [VARIANTS[k] for k in names]


def run_config(args, variant, head, seed, cache_path, meta, bins):
    # Stable identity excludes runtime Git status; original extraction Git info is retained.
    return {"protocol": "original-mask-v1", "variant": variant.config(), "head": head, "seed": seed,
            "epochs": min(args.epochs, 2) if args.smoke_test else args.epochs,
            "patience": args.patience, "batch_size": args.batch_size,
            "micro_batch_size": args.micro_batch_size, "workers": args.workers,
            "validation_batch_size": args.validation_batch_size, "persistent_workers": args.workers > 0,
            "execution_version": "batched-v2", "training_sources": source_hashes(),
            "lr": args.linear_lr if head == "linear" else args.cnn8_lr,
            "weight_decay": args.weight_decay, "feature_cache": str(cache_path),
            "cache_metadata": meta, "component_size_bins": bins,
            "torch_version": str(torch.__version__), "device": args.device, "manifest": str(args.manifest.resolve()),
            "loss": "mean per-original valid-pixel cross entropy; crop areas pooled",
            "optimizer": "AdamW", "scheduler": None,
            "checkpoint_selection": "validation mIoU at 0.5; CE tie-break; tolerance 1e-8",
            "deterministic_algorithms": args.determinism != "off", "determinism_policy": args.determinism,
            "deterministic_warn_only": args.determinism == "warn",
            "nonscientific_smoke_test": args.smoke_test, "timing_images": args.timing_images}


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.encoder_name == "dinov2_vitl14" and (not args.pinned_encoder or args.smoke_test):
        p.error("Large evaluation requires an explicit pinned encoder and real features")
    if args.pinned_encoder and args.encoder_checkpoint is None:
        p.error("--pinned-encoder requires --encoder-checkpoint")
    for key in ("epochs", "patience", "batch_size", "micro_batch_size", "validation_batch_size", "extraction_batch_size", "timing_images"):
        if getattr(args, key) < 1:
            p.error(f"--{key.replace('_', '-')} must be positive")
    if args.workers < 0 or args.example_count < 0 or min(args.linear_lr, args.cnn8_lr) <= 0 or args.weight_decay < 0:
        p.error("Invalid worker/example count, learning rate or weight decay")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.heads)) != len(args.heads):
        p.error("Seeds/heads must be unique")
    if args.smoke_test:
        args.workers = 0
    variants = selected_variants(args)
    root = args.output_root.resolve()
    if args.smoke_test:
        root = root / "NONSCIENTIFIC_SMOKE"
    if args.stage == "report" and not args.dry_run:
        from src.aachen_ablation.report import create_report
        create_report(root, args.example_count)
        return
    manifest = args.manifest.resolve()
    rows = manifest_rows(manifest, 2 if args.smoke_test else None)
    # Full inspection before any GPU allocation; never reads test files.
    if args.verified_data:
        if args.smoke_test:
            p.error('Verified scientific inventories cannot be used for a smoke test')
        from src.ssl.ablation_data import segmentation
        records, preflight = segmentation(args.verified_data, manifest, rows, variants)
    else:
        records, preflight = inspect_data(manifest, rows, variants)
    contracts = {v.name: cache_contract(manifest, records, v, args.cache_dtype,
                                       args.encoder_checkpoint, args.smoke_test, args.encoder_name) for v in variants}
    if args.pinned_encoder:
        from src.ssl.upstream import PIN
        for contract in contracts.values():
            contract.update(encoder_source="pinned Meta checkout", upstream_commit=PIN["commit"])
            contract["pinned_encoder_sources"] = {name: value for name, value in source_hashes().items()
                                                  if name in ("src/aachen_ablation/cache.py",
                                                              "src/downstream/pinned_encoder.py")}
    cache_root = args.cache_root.resolve() if args.cache_root else root / "caches"
    cache_paths = {v.name: find_cache(cache_root, contracts[v.name]) for v in variants}
    plan = {"preflight": preflight, "output_root": str(root),
            "nonscientific_smoke_test": args.smoke_test, "runs": []}
    for v in variants:
        contract = contracts[v.name]
        cache = cache_paths[v.name]
        dim = contract["feature_shape"][1]
        size = len(records) * dim * v.tokens * (4 if args.cache_dtype == "float32" else 2)
        plan["runs"].append({"variant": v.name, "input_wh": [v.width, v.height],
                             "feature_shape": contract["feature_shape"], "patch_tokens": v.tokens,
                             "cache": str(cache), "cache_GiB": size/2**30,
                             "feature_input_MiB_per_micro_batch": args.micro_batch_size*dim*v.tokens*4/2**20,
                             "cnn8_activation_estimate_MiB": args.micro_batch_size*v.tokens*(dim+24*128)*4/2**20,
                             "original_logits_MiB_per_micro_batch": args.micro_batch_size*max(r["original_size"][0]*r["original_size"][1] for r in records)*2*4/2**20,
                             "training_runs": [f"{v.name}/{h}/seed{s}" for h in args.heads for s in args.seeds]})
    plan["total_cache_GiB"] = sum(r["cache_GiB"] for r in plan["runs"])
    plan["memory_note"] = "Cache estimates exclude NPY headers. Training activations depend on original masks, head and micro batch; A100 80 GB recommended."
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return
    root.mkdir(parents=True, exist_ok=True)
    # Immutable provenance keyed by the complete inspected data snapshot.
    preflight_path = root / f"preflight_{digest(records)[:16]}.json"
    if not preflight_path.exists():
        atomic_json(preflight_path, {**preflight, "manifest": str(manifest), "git": git_info(), "records": records})
    if args.stage == "preflight":
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        device = select_device(args.device)
    except (ValueError, RuntimeError) as exc:
        p.error(str(exc))
    args.device = str(device)
    if device.type != "cuda" and not args.smoke_test:
        p.error("Scientific extraction/training/evaluation requires CUDA; CPU is available with --smoke-test")
    encoder = None
    if args.stage in ("extract", "all"):
        # Reuse precisely the same frozen model across every resolution/crop variant.
        existing = []
        for v in variants:
            path = cache_paths[v.name]
            if (path / "cache_metadata.json").exists():
                existing.append(validate_cache(path, contracts[v.name], complete=False))
        known_weights = {m["encoder_state_sha256"] for m in existing if "encoder_state_sha256" in m}
        if len(known_weights) > 1:
            raise ValueError("Backbone weights differ across variant caches")
        incomplete = len(existing) < len(variants) or any(not m.get("complete") for m in existing)
        if incomplete:
            snapshot = next((m["encoder_weights_snapshot"] for m in existing if "encoder_weights_snapshot" in m), None)
            from functools import partial
            factory = partial(make_pinned_encoder, name=args.encoder_name) if args.pinned_encoder else make_encoder
            encoder = factory(device, snapshot or args.encoder_checkpoint, args.smoke_test)
            if known_weights and encoder_fingerprint(encoder) not in known_weights:
                raise ValueError("Encoder changed across variants; use the original checkpoint")
        for v in variants:
            extract_cache(cache_paths[v.name], contracts[v.name], manifest, v, device,
                          encoder, args.extraction_batch_size, args.resume, workers=args.workers)
        del encoder
        encoder = None
    if args.stage == "extract":
        return
    bins = training_size_bins(load_mask(manifest, r).numpy() for r in records if r["split"] == "train")
    bins["manifest_sha256"] = contracts[variants[0].name]["manifest_sha256"]
    validated = {v.name: validate_cache(cache_paths[v.name], contracts[v.name]) for v in variants}
    if len({m["encoder_state_sha256"] for m in validated.values()}) != 1:
        raise ValueError("Backbone weights differ across variant caches")
    for v in variants:
        contract = contracts[v.name]
        cache_path = cache_paths[v.name]
        meta = validated[v.name]
        train_ds = OriginalCachedDataset(cache_path, contract, manifest, "train", verified_metadata=meta)
        val_ds = OriginalCachedDataset(cache_path, contract, manifest, "val", verified_metadata=meta)
        for head in args.heads:
            for seed in args.seeds:
                config = run_config(args, v, head, seed, cache_path, meta, bins)
                output = root / "runs" / v.name / head / f"seed{seed}_{digest(config)[:12]}"
                if args.stage in ("train", "all"):
                    train_run(output, config, train_ds, val_ds, v, device, args.resume)
                if args.stage in ("evaluate", "all"):
                    evaluate_run(output, config, val_ds, v, device, bins)
                    metrics_path = output / "validation_metrics.json"
                    metrics = json.loads(metrics_path.read_text())
                    if metrics["end_to_end_seconds_per_original"] is None:
                        from functools import partial
                        factory = partial(make_pinned_encoder, name=args.encoder_name) if args.pinned_encoder else make_encoder
                        encoder = factory(device, meta.get("encoder_weights_snapshot") or args.encoder_checkpoint, args.smoke_test)
                        if encoder_fingerprint(encoder) != meta["encoder_state_sha256"]:
                            raise ValueError("Timing encoder differs from cached encoder")
                        model = make_decoder(head, meta["contract"]["feature_shape"][1]).to(device)
                        state = torch.load(output / "best_model.pt", map_location="cpu", weights_only=True)
                        model.load_state_dict(state["model_state_dict"])
                        memory = CudaMemoryProfiler(device)
                        memory.start()
                        metrics.update(benchmark_end_to_end(model, encoder, val_ds, v, device, args.timing_images))
                        metrics["timing_peak_gpu_bytes"] = memory.peak_bytes()
                        metrics["timing_memory_profiling"] = memory.metadata()
                        atomic_json(metrics_path, metrics)
                        del encoder, model
                        encoder = None
    if args.stage == "all":
        from src.aachen_ablation.report import create_report
        create_report(root, args.example_count)


if __name__ == "__main__":
    main()
