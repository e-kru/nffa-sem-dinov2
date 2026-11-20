"""Small synthetic before/after benchmark; never loads DINO or real data.

The reference engine and dataset are read from the pinned pre-optimization Git
revision. Fixtures, cProfile output and measurements are saved for inspection.
"""
import argparse
import cProfile
import copy
import json
import os
import pstats
import subprocess
import time
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.aachen_ablation import cache, engine
from src.aachen_ablation.data import inspect_data, manifest_rows
from src.aachen_ablation.geometry import VARIANTS, content_box, regions

REFERENCE_REVISION = "f3a90dfb13a03267d10fc458299bb23b1d9bd8e2"
REFERENCE_REPOSITORY = Path(os.environ.get("SEM_AACHEN_REFERENCE_REPO", "."))


def reference_module(name, revision, repository=REFERENCE_REPOSITORY):
    code = subprocess.check_output(
        ["git", "-C", str(repository), "show", f"{revision}:src/aachen_ablation/{name}.py"], text=True,
    )
    module = types.ModuleType(f"reference_{name}")
    module.__package__ = "src.aachen_ablation"
    module.__file__ = str(Path(__file__).resolve())
    exec(compile(code, f"git:{revision}/{name}.py", "exec"), module.__dict__)
    return module


def synthetic_cache(root, variant, count, width, height):
    import csv
    root.mkdir(parents=True)
    rng = np.random.default_rng(121)
    rows = []
    for i in range(count):
        image = rng.integers(20, 180, (height, width, 3), dtype=np.uint8)
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[height//4:height//2, width//4:width//2] = 1
        mask[0, :5] = 255
        Image.fromarray(image).save(root / f"image{i}.png")
        Image.fromarray(mask).save(root / f"mask{i}.png")
        rows.append(dict(image_id=str(i), image_path=f"image{i}.png", mask_path=f"mask{i}.png",
                         group_id=str(i), split="train" if i < count*2//3 else "val"))
    manifest = root / "manifest.csv"
    with manifest.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    records, _ = inspect_data(manifest, manifest_rows(manifest), [variant])
    contract = cache.cache_contract(manifest, records, variant, "float32", None, True)
    destination = root / "features"
    destination.mkdir()
    samples = []
    for i, row in enumerate(records):
        file = destination / f"{i:06d}.npy"
        np.save(file, rng.normal(size=contract["feature_shape"]).astype("float32"))
        boxes = regions(width, height, variant.quadrants)
        samples.append(dict(record=row, file=file.name, sha256=cache.sha256_file(file), crop_coordinates=boxes,
                            input_content_boxes=[content_box(b[2]-b[0], b[3]-b[1], variant) for b in boxes]))
    meta = dict(contract=contract, complete=True, samples=samples,
                encoder_state_sha256="synthetic-benchmark-only")
    cache.atomic_json(destination / "cache_metadata.json", meta)
    return manifest, destination, contract



def verify_same_inputs(reference, model, dataset, variant, device):
    original, optimized = copy.deepcopy(model).eval(), copy.deepcopy(model).eval()
    samples = [dataset[i] for i in range(min(2, len(dataset)))]
    batch = engine.pack_originals(samples)
    old_grid = torch.stack([original(s["features"].to(device)) for s in samples])
    expected = torch.stack([reference.original_loss(z, s["target"].to(device), variant) for z, s in zip(old_grid, samples)])
    new_grid = engine.forward_batch(optimized, batch, device)
    full = engine.project_batch(new_grid, batch["targets"], variant)
    actual = engine.full_resolution_losses(full, batch["targets"], device)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
    expected.mean().backward()
    actual.mean().backward()
    max_gradient_error = 0.
    for a, b in zip(original.parameters(), optimized.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=3e-4, atol=3e-6)
        max_gradient_error = max(max_gradient_error, float((a.grad-b.grad).abs().max().cpu()))
    old = reference.evaluate(original, samples, variant, device)
    new = engine.evaluate(optimized, samples, variant, device, batch_size=8)
    assert old["thresholds"] == new["thresholds"]
    return {"dropout_disabled_for_controlled_comparison": True,
            "max_absolute_per_original_loss_error": float((actual-expected).abs().max().detach().cpu()),
            "max_absolute_parameter_gradient_error": max_gradient_error,
            "confusion_matrices_identical": True}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("before", "after", "both"), default="both")
    parser.add_argument("--reference-revision", default=REFERENCE_REVISION)
    parser.add_argument("--reference-repository", type=Path, default=REFERENCE_REPOSITORY,
                        help="Local Git checkout containing the reference revision")
    parser.add_argument("--variant", choices=tuple(VARIANTS), default="A1")
    parser.add_argument("--heads", nargs="+", choices=("linear", "cnn8"), default=["linear", "cnn8"])
    parser.add_argument("--epochs", type=int, default=3, help="First epoch is warm-up; remaining epochs measured")
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--workers", type=int, default=0, help="0 permits use in IPC-restricted local sandbox")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.epochs < 2 or args.count < 6 or min(args.width, args.height) < 2 or args.workers < 0:
        parser.error("Use epochs>=2, count>=6 and positive dimensions")
    repo = args.reference_repository
    try:
        reference_engine = reference_module("engine", args.reference_revision, repo)
        reference_cache = reference_module("cache", args.reference_revision, repo)
        reference_cache.load_original = reference_module("data", args.reference_revision, repo).load_original
    except subprocess.CalledProcessError:
        parser.error("Supply --reference-repository with a local copy of the requested Git revision")
    from src.aachen_ablation.devices import select_device
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = select_device(args.device)
    root = args.output.resolve()
    if root.exists():
        raise FileExistsError(f"Benchmark destination must be new: {root}")
    root.mkdir(parents=True)
    variant = VARIANTS[args.variant]
    manifest, path, contract = synthetic_cache(root / "synthetic", variant, args.count, args.width, args.height)
    output = {"nonscientific_synthetic_benchmark": True, "device": str(device), "cuda_available": torch.cuda.is_available(),
              "torch_version": str(torch.__version__), "cpu_threads": torch.get_num_threads(),
              "originals": {"train": args.count*2//3, "val": args.count-args.count*2//3},
              "batch_size": 8, "before_micro_batch_size": 1, "after_micro_batch_size": 8, "seed": 42,
              "workers": args.workers, "reference_revision": args.reference_revision,
              "geometry": [args.width, args.height], "variant": args.variant, "warmup_epochs": 1, "measurements": [], "numerical_agreement": {},
              "includes_cprofile_overhead": True, "git": cache.git_info()}
    for mode in (["before", "after"] if args.mode == "both" else [args.mode]):
        implementation = reference_engine if mode == "before" else engine
        dataset_class = reference_cache.OriginalCachedDataset if mode == "before" else cache.OriginalCachedDataset
        train = dataset_class(path, contract, manifest, "train")
        val = dataset_class(path, contract, manifest, "val")
        for head in args.heads:
            engine.set_seed(42)
            model = engine.make_decoder(head).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001 if head == "linear" else .0003, weight_decay=.0001)
            profiler = cProfile.Profile()
            rows = []
            loader = None
            if mode == "after" and hasattr(engine, "make_loader"):
                loader = engine.make_loader(train, 8, args.workers, device, shuffle=True, seed=42)
                val_loader = engine.make_loader(val, 8, args.workers, device)
            policy = "warn" if mode == "before" else "off"
            with implementation.deterministic_execution(policy):
                for epoch in range(1, args.epochs+1):
                    if epoch == 2:
                        profiler.enable()
                    start = time.perf_counter()
                    kwargs = {"loader": loader} if loader is not None else {}
                    result = implementation.train_epoch(model, train, optimizer, variant, device, 8,
                                                        1 if mode == "before" else 8, 42, epoch, args.workers, **kwargs)
                    training = time.perf_counter()-start
                    start = time.perf_counter()
                    extra = {"loader": val_loader, "batch_size": 8} if loader is not None else {}
                    evaluation = implementation.evaluate(model, val, variant, device, workers=args.workers, **extra)
                    validation = time.perf_counter()-start
                    row = {"epoch": epoch, "train_seconds": training, "validation_seconds": validation,
                           "epoch_seconds": training+validation, "loss": result["train_loss"],
                           "val_loss": evaluation["loss"], "ma_iou": evaluation["thresholds"][0]["ma_iou"],
                           "instrumentation": {k: v for k, v in result.items() if k.endswith("seconds")}}
                    rows.append(row)
                    print(mode, head, json.dumps(row), flush=True)
            profiler.disable()
            with (root / f"profile_{mode}_{head}.txt").open("w") as file:
                pstats.Stats(profiler, stream=file).strip_dirs().sort_stats("cumulative").print_stats(35)
            measured = rows[1:]
            output["measurements"].append({"mode": mode, "head": head, "epochs": rows,
                "mean_epoch_seconds": float(np.mean([r["epoch_seconds"] for r in measured])),
                "mean_train_seconds": float(np.mean([r["train_seconds"] for r in measured])),
                "mean_validation_seconds": float(np.mean([r["validation_seconds"] for r in measured]))})
            if loader is not None:
                del loader, val_loader
    if args.mode in ("after", "both"):
        dataset = cache.OriginalCachedDataset(path, contract, manifest, "val")
        with engine.deterministic_execution("off"):
            for head in args.heads:
                engine.set_seed(42)
                output["numerical_agreement"][head] = verify_same_inputs(
                    reference_engine, engine.make_decoder(head).to(device), dataset, variant, device)
    cache.atomic_json(root / "benchmark.json", output)
    print("Saved", root / "benchmark.json", flush=True)


if __name__ == "__main__":
    main()
