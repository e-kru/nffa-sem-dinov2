"""Original-image training budgets, original-raster CE and common evaluation."""
import csv
import json
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from src.downstream.engine import set_seed
from src.downstream.heads import DeepCNN, make_head
from .cache import atomic_json, atomic_torch
from .devices import CudaMemoryProfiler, normalize_device
from .geometry import content_box, project_logits, reconstruct, regions
from .loading import make_loader, pack_originals, slice_batch
from .timing import PhaseTimer
from .metrics import (THRESHOLDS, component_rows, confusion, scores,
                      summarize_components)


def save_csv(path, rows, fields=None):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".partial")
    with temp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or (list(rows[0]) if rows else []))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def make_decoder(kind, feature_dim=384):
    return DeepCNN(feature_dim, 8) if kind == "cnn8" else make_head("segmentation", "linear", feature_dim, 2)


def identity_collate(samples):
    return samples


def original_loss(grid_logits, target, variant):
    """Sum valid-pixel CE across all quadrants / valid pixels in this ORIGINAL.

    This weights odd-sized quadrants by valid area; all-ignore crops contribute 0.
    All originals carry equal weight in the effective optimization batch.
    """
    height, width = target.shape
    if len(grid_logits) != (4 if variant.quadrants else 1):
        raise ValueError("Wrong number of crop logits for original loss")
    loss = grid_logits.sum() * 0
    for z, (x0, y0, x1, y1) in zip(grid_logits, regions(width, height, variant.quadrants)):
        labels = target[y0:y1, x0:x1]
        logits = project_logits(z, (x1-x0, y1-y0), variant)
        loss = loss + F.cross_entropy(logits[None], labels[None], ignore_index=255, reduction="sum")
    return loss / (target != 255).sum().clamp_min(1)


def forward_batch(model, samples, device):
    device = normalize_device(device)
    batch = samples if isinstance(samples, dict) else pack_originals(samples)
    features = batch["features"]
    b, q = features.shape[:2]
    features = features.flatten(0, 1).to(device, non_blocking=True)
    return model(features).reshape(b, q, 2, *features.shape[-2:])


def project_batch(grid_logits, targets, variant):
    """Vectorize original-raster interpolation; retain ragged/odd geometry."""
    if grid_logits.shape[1] != (4 if variant.quadrants else 1):
        raise ValueError("Wrong number of crops")
    groups = {}
    for i, target in enumerate(targets):
        groups.setdefault(tuple(target.shape), []).append(i)
    result = [None] * len(targets)
    for (h, w), indices in groups.items():
        z = grid_logits if len(indices) == len(targets) else grid_logits[indices]
        parts = []
        for q, (x0, y0, x1, y1) in enumerate(regions(w, h, variant.quadrants)):
            width, height = x1-x0, y1-y0
            inner = content_box(width, height, variant)
            part = z[:, q]
            if inner != (0, 0, variant.width, variant.height):
                part = F.interpolate(part, (variant.height, variant.width), mode="bilinear", align_corners=False)
                a, b, c, d = inner
                part = part[:, :, b:d, a:c]
            parts.append(F.interpolate(part, (height, width), mode="bilinear", align_corners=False))
        full = parts[0] if len(parts) == 1 else torch.cat((
            torch.cat(parts[:2], dim=3), torch.cat(parts[2:], dim=3)), dim=2)
        if len(indices) == len(targets):
            return full
        for i, value in zip(indices, full):
            result[i] = value
    return result


def full_resolution_losses(full, targets, device):
    """Exactly the mean valid-pixel CE per ORIGINAL, then average originals."""
    if isinstance(full, torch.Tensor):
        labels = targets if isinstance(targets, torch.Tensor) else torch.stack(list(targets))
        labels = labels.to(device, non_blocking=True)
        pixel_losses = F.cross_entropy(full, labels, ignore_index=255, reduction="none")
        return pixel_losses.flatten(1).sum(1) / (labels != 255).flatten(1).sum(1).clamp_min(1)
    return torch.stack([F.cross_entropy(z[None], y[None].to(device, non_blocking=True),
                                       ignore_index=255, reduction="sum") / (y != 255).sum().clamp_min(1).to(device)
                        for z, y in zip(full, targets)])


def _train_effective_batch(model, batch, optimizer, variant, device, micro_size):
    total = torch.zeros((), device=device)
    count = len(batch["targets"])
    for start in range(0, count, micro_size):
        micro = slice_batch(batch, start, start+micro_size)
        logits = forward_batch(model, micro, device)
        full = project_batch(logits, micro["targets"], variant)
        losses = full_resolution_losses(full, micro["targets"], device)
        (losses.sum()/count).backward()
        total += losses.detach().sum()
    return total


def train_epoch(model, dataset, optimizer, variant, device, batch_size, micro_batch_size,
                seed, epoch, workers=0, *, loader=None):
    """One optimizer step per effective original batch; retry CUDA OOM safely."""
    device = normalize_device(device)
    loader = loader if loader is not None else make_loader(dataset, batch_size, workers, device, True, seed)
    loader.sampler.epoch = epoch
    timer = PhaseTimer(device)
    model.train()
    total_loss = torch.zeros((), device=device)
    seen, steps, retries = 0, 0, 0
    actual_micro = min(batch_size, micro_batch_size)
    for batch in loader:
        timer.loaded()
        before = rng_state(device)
        while True:
            optimizer.zero_grad(set_to_none=True)
            try:
                batch_loss = _train_effective_batch(model, batch, optimizer, variant, device, actual_micro)
                break
            except torch.cuda.OutOfMemoryError:
                if actual_micro == 1:
                    raise
                actual_micro = max(1, actual_micro//2)
                retries += 1
            # Leave the exception scope before releasing CUDA's cached blocks.
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            restore_rng(before, device)
            print(f"CUDA OOM: retrying unchanged effective batch with microbatch={actual_micro}", flush=True)
        optimizer.step()
        total_loss += batch_loss
        seen += len(batch["targets"])
        steps += 1
        timer.computed()
    timings = timer.finish()
    return {"train_loss": float(total_loss.cpu())/seen, "optimizer_steps": steps,
            "originals_seen": seen, "actual_micro_batch_size": actual_micro, "oom_retries": retries,
            **timings}


def sync(device):
    device = normalize_device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device.index)


@torch.inference_mode()
def evaluate(model, dataset, variant, device, thresholds=(.5,), bins=None, collect=False,
             workers=0, *, batch_size=8, loader=None):
    """Batched original-raster evaluation; no per-image device synchronization."""
    device = normalize_device(device)
    model.eval()
    loader = loader if loader is not None else make_loader(dataset, batch_size, workers, device)
    timer = PhaseTimer(device)
    cms = torch.zeros((len(thresholds), 2, 2), device=device, dtype=torch.int64)
    loss_sum = torch.zeros((), device=device)
    finite = torch.ones((), dtype=torch.bool, device=device)
    component_results, per_image = [], []
    for batch in loader:
        timer.loaded()
        grid = forward_batch(model, batch, device)
        full = project_batch(grid, batch["targets"], variant)
        timer.decoder_done()
        loss_sum += full_resolution_losses(full, batch["targets"], device).sum()
        if isinstance(full, torch.Tensor):
            probability = torch.sigmoid(full[:, 1]-full[:, 0])
            labels = batch["targets"]
            labels = (labels if isinstance(labels, torch.Tensor) else torch.stack(labels)).to(device, non_blocking=True)
            pairs = [(probability, labels)]
        else:
            probability = [torch.sigmoid(z[1]-z[0]) for z in full]
            pairs = [(p, y.to(device, non_blocking=True)) for p, y in zip(probability, batch["targets"])]
        for p, truth in pairs:
            finite &= torch.isfinite(p).all()
            ma, bg = truth == 1, truth == 0
            for j, threshold in enumerate(thresholds):
                pred = p >= threshold
                cms[j, 0, 0] += (bg & ~pred).sum()
                cms[j, 0, 1] += (bg & pred).sum()
                cms[j, 1, 0] += (ma & ~pred).sum()
                cms[j, 1, 1] += (ma & pred).sum()
        if collect:
            probabilities = probability.cpu().numpy() if isinstance(probability, torch.Tensor) else [p.cpu().numpy() for p in probability]
            for p, truth, record in zip(probabilities, batch["targets"], batch["records"]):
                truth = truth.numpy()
                if not np.isfinite(p).all():
                    raise ValueError("Nonfinite predictions")
                for threshold in thresholds:
                    row = {"image_id": record["image_id"], "threshold": threshold, **scores(confusion(p, truth, threshold))}
                    row["confusion_matrix"] = json.dumps(row["confusion_matrix"])
                    per_image.append(row)
                    component_results.extend(component_rows(truth, p >= threshold, bins, record["image_id"], threshold))
        timer.computed()
    timings = timer.finish()
    loss = float(loss_sum.cpu())/len(dataset)
    if not np.isfinite(loss) or not bool(finite.cpu()):
        raise ValueError("Nonfinite predictions/loss")
    return {"loss": loss, "thresholds": [{"threshold": t, **scores(cm)} for t, cm in zip(thresholds, cms.cpu().numpy())],
            "cached_decoder_seconds_per_original": timings["decoder_seconds"]/len(dataset) if timings["decoder_seconds"] is not None else None,
            "components": component_results, "per_image": per_image, **timings}


def rng_state(device="cpu"):
    device = normalize_device(device)
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda_device": str(device),
            "cuda": torch.cuda.get_rng_state(device.index) if device.type == "cuda" else None}


def restore_rng(state, device="cpu"):
    device = normalize_device(device)
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if device.type == "cuda" and state["cuda"] is not None:
        value = state["cuda"]
        if isinstance(value, list):  # Older checkpoints saved every visible GPU.
            value = value[device.index]
        torch.cuda.set_rng_state(value.cpu(), device.index)


@contextmanager
def deterministic_execution(policy):
    """Scope process-wide Torch flags, including restoration after exceptions.

    CUDA segmentation kernels (e.g. bilinear backward or NLL reductions) may
    lack deterministic implementations. ``off`` disables the slow deterministic
    replacements while callers still seed all RNGs. ``warn`` requests deterministic kernels
    where supported and reports the remaining operations; ``strict`` fails.
    Neither policy changes the objective or silently retries a failed run.
    """
    if policy not in ("off", "strict", "warn"):
        raise ValueError(f"Unknown determinism policy: {policy}")
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    benchmark = torch.backends.cudnn.benchmark
    cudnn_deterministic = torch.backends.cudnn.deterministic
    try:
        torch.use_deterministic_algorithms(policy != "off", warn_only=policy == "warn")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = policy != "off"
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cudnn.deterministic = cudnn_deterministic


def train_run(output, config, train_ds, val_ds, variant, device, resume=False):
    # Preserve strict semantics for callers with an older explicit configuration.
    with deterministic_execution(config.get("determinism_policy", "strict")):
        return _train_run(output, config, train_ds, val_ds, variant, device, resume)


def _train_run(output, config, train_ds, val_ds, variant, device, resume=False):
    device = normalize_device(device)
    output = Path(output)
    config_path = output / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError(f"Run configuration mismatch: {output}")
        if (output / "training_complete.json").exists():
            print(f"Training complete, skipping: {output}", flush=True)
            return
        if not resume:
            raise FileExistsError(f"Interrupted run, use --resume: {output}")
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Unrecognized output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(config_path, config)
    atomic_json(output / "cache_metadata.json", config["cache_metadata"])
    set_seed(config["seed"])
    with (output / "run.log").open("a") as log:
        log.write(f"determinism_policy={config.get('determinism_policy', 'strict')}\n")
    model = make_decoder(config["head"], config.get("cache_metadata", {}).get("contract", {}).get("feature_shape", [1, 384])[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    history, best_score, best_loss, best_epoch, bad_epochs, first = [], -1., float("inf"), 0, 0, 1
    latest = output / "last_state.pt"
    if latest.exists():
        # Only load this pipeline's own local optimizer/RNG checkpoint (contains numpy state).
        state = torch.load(latest, map_location="cpu", weights_only=False)
        if state["config"] != config:
            raise ValueError("Resume state/config mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        history, best_score, best_loss, best_epoch, bad_epochs = (
            state[k] for k in ("history", "best_score", "best_loss", "best_epoch", "bad_epochs"))
        first = state["epoch"] + 1
        restore_rng(state["rng"], device)
        # Repair artifacts after interruption between best and epoch checkpoint writes.
        atomic_torch(output / "best_model.pt", state["best_checkpoint"])
        existing_history = output / "history.csv"
        if existing_history.exists():
            with existing_history.open() as handle:
                measured = {int(row["epoch"]): row for row in csv.DictReader(handle)}
            for row in history:
                for key, value in measured.get(row["epoch"], {}).items():
                    if key.endswith("seconds"):
                        row[key] = float(value) if value else None
        save_csv(output / "history.csv", history)
    train_loader = make_loader(train_ds, config["batch_size"], config["workers"], device, True, config["seed"])
    val_loader = make_loader(val_ds, config.get("validation_batch_size", config["batch_size"]), config["workers"], device)
    memory = CudaMemoryProfiler(device)
    memory.start()
    runtime_micro = min(config["micro_batch_size"], history[-1].get("actual_micro_batch_size", config["micro_batch_size"])) if history else config["micro_batch_size"]
    for epoch in range(first, config["epochs"]+1):
        if bad_epochs >= config["patience"]:
            break
        epoch_started = time.perf_counter()
        row = train_epoch(model, train_ds, optimizer, variant, device, config["batch_size"],
                          runtime_micro, config["seed"], epoch, config["workers"], loader=train_loader)
        runtime_micro = row["actual_micro_batch_size"]
        val = evaluate(model, val_ds, variant, device, workers=config["workers"], loader=val_loader)
        row.update(train_seconds=row["phase_seconds"], validation_seconds=val["phase_seconds"],
                   validation_data_loading_seconds=val["data_loading_seconds"],
                   validation_gpu_computation_seconds=val["gpu_computation_seconds"],
                   epoch_seconds=time.perf_counter()-epoch_started)
        score = val["thresholds"][0]["miou"]
        history.append({"epoch": epoch, **row, "val_loss": val["loss"], "val_miou_05": score,
                        "val_ma_iou_05": val["thresholds"][0]["ma_iou"]})
        checkpoint_started = time.perf_counter()
        improved = score > best_score+1e-8 or (abs(score-best_score) <= 1e-8 and val["loss"] < best_loss-1e-8)
        if improved:
            best_score, best_loss, best_epoch, bad_epochs = score, val["loss"], epoch, 0
            checkpoint = {"model_state_dict": model.state_dict(), "epoch": epoch,
                          "best_score": best_score, "selection": "val mIoU at threshold 0.5", "config": config}
            atomic_torch(output / "best_model.pt", checkpoint)
        else:
            bad_epochs += 1
        history[-1]["checkpoint_seconds"] = 0.
        # Retain a best snapshot inside the atomic epoch checkpoint for crash consistency.
        best_checkpoint = torch.load(output / "best_model.pt", map_location="cpu", weights_only=True)
        atomic_torch(latest, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                             "rng": rng_state(device), "epoch": epoch, "history": history,
                             "best_score": best_score, "best_loss": best_loss,
                             "best_epoch": best_epoch, "bad_epochs": bad_epochs,
                             "best_checkpoint": best_checkpoint, "config": config})
        history[-1]["checkpoint_seconds"] = time.perf_counter()-checkpoint_started
        row["epoch_seconds"] = time.perf_counter()-epoch_started
        history[-1]["epoch_seconds"] = row["epoch_seconds"]
        save_csv(output / "history.csv", history)
        line = (f"epoch={epoch} train_CE={row['train_loss']:.6f} val_mIoU={score:.6f} steps={row['optimizer_steps']} "
                f"epoch_s={row['epoch_seconds']:.3f} data_s={row['data_loading_seconds']:.3f} "
                f"gpu_s={row['gpu_computation_seconds']} val_s={row['validation_seconds']:.3f} "
                f"micro={row['actual_micro_batch_size']}")
        print(line, flush=True)
        with (output / "run.log").open("a") as log:
            log.write(line+"\n")
    atomic_json(output / "training_complete.json", {"best_epoch": best_epoch, "epochs_run": len(history),
                "train_peak_gpu_bytes": memory.peak_bytes(),
                "train_memory_profiling": memory.metadata()})


def evaluate_run(output, config, dataset, variant, device, bins):
    with deterministic_execution(config.get("determinism_policy", "strict")):
        return _evaluate_run(output, config, dataset, variant, device, bins)


def _evaluate_run(output, config, dataset, variant, device, bins):
    device = normalize_device(device)
    output = Path(output)
    if json.loads((output / "config.json").read_text()) != config:
        raise ValueError("Evaluation/run configuration mismatch")
    if not (output / "training_complete.json").exists():
        raise ValueError("Training incomplete, resume training first")
    if (output / "validation_metrics.json").exists():
        print(f"Evaluation complete, skipping: {output}", flush=True)
        return
    checkpoint = torch.load(output / "best_model.pt", map_location="cpu", weights_only=True)
    model = make_decoder(config["head"], config.get("cache_metadata", {}).get("contract", {}).get("feature_shape", [1, 384])[1]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    memory = CudaMemoryProfiler(device)
    memory.start()
    val_loader = make_loader(dataset, config.get("validation_batch_size", config["batch_size"]), config["workers"], device)
    result = evaluate(model, dataset, variant, device, THRESHOLDS, workers=config["workers"], loader=val_loader)
    best = max(result["thresholds"], key=lambda r: (r["ma_iou"], -abs(r["threshold"]-.5)))
    fixed = next(r for r in result["thresholds"] if r["threshold"] == .5)
    operating = list(dict.fromkeys([.5, best["threshold"]]))
    detail = evaluate(model, dataset, variant, device, operating, bins, collect=True, workers=config["workers"], loader=val_loader)
    save_csv(output / "threshold_metrics.csv", result["thresholds"])
    save_csv(output / "component_metrics.csv", detail["components"], fields=[
        "image_id", "component_id", "area_pixels", "size_class", "correct_pixels",
        "pixel_recall", "detected_at_25pct", "threshold"])
    save_csv(output / "per_image_metrics.csv", detail["per_image"])
    atomic_json(output / "component_size_bins.json", bins)
    training = json.loads((output / "training_complete.json").read_text())
    report = {"variant": variant.name, "head": config["head"], "seed": config["seed"],
              "best_epoch": checkpoint["epoch"], "validation_at_0_50": fixed,
              "validation_at_best_threshold": best, "threshold_selection": "validation MA-IoU; ties nearest 0.5",
              "split": "val", "test_evaluated": False, "samples": len(dataset),
              "patch_tokens_per_original": variant.tokens,
              "nonscientific_smoke_test": config["nonscientific_smoke_test"],
              "cached_decoder_seconds_per_original": result["cached_decoder_seconds_per_original"],
              "timing_scope": "batched cached features H2D + decoder + original logit reconstruction; excludes encoder, loss, metrics and disk IO",
              "cached_decoder_batch_size": config.get("validation_batch_size", config["batch_size"]),
              "end_to_end_seconds_per_original": None,
              "eval_peak_gpu_bytes": memory.peak_bytes(),
              "eval_memory_profiling": memory.metadata(),
              **training,
              "components": {str(t): summarize_components([r for r in detail["components"] if r["threshold"] == t])
                             for t in operating}}
    # Completion marker is published only after every detailed artifact.
    atomic_json(output / "validation_metrics.json", report)
    with (output / "run.log").open("a") as log:
        log.write(f"validation complete; threshold={best['threshold']}; test_evaluated=False\n")


@torch.inference_mode()
def benchmark_end_to_end(model, encoder, dataset, variant, device, count):
    """Original RGB in RAM -> normalized tensors -> encoder -> original logits."""
    from .data import load_original
    from .geometry import prepare_region
    device = normalize_device(device)
    model.eval()
    times = []
    for i in range(min(count, len(dataset))):
        sample = dataset.samples[i]
        image, _ = load_original(dataset.manifest, sample["record"])
        boxes = regions(*image.size, variant.quadrants)
        def predict():
            x = torch.stack([prepare_region(image, b, variant) for b in boxes]).to(device)
            z = model(encoder.forward_dense(x))
            return reconstruct(z, image.size, variant)
        if i == 0:
            predict()
            sync(device)
        sync(device)
        start = time.perf_counter()
        predict()
        sync(device)
        times.append(time.perf_counter()-start)
    return {"end_to_end_seconds_per_original": float(np.mean(times)),
            "end_to_end_timing_images": len(times),
            "end_to_end_timing_scope": "original RGB in RAM; preprocessing + H2D + frozen encoder + decoder + reconstruction; excludes disk IO"}
