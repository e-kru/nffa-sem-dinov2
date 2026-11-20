"""Shared supervised trainer for all four task/head combinations."""

import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .metrics import PixelConfusion, classification_metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _logits(model, x, *, cached: bool):
    return model(x)  # cached: model is the head; images: model is DownstreamModel


def _valid_count(task: str, y: torch.Tensor, ignore_index: int) -> int:
    return int(y.numel()) if task == "classification" else int((y != ignore_index).sum())


@torch.no_grad()
def evaluate(model, loader, device, task, num_classes, ignore_index=255, cached=False):
    model.eval()
    loss_sum = 0.0
    count = 0
    cm = PixelConfusion(num_classes, ignore_index) if task == "segmentation" else None
    ys, ps = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        valid = _valid_count(task, y, ignore_index)
        if valid == 0:
            continue
        logits = _logits(model, x, cached=cached)
        criterion = nn.functional.cross_entropy(logits, y, ignore_index=ignore_index,
                                                reduction="sum")
        loss_sum += criterion.item()
        count += valid
        if task == "segmentation":
            cm.update(logits, y)
        else:
            ys.append(y.cpu().numpy())
            ps.append(logits.argmax(dim=1).cpu().numpy())
    if count == 0:
        raise ValueError("No valid evaluation labels")
    result = {"loss": loss_sum / count}
    result.update(cm.compute() if cm is not None else classification_metrics(ys, ps, num_classes))
    return result


def train(model, train_data, val_data, *, output: Path, task: str,
          num_classes: int, epochs: int = 100, patience: int = 15,
          batch_size: int = 256, lr: float = 1e-3,
          encoder_lr: float = 1e-5, weight_decay: float = 1e-4,
          ignore_index: int = 255, seed: int = 42,
          workers: int = 0, cached: bool = False,
          train_encoder: bool = False, config: dict | None = None) -> dict:
    if epochs <= 0 or patience <= 0 or batch_size <= 1:
        raise ValueError("Invalid epochs/patience/batch size")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    params = [{"params": model.parameters(), "lr": lr}]
    if not cached and train_encoder:
        params = [
            {"params": model.head.parameters(), "lr": lr},
            {"params": model.backbone.parameters(), "lr": encoder_lr},
        ]
    optimizer = torch.optim.AdamW(params, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, generator=generator,
        num_workers=workers, pin_memory=(device.type == "cuda"),
        drop_last=(task == "classification" and len(train_data) % batch_size == 1),
    )
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                            num_workers=workers, pin_memory=(device.type == "cuda"))
    metric_key = "macro_f1" if task == "classification" else "miou"
    best_score, best_loss, bad_epochs = -1.0, float("inf"), 0
    history = []
    start = time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum, count = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            valid = _valid_count(task, y, ignore_index)
            if valid == 0:
                continue
            optimizer.zero_grad(set_to_none=True)
            logits = _logits(model, x, cached=cached)
            loss = nn.functional.cross_entropy(logits, y, ignore_index=ignore_index)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * valid
            count += valid
        if not count:
            raise ValueError("No valid training labels")
        val = evaluate(model, val_loader, device, task, num_classes, ignore_index, cached)
        row = {"epoch": epoch, "train_loss": loss_sum / count,
               **{f"val_{key}": value for key, value in val.items()}}
        history.append(row)
        score = val[metric_key]
        improved = (score > best_score + 1e-8 or
                    (abs(score - best_score) <= 1e-8 and val["loss"] < best_loss - 1e-8))
        if improved:
            best_score, best_loss, bad_epochs, best_epoch = score, val["loss"], 0, epoch
            # Saves full model incl. encoder when using image mode, only head otherwise.
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                        "task": task, "metric": metric_key, "best_score": best_score,
                        "config": config or {}}, output / "best_model.pt")
        else:
            bad_epochs += 1
        print(f"epoch {epoch:03d} | train CE={row['train_loss']:.4f} | "
              f"val CE={val['loss']:.4f} {metric_key}={score:.4f}"
              + (" *" if improved else ""), flush=True)
        if bad_epochs >= patience:
            break
    with (output / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    summary = {"task": task, "metric": metric_key, "best_val_score": best_score,
               "best_val_loss": best_loss, "best_epoch": best_epoch,
               "epochs_run": len(history), "duration_s": round(time.monotonic()-start, 2),
               "test_evaluated": False, "config": config or {}}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
