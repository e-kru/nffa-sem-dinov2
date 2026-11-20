"""Task-specific metrics, consistently computed on validation partitions."""

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


def classification_metrics(labels: list[np.ndarray], preds: list[np.ndarray],
                           num_classes: int) -> dict[str, float]:
    y = np.concatenate(labels)
    p = np.concatenate(preds)
    return {
        "accuracy": float(accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, labels=list(range(num_classes)),
                                  average="macro", zero_division=0)),
    }


class PixelConfusion:
    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.n = num_classes
        self.ignore_index = ignore_index
        self.matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        pred = logits.argmax(dim=1).reshape(-1).cpu().long()
        true = targets.reshape(-1).cpu().long()
        valid = true != self.ignore_index
        pred, true = pred[valid], true[valid]
        if true.numel() and (true.min() < 0 or true.max() >= self.n):
            raise ValueError("Mask values must be 0..num_classes-1 or ignore_index")
        values = true * self.n + pred
        self.matrix += torch.bincount(values, minlength=self.n ** 2).reshape(self.n, self.n)

    def compute(self) -> dict[str, float]:
        cm = self.matrix.float()
        intersection = cm.diag()
        gt = cm.sum(dim=1)
        pred = cm.sum(dim=0)
        union = gt + pred - intersection
        # Classes entirely absent in validation GT are omitted from macro mIoU.
        present = gt > 0
        miou = (intersection[present] / union[present].clamp_min(1)).mean()
        total = cm.sum()
        return {
            "pixel_accuracy": float((intersection.sum() / total).item()) if total else 0.0,
            "miou": float(miou.item()) if bool(present.any()) else 0.0,
        }
