"""Global confusion and train-defined 8-connected MA island metrics."""
import numpy as np
from scipy import ndimage

THRESHOLDS = [i / 20 for i in range(1, 20)]


def confusion(probability, target, threshold):
    valid = target != 255
    truth, pred = target[valid].astype(np.int64), probability[valid] >= threshold
    return np.bincount(2 * truth + pred, minlength=4).reshape(2, 2)


def scores(cm):
    tn, fp, fn, tp = map(int, cm.ravel())
    def div(a, b):
        return a / b if b else 0.0
    ma, bg = div(tp, tp+fp+fn), div(tn, tn+fp+fn)
    return {"ma_iou": ma, "background_iou": bg, "miou": (ma+bg)/2,
            "ma_precision": div(tp, tp+fp), "ma_recall": div(tp, tp+fn),
            "confusion_matrix": cm.tolist()}


def components(target):
    labels, count = ndimage.label(target == 1, structure=np.ones((3, 3), dtype=np.uint8))
    areas = np.bincount(labels.ravel(), minlength=count+1)[1:]
    return labels, areas


def training_size_bins(targets):
    areas = []
    for target in targets:
        areas.extend(components(np.asarray(target))[1].tolist())
    if not areas:
        raise ValueError("Training masks contain no MA components")
    cuts = np.quantile(areas, [1/3, 2/3]).tolist()
    return {"source": "train_only", "connectivity": 8,
            "quantiles": [1/3, 2/3], "cuts_pixels": cuts,
            "train_component_count": len(areas),
            "assignment": "small <= cut1; medium <= cut2; large > cut2"}


def component_rows(target, prediction, bins, image_id, threshold):
    labels, areas = components(target)
    detected = np.bincount(labels[prediction].ravel(), minlength=len(areas)+1)[1:]
    lo, hi = bins["cuts_pixels"]
    return [{"image_id": image_id, "component_id": i+1, "area_pixels": int(area),
             "size_class": "small" if area <= lo else "medium" if area <= hi else "large",
             "correct_pixels": int(hit), "pixel_recall": float(hit/area),
             "detected_at_25pct": int(hit/area >= .25), "threshold": threshold}
            for i, (area, hit) in enumerate(zip(areas, detected))]


def summarize_components(rows):
    result = {}
    for name in ("small", "medium", "large"):
        part = [r for r in rows if r["size_class"] == name]
        area = sum(r["area_pixels"] for r in part)
        result[name] = {"gt_islands": len(part),
                        "mean_island_pixel_recall": float(np.mean([r["pixel_recall"] for r in part])) if part else None,
                        "pixel_weighted_recall": sum(r["correct_pixels"] for r in part)/area if area else None,
                        "fraction_detected_at_25pct": float(np.mean([r["detected_at_25pct"] for r in part])) if part else None}
    return result
