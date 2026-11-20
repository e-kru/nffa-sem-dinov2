"""Visualise frozen-DINOv2 Aachen-Heerlen segmentation probe on VAL ONLY.

Place under scripts/visualize_aachen_predictions.py in the nffa-sem-dinov2 repo.
Uses cached spatial features and the existing, best trained segmentation head;
never reruns DINOv2 and never accesses test features, test labels, or test metrics.

Run from repo root:
  python -m scripts.visualize_aachen_predictions --n 18 --seed 42
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from scripts.train_aachen_cached_segmentation import (
    CachedSegmentationProbe,
    DenseCachedSegmentationDataset,
    sha256_file,
)
from src.downstream.data import load_rows

HEIGHT, WIDTH = 336, 448
COLORS = {
    "gt": (64, 220, 127),     # green: ground-truth polygons
    "pred": (39, 187, 246),   # blue: model-predicted MA regions
    "tp": (64, 220, 127),     # green: correct positive
    "fp": (255, 186, 65),     # amber: predicted MA but unlabelled
    "fn": (243, 89, 93),      # red: annotated MA missed
    "ignore": (170, 170, 170),
}


def load_font(size):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def resolve(manifest, path_string):
    p = Path(path_string)
    return p if p.is_absolute() else manifest.parent / p


def blend_overlay(base, mask, color, alpha=0.46):
    img = np.array(base, dtype=np.uint8, copy=True)
    area = np.asarray(mask, dtype=bool)
    if area.any():
        img[area] = np.clip(
            (1.0 - alpha) * img[area].astype(np.float32)
            + alpha * np.array(color, dtype=np.float32), 0, 255
        ).astype(np.uint8)
    return Image.fromarray(img, mode="RGB")


def error_overlay(base, truth, pred):
    # Underlying SEM remains visible; colored pixels show TP/FP/FN.
    result = base
    result = blend_overlay(result, (truth == 1) & (pred == 1), COLORS["tp"], 0.75)
    result = blend_overlay(result, (truth == 0) & (pred == 1), COLORS["fp"], 0.75)
    result = blend_overlay(result, (truth == 1) & (pred == 0), COLORS["fn"], 0.75)
    return result


def per_image_stats(truth, pred):
    valid = truth != 255
    tp = int(np.count_nonzero((truth == 1) & (pred == 1) & valid))
    fp = int(np.count_nonzero((truth == 0) & (pred == 1) & valid))
    fn = int(np.count_nonzero((truth == 1) & (pred == 0) & valid))
    tn = int(np.count_nonzero((truth == 0) & (pred == 0) & valid))
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "iou_ma": tp / (tp + fp + fn) if (tp + fp + fn) else float("nan"),
        "precision_ma": tp / (tp + fp) if (tp + fp) else float("nan"),
        "recall_ma": tp / (tp + fn) if (tp + fn) else float("nan"),
        "gt_ma_percent": 100.0 * (tp + fn) / max(1, tp + fp + fn + tn),
        "pred_ma_percent": 100.0 * (tp + fp) / max(1, tp + fp + fn + tn),
    }


def draw_panel(image_id, image, truth, pred, stats):
    background = (19, 25, 33)
    gap, margin, title_h, bottom_h = 12, 14, 66, 22
    w = 4 * WIDTH + 3 * gap + 2 * margin
    h = HEIGHT + title_h + bottom_h
    output = Image.new("RGB", (w, h), background)
    draw = ImageDraw.Draw(output)
    font = load_font(21)
    small_font = load_font(17)
    panels = [
        image,
        blend_overlay(image, truth == 1, COLORS["gt"]),
        blend_overlay(image, pred == 1, COLORS["pred"]),
        error_overlay(image, truth, pred),
    ]
    titles = ["SEM-Bild", "Expertenmaske (gruen)", "Modellvorhersage (blau)", "Fehler: TP gruen / FP gelb / FN rot"]
    iou = stats["iou_ma"]
    iou_str = f"{iou:.3f}" if np.isfinite(iou) else "n/a"
    draw.text((margin, 8), f"{image_id}   |   MA-IoU: {iou_str}   |   Recall: {stats['recall_ma']:.1%}",
              fill=(238, 240, 244), font=font)
    for j, (panel, title) in enumerate(zip(panels, titles)):
        x = margin + j * (WIDTH + gap)
        draw.text((x, 42), title, fill=(198, 209, 224), font=small_font)
        output.paste(panel, (x, title_h))
    draw.text((margin, title_h + HEIGHT + 4),
              "Gleiche 448x336-Bildgeometrie wie beim Training. Gelb kann auch unannotierte MA-Inseln enthalten.",
              fill=(165, 178, 193), font=small_font)
    return output


def overview(images, out_dir, per_page=6):
    scale = 0.62
    for page_idx, start in enumerate(range(0, len(images), per_page), start=1):
        group = images[start:start + per_page]
        previews = [im.resize((round(im.width * scale), round(im.height * scale)), Image.Resampling.LANCZOS)
                    for im in group]
        pad, gap = 16, 8
        w = max(x.width for x in previews) + 2 * pad
        h = sum(x.height for x in previews) + gap * (len(previews) - 1) + 2 * pad + 36
        sheet = Image.new("RGB", (w, h), (19, 25, 33))
        draw = ImageDraw.Draw(sheet)
        draw.text((pad, 12), f"Aachen-Heerlen: zufaellige Validation-Beispiele | Blatt {page_idx}",
                  fill="white", font=load_font(22))
        y = pad + 36
        for im in previews:
            sheet.paste(im, (pad, y))
            y += im.height + gap
        dest = out_dir / f"overview_{page_idx:02}.png"
        sheet.save(dest)
        print("Overview:", dest, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path,
                   default=Path("outputs/aachen_heerlen/features/vits14_448x336_float32"))
    p.add_argument("--manifest", type=Path,
                   default=Path("outputs/aachen_heerlen/splits/group_split_seed42.csv"))
    p.add_argument("--checkpoint", type=Path,
                   default=Path("outputs/aachen_heerlen/runs/linear_vits14_448x336/best_model.pt"))
    p.add_argument("--output", type=Path,
                   default=Path("outputs/aachen_heerlen/visualizations/linear_val_seed42"))
    p.add_argument("--n", type=int, default=18, help="Random examples from validation set")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    args = p.parse_args()

    cache_dir, manifest, checkpoint_path = (
        args.cache_dir.resolve(), args.manifest.resolve(), args.checkpoint.resolve()
    )
    meta = json.loads((cache_dir / "cache_metadata.json").read_text(encoding="utf-8"))
    manifest_hash = sha256_file(manifest)
    if meta.get("manifest_sha256") != manifest_hash:
        raise ValueError("Manifest SHA256 differs from cached features; aborting")
    if meta.get("input_shape_per_image") != [3, HEIGHT, WIDTH]:
        raise ValueError("Cache input geometry does not match 448x336")
    if meta.get("limit_per_split") is not None:
        raise ValueError("Partial feature cache is not suitable for this visualization")
    if meta.get("encoder") != "dinov2_vits14":
        raise ValueError("Unexpected feature encoder")
    if meta.get("preprocessing_file_sha256") != sha256_file(Path("src/preprocessing_v2.py")):
        raise ValueError("Preprocessing source changed after caching. Verify geometry first.")

    rows = load_rows(manifest)
    ds = DenseCachedSegmentationDataset(cache_dir, manifest, "val", rows)
    if not 1 <= args.n <= len(ds):
        p.error(f"--n must lie between 1 and {len(ds)}")
    image_rows = {r["image_id"]: r for r in rows if r["split"] == "val"}

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = state.get("config", {})
    if config.get("feature_manifest_sha256") != manifest_hash:
        raise ValueError("Checkpoint was trained with a different feature manifest")
    if config.get("limit") is not None:
        raise ValueError("Checkpoint comes from a partial smoke test")
    head_kind = config.get("head")
    if head_kind not in ("linear", "mlp"):
        raise ValueError(f"Unexpected head architecture: {head_kind}")
    if config.get("output_shape") != [2, HEIGHT, WIDTH]:
        raise ValueError("Checkpoint output geometry mismatch")
    model = CachedSegmentationProbe(int(ds.features.shape[1]), head_kind)
    model.load_state_dict(state["model_state_dict"], strict=True)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA selected but not available")
    model = model.to(device).eval()

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    chosen = sorted(int(i) for i in rng.choice(len(ds), size=args.n, replace=False))
    print(f"Validation only: {len(ds)} images. Random subset: {len(chosen)}. Device={device}.", flush=True)
    written, records = [], []
    with torch.inference_mode():
        for idx in chosen:
            row = ds.rows[idx]
            image_id = row["image_id"]
            feature, labels = ds[idx]
            prediction = model(feature.unsqueeze(0).to(device)).argmax(dim=1)[0].cpu().numpy()
            truth = labels.numpy()
            if prediction.shape != (HEIGHT, WIDTH) or truth.shape != (HEIGHT, WIDTH):
                raise ValueError(f"Size mismatch for {image_id}")
            path = resolve(manifest, image_rows[image_id]["image_path"])
            with Image.open(path) as original:
                img = TF.resize(original.convert("RGB"), [HEIGHT, WIDTH],
                                interpolation=InterpolationMode.BILINEAR)
            metrics = per_image_stats(truth, prediction)
            panel = draw_panel(image_id, img, truth, prediction, metrics)
            outfile = out / f"comparison_{Path(image_id).stem}.png"
            panel.save(outfile)
            written.append(panel)
            records.append({"image_id": image_id, "cache_index": row["cache_index"],
                            **metrics})
            print(f"{image_id}: MA-IoU={metrics['iou_ma']:.3f}", flush=True)

    with (out / "selected_val_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    overview(written, out)
    (out / "README.txt").write_text(
        "Validation-only visualization: random images chosen with seed " + str(args.seed)
        + ".\nExpert annotations = green, model predictions = blue.\n"
        + "Error panel: green=true positive, yellow=false positive, red=false negative.\n"
        + "Unannotated true MA structures may be counted as false positives.\n"
        + "All images resized to 448x336 without cropping. Test set untouched.\n",
        encoding="utf-8"
    )
    print(f"\nDONE: {len(written)} comparisons + {((len(written)-1)//6+1)} overview sheets in {out}")
    print("Results in selected_val_metrics.csv are only for the sampled images, not the full validation set.")
    print("Test split not read or evaluated.")


if __name__ == "__main__":
    main()
