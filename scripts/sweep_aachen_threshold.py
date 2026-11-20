"""Threshold-Sweep für Aachen-Heerlen, ausschließlich auf Validation."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.train_aachen_cached_segmentation import (
    CachedSegmentationProbe,
    DenseCachedSegmentationDataset,
    sha256_file,
)
from src.downstream.data import load_rows


HEIGHT, WIDTH = 336, 448


def div(a, b):
    return float(a) / float(b) if b else 0.0


def score_counts(tn, fp, fn, tp):
    iou_ma = div(tp, tp + fp + fn)
    iou_background = div(tn, tn + fp + fn)

    return {
        "iou_ma": iou_ma,
        "iou_background": iou_background,
        "miou": (iou_ma + iou_background) / 2.0,
        "ma_precision": div(tp, tp + fp),
        "ma_recall": div(tp, tp + fn),
        "predicted_ma_fraction": div(
            tp + fp, tn + fp + fn + tp
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def counts_from_buckets(hist, thresholds):
    """Konfusionsmatrix für jeden Threshold aus einem Histogramm."""

    m = len(thresholds)
    assert hist.shape == (2, m + 1)

    suffix = np.cumsum(
        hist[:, ::-1], axis=1
    )[:, ::-1]

    total_bg = int(hist[0].sum())
    total_ma = int(hist[1].sum())

    result = []

    for i, threshold in enumerate(thresholds):
        fp = int(suffix[0, i + 1])
        tp = int(suffix[1, i + 1])
        tn = total_bg - fp
        fn = total_ma - tp

        result.append({
            "threshold": float(threshold),
            **score_counts(tn, fp, fn, tp),
        })

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(
            "outputs/aachen_heerlen/features/"
            "vits14_448x336_float32"
        ),
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "outputs/aachen_heerlen/splits/"
            "group_split_seed42.csv"
        ),
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "outputs/aachen_heerlen/runs/"
            "linear_vits14_448x336/best_model.pt"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/aachen_heerlen/threshold_sweep/"
            "linear_vits14_val"
        ),
    )

    parser.add_argument(
        "--batch-size", type=int, default=8
    )

    parser.add_argument(
        "--num-workers", type=int, default=0
    )

    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )

    args = parser.parse_args()

    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("Invalid batch-size or num-workers")

    # Thresholds: 0.05, 0.10, ..., 0.95
    thresholds = np.round(
        np.arange(1, 20) / 20, 2
    )

    manifest = args.manifest.resolve()
    cache_dir = args.cache_dir.resolve()
    checkpoint_path = args.checkpoint.resolve()

    # ---------------------------------------------
    # 1. Cache und Manifest validieren
    # ---------------------------------------------

    cache_meta = json.loads(
        (cache_dir / "cache_metadata.json").read_text(
            encoding="utf-8"
        )
    )

    manifest_hash = sha256_file(manifest)

    if manifest_hash != cache_meta.get("manifest_sha256"):
        raise RuntimeError(
            "Manifest differs from feature cache"
        )

    if cache_meta.get("limit_per_split") is not None:
        raise RuntimeError(
            "Cannot evaluate a partial smoke-test cache"
        )

    if cache_meta.get("encoder") != "dinov2_vits14":
        raise RuntimeError("Unexpected encoder")

    if cache_meta.get("input_shape_per_image") != [
        3, HEIGHT, WIDTH
    ]:
        raise RuntimeError("Wrong input geometry")

    if cache_meta.get(
        "preprocessing_file_sha256"
    ) != sha256_file(Path("src/preprocessing_v2.py")):
        raise RuntimeError(
            "Preprocessing changed since feature extraction"
        )

    # ---------------------------------------------
    # 2. Nur Validation-Datensatz laden
    # ---------------------------------------------

    ds = DenseCachedSegmentationDataset(
        cache_dir,
        manifest,
        "val",
        load_rows(manifest),
    )

    # ---------------------------------------------
    # 3. Bestes trainiertes Modell laden
    # ---------------------------------------------

    state = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    config = state.get("config", {})

    if state.get("task") != "segmentation":
        raise RuntimeError(
            "Checkpoint is not a segmentation model"
        )

    if config.get(
        "feature_manifest_sha256"
    ) != manifest_hash:
        raise RuntimeError(
            "Checkpoint uses different manifest"
        )

    if config.get("limit") is not None:
        raise RuntimeError(
            "Checkpoint trained on smoke-test data"
        )

    if config.get("head") not in ("linear", "mlp"):
        raise RuntimeError(
            "Unexpected head architecture"
        )

    if config.get("output_shape") != [
        2, HEIGHT, WIDTH
    ]:
        raise RuntimeError(
            "Wrong output geometry"
        )

    device = (
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else args.device

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    model = CachedSegmentationProbe(
        int(ds.features.shape[1]),
        config["head"],
    )

    model.load_state_dict(
        state["model_state_dict"],
        strict=True,
    )

    model.to(device)
    model.eval()

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    print(
        f"Validation: {len(ds)} images | "
        f"Device: {device} | "
        f"Head: {config['head']}",
        flush=True,
    )

    # ---------------------------------------------
    # 4. Wahrscheinlichkeiten berechnen
    # ---------------------------------------------

    # Zeilen: tatsächliche Klassen 0 und 1
    # Spalten: Wahrscheinlichkeitsintervalle
    hist = np.zeros(
        (2, len(thresholds) + 1),
        dtype=np.int64,
    )

    with torch.inference_mode():
        for batch_idx, (feature, mask) in enumerate(
            loader, start=1
        ):
            logits = model(
                feature.to(device, non_blocking=True)
            )

            # Softmax für zwei Klassen:
            # p(MA) = sigmoid(logit_MA - logit_BG)
            prob_ma = torch.sigmoid(
                logits[:, 1] - logits[:, 0]
            ).cpu().numpy().reshape(-1)

            label = mask.numpy().reshape(-1)

            valid = (label == 0) | (label == 1)

            if not np.any(valid):
                continue

            # Wahrscheinlichkeiten einmal einsortieren.
            # Daraus lassen sich alle Thresholds auswerten.
            bucket = np.searchsorted(
                thresholds,
                prob_ma[valid],
                side="right",
            )

            n_buckets = len(thresholds) + 1

            encoded = (
                label[valid].astype(np.int64)
                * n_buckets
                + bucket
            )

            hist += np.bincount(
                encoded,
                minlength=2 * n_buckets,
            ).reshape(2, n_buckets)

            if (
                batch_idx % 10 == 0
                or batch_idx == len(loader)
            ):
                print(
                    f"Processed "
                    f"{min(batch_idx * args.batch_size, len(ds))}"
                    f"/{len(ds)}",
                    flush=True,
                )

    # ---------------------------------------------
    # 5. Metriken für alle Thresholds
    # ---------------------------------------------

    records = counts_from_buckets(
        hist, thresholds
    )

    default = next(
        r for r in records
        if abs(r["threshold"] - 0.5) < 1e-9
    )

    # Wir optimieren die MA-IoU,
    # nicht die Hintergrund-dominierte Accuracy.
    best = max(
        records,
        key=lambda r: (
            r["iou_ma"],
            -abs(r["threshold"] - 0.5),
        ),
    )

    # ---------------------------------------------
    # 6. Sanity-Check gegen vorherige Evaluation
    # ---------------------------------------------

    reference = (
        checkpoint_path.parent / "val_class_metrics.json"
    )

    if reference.exists():
        old = json.loads(
            reference.read_text(encoding="utf-8")
        )

        expected = old.get(
            "confusion_matrix_true_rows_pred_columns"
        )

        actual = [
            [default["tn"], default["fp"]],
            [default["fn"], default["tp"]],
        ]

        if expected is not None:
            max_diff = int(
                np.max(
                    np.abs(
                        np.asarray(actual, dtype=np.int64)
                        - np.asarray(expected, dtype=np.int64)
                    )
                )
            )

            print(
                "Sanity check at threshold 0.50:",
                "max confusion-cell difference =",
                max_diff,
            )

            if max_diff > 100:
                print(
                    "WARNING: CPU/GPU numerical differences "
                    "near threshold 0.50 may occur."
                )

            if max_diff > 10000:
                raise RuntimeError(
                    "Evaluation differs substantially "
                    "from saved validation metrics"
                )

    # ---------------------------------------------
    # 7. Ergebnisse speichern
    # ---------------------------------------------

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with (
        output_dir / "threshold_metrics.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(records[0].keys()),
        )
        writer.writeheader()
        writer.writerows(records)

    summary = {
        "split": "val",
        "samples": len(ds),
        "test_evaluated": False,
        "checkpoint": str(checkpoint_path),
        "manifest_sha256": manifest_hash,
        "selection_metric": "iou_ma",
        "threshold_default": 0.5,
        "default_metrics": default,
        "best_metrics": best,
        "note": (
            "Threshold optimized on validation only."
        ),
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    # ---------------------------------------------
    # 8. Plot erstellen
    # ---------------------------------------------

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5))

        for key, label in [
            ("iou_ma", "MA-IoU"),
            ("miou", "mIoU"),
            ("ma_precision", "MA-Precision"),
            ("ma_recall", "MA-Recall"),
        ]:
            ax.plot(
                [r["threshold"] for r in records],
                [r[key] for r in records],
                label=label,
            )

        ax.axvline(
            0.5,
            linestyle="--",
            linewidth=1,
            label="Default 0.50",
        )

        ax.axvline(
            best["threshold"],
            linestyle=":",
            linewidth=1,
            label=f"Best {best['threshold']:.2f}",
        )

        ax.set(
            xlabel="MA probability threshold",
            ylabel="Validation score",
            ylim=(0, 1),
        )

        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()

        fig.savefig(
            output_dir / "threshold_sweep.png",
            dpi=160,
        )

        plt.close(fig)

    except ImportError:
        print(
            "matplotlib not available; skipping plot"
        )

    # ---------------------------------------------
    # 9. Resultate im Terminal ausgeben
    # ---------------------------------------------

    print("\nVAL ONLY — THRESHOLD SWEEP")
    print(
        "Threshold | MA-IoU | mIoU | "
        "Precision | Recall"
    )

    for r in records:
        flag = " <-- BEST" if r is best else ""

        print(
            f"{r['threshold']:8.2f} | "
            f"{r['iou_ma']:6.4f} | "
            f"{r['miou']:6.4f} | "
            f"{r['ma_precision']:9.4f} | "
            f"{r['ma_recall']:6.4f}"
            f"{flag}"
        )

    print(
        "\nAt 0.50:",
        f"MA-IoU={default['iou_ma']:.4f}",
        f"Precision={default['ma_precision']:.4f}",
        f"Recall={default['ma_recall']:.4f}",
    )

    print(
        "\nBEST THRESHOLD:",
        best["threshold"],
        "MA-IoU:",
        best["iou_ma"],
    )

    print("\nSaved:", output_dir)
    print("TEST SET NOT EVALUATED")


if __name__ == "__main__":
    main()
