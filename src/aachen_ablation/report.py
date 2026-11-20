"""Validation-only tables, seed aggregation and reproducible qualitative panels."""
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .cache import OriginalCachedDataset, atomic_json
from .data import digest, load_original, resolve, sha256_file
from .engine import make_decoder, save_csv
from .geometry import VARIANTS, reconstruct


def cohort_identity(config):
    meta = config["cache_metadata"]
    contract = meta["contract"]
    return {k: config[k] for k in ("protocol", "epochs", "patience", "batch_size", "micro_batch_size",
                                   "weight_decay", "loss", "optimizer", "scheduler", "torch_version",
                                   "deterministic_algorithms", "nonscientific_smoke_test")} | {
        "execution_version": config.get("execution_version", "original-v1"),
        "determinism_policy": config.get("determinism_policy", "strict"),
        "manifest_sha256": contract["manifest_sha256"], "records": contract["records"],
        "sources": config.get("training_sources", contract["sources"]), "dtype": contract["dtype"],
        "encoder_state_sha256": meta["encoder_state_sha256"],
        "component_size_bins": config["component_size_bins"]}


def create_report(root, example_count=3):
    root = Path(root)
    entries = []
    for path in sorted((root / "runs").glob("*/*/*/validation_metrics.json")):
        metrics = json.loads(path.read_text())
        config = json.loads((path.parent / "config.json").read_text())
        if metrics.get("test_evaluated") is not False or metrics.get("split") != "val":
            raise ValueError("Report accepts validation-only results")
        entries.append((path.parent, config, metrics))
    if not entries:
        raise FileNotFoundError(f"No completed validation runs in {root}")
    cohorts = defaultdict(list)
    for entry in entries:
        cohorts[digest(cohort_identity(entry[1]))[:12]].append(entry)
    destinations = []
    for cohort, part in cohorts.items():
        # Snapshot destination makes reports immutable; different evaluations get new reports.
        snapshot = digest([(str(p), m) for p, _, m in part])[:12]
        out = root / "reports" / f"{cohort}_{snapshot}_examples{example_count}"
        if (out / "report_complete.json").exists():
            print(f"Report complete, skipping: {out}", flush=True)
            destinations.append(out)
            continue
        out.mkdir(parents=True, exist_ok=True)
        atomic_json(out / "cohort.json", cohort_identity(part[0][1]))
        rows, component_table = [], []
        seen = set()
        for path, config, metrics in part:
            key = metrics["variant"], metrics["head"], metrics["seed"]
            if key in seen:
                raise ValueError(f"Multiple configurations for {key}; separate output roots to compare LR protocols")
            seen.add(key)
            for operating, field in (("fixed_0.5", "validation_at_0_50"), ("val_selected", "validation_at_best_threshold")):
                row = {"variant": key[0], "head": key[1], "seed": key[2], "operating_point": operating,
                       **metrics[field], "best_epoch": metrics["best_epoch"],
                       "patch_tokens": metrics["patch_tokens_per_original"],
                       "cached_decoder_seconds_per_original": metrics["cached_decoder_seconds_per_original"],
                       "end_to_end_seconds_per_original": metrics["end_to_end_seconds_per_original"],
                       "train_peak_gpu_bytes": metrics["train_peak_gpu_bytes"],
                       "eval_peak_gpu_bytes": metrics["eval_peak_gpu_bytes"],
                       "run": str(path)}
                row["confusion_matrix"] = json.dumps(row["confusion_matrix"])
                rows.append(row)
                threshold = metrics[field]["threshold"]
                for name, stats in metrics["components"][str(threshold)].items():
                    component_table.append({"variant": key[0], "head": key[1], "seed": key[2],
                                            "operating_point": operating, "size_class": name, **stats})
        save_csv(out / "comparison.csv", rows)
        save_csv(out / "island_comparison.csv", component_table)
        aggregates = []
        for key in sorted({(r["variant"], r["head"], r["operating_point"]) for r in rows}):
            group = [r for r in rows if (r["variant"], r["head"], r["operating_point"]) == key]
            record = {"variant": key[0], "head": key[1], "operating_point": key[2], "n_seeds": len(group),
                      "seeds": ",".join(str(r["seed"]) for r in group)}
            for metric in ("ma_iou", "ma_recall", "miou", "background_iou", "ma_precision",
                           "end_to_end_seconds_per_original", "cached_decoder_seconds_per_original"):
                values = [r[metric] for r in group if r[metric] is not None]
                record[metric+"_mean"] = float(np.mean(values)) if values else None
                record[metric+"_std"] = float(np.std(values, ddof=1)) if len(values)>1 else 0. if values else None
            aggregates.append(record)
        save_csv(out / "seed_summary.csv", aggregates)
        plot_report(out, aggregates, component_table, part[0][1]["nonscientific_smoke_test"])
        if example_count:
            example_panels(out, part, example_count)
        expected = {f"{v}/{h}" for v in VARIANTS for h in ("linear", "cnn8")}
        actual = {f"{m['variant']}/{m['head']}" for _, _, m in part}
        atomic_json(out / "report_complete.json", {"runs": len(part), "test_evaluated": False,
                    "missing_configurations": sorted(expected-actual), "complete_eight_configurations": expected <= actual,
                    "nonscientific_smoke_test": part[0][1]["nonscientific_smoke_test"]})
        print(f"Report: {out} (missing: {sorted(expected-actual)})", flush=True)
        destinations.append(out)
    return destinations


def plot_report(out, aggregate, components, smoke):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/aachen-ablation-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    def finish(fig, name):
        if smoke:
            fig.suptitle("NONSCIENTIFIC SMOKE TEST — synthetic encoder")
        fig.tight_layout()
        fig.savefig(out / (name+".png"), dpi=160)
        fig.savefig(out / (name+".pdf"))
        plt.close(fig)

    for operating in ("fixed_0.5", "val_selected"):
        subset = [r for r in aggregate if r["operating_point"] == operating]
        for metric in ("ma_iou", "ma_recall"):
            fig, ax = plt.subplots(figsize=(6, 4))
            for head in ("linear", "cnn8"):
                part = sorted([r for r in subset if r["head"] == head and r["variant"].startswith("A")],
                              key=lambda r: r["variant"])
                if part:
                    ax.errorbar([VARIANTS[r["variant"]].width for r in part],
                                [r[metric+"_mean"] for r in part],
                                yerr=[r[metric+"_std"] for r in part], marker="o", label=head)
            ax.set(xlabel="Full-image input width (4:3)", ylabel=metric, title=operating)
            ax.legend()
            finish(fig, f"{metric}_resolution_{operating}")
        fig, ax = plt.subplots(figsize=(7, 4))
        for hi, head in enumerate(("linear", "cnn8")):
            part = sorted([r for r in subset if r["head"] == head], key=lambda r: r["variant"])
            x = [list(VARIANTS).index(r["variant"])+(.18 if hi else -.18) for r in part]
            ax.bar(x, [r["ma_iou_mean"] for r in part], width=.34, label=head,
                   yerr=[r["ma_iou_std"] for r in part])
        ax.set(xticks=range(4), xticklabels=list(VARIANTS), ylabel="MA IoU", title=operating)
        ax.legend()
        finish(fig, f"ma_iou_strategies_{operating}")
        fig, ax = plt.subplots(figsize=(7, 4))
        for r in subset:
            timing = r["end_to_end_seconds_per_original_mean"]
            if timing is not None:
                ax.scatter(timing*1000, r["ma_iou_mean"])
                ax.annotate(r["variant"]+"/"+r["head"], (timing*1000, r["ma_iou_mean"]), fontsize=7)
        ax.set(xlabel="End-to-end inference ms / original (no disk IO)", ylabel="MA IoU", title=operating)
        finish(fig, f"quality_vs_time_{operating}")
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, metric in zip(axes, ("mean_island_pixel_recall", "pixel_weighted_recall", "fraction_detected_at_25pct")):
            for key in sorted({(r["variant"], r["head"]) for r in components}):
                means, stds = [], []
                for name in ("small", "medium", "large"):
                    values = [r[metric] for r in components if (r["variant"], r["head"]) == key
                              and r["operating_point"] == operating and r["size_class"] == name and r[metric] is not None]
                    means.append(np.mean(values) if values else np.nan)
                    stds.append(np.std(values, ddof=1) if len(values)>1 else 0.)
                ax.errorbar(range(3), means, yerr=stds, marker="o", label="/".join(key))
            ax.set(xticks=range(3), xticklabels=["small", "medium", "large"], ylabel=metric)
        axes[-1].legend(fontsize=6)
        finish(fig, f"island_size_{operating}")


@torch.inference_mode()
def example_panels(out, entries, count):
    from matplotlib import pyplot as plt
    # Use the smallest available seed per variant/head; IDs sampled once across ALL models.
    selected = {}
    for entry in sorted(entries, key=lambda e: e[2]["seed"]):
        selected.setdefault((entry[2]["variant"], entry[2]["head"]), entry)
    datasets = {}
    common = None
    for key, (_, config, _) in selected.items():
        contract = config["cache_metadata"]["contract"]
        ds = OriginalCachedDataset(config["feature_cache"], contract,
                                   # Absolute manifest is recorded in the preflight; original records resolve relative to it.
                                   config["manifest"], "val")
        datasets[key] = ds
        ids = {s["record"]["image_id"] for s in ds.samples}
        common = ids if common is None else common & ids
    ids = sorted(common)
    chosen = np.random.default_rng(42).choice(ids, min(count, len(ids)), replace=False).tolist()
    atomic_json(out / "selected_examples.json", {"seed": 42, "validation_image_ids": chosen,
                "model_seeds": {"/".join(k): e[2]["seed"] for k, e in selected.items()}})
    predictions = {image_id: [] for image_id in chosen}
    originals = {}
    for key, (path, config, metrics) in sorted(selected.items()):
        ds = datasets[key]
        model = make_decoder(key[1]).eval()
        state = torch.load(path / "best_model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state["model_state_dict"])
        for image_id in chosen:
            idx = next(i for i, s in enumerate(ds.samples) if s["record"]["image_id"] == image_id)
            sample = ds[idx]
            row = sample["record"]
            for field in ("image", "mask"):
                if sha256_file(resolve(ds.manifest, row[field+"_path"])) != row[field+"_sha256"]:
                    raise ValueError("Original changed since feature extraction")
            image, mask = load_original(ds.manifest, row)
            originals[image_id] = image, mask.numpy()
            z = reconstruct(model(sample["features"]), image.size, VARIANTS[key[0]])
            probability = torch.sigmoid(z[1]-z[0]).numpy()
            # Fixed threshold keeps qualitative comparisons on the same operating rule.
            predictions[image_id].append(("/".join(key)+f" s{metrics['seed']} t=0.5", probability >= .5))
    for index, image_id in enumerate(chosen):
        panels = predictions[image_id]
        fig, axes = plt.subplots(1, 2+len(panels), figsize=(3*(2+len(panels)), 3))
        image, mask = originals[image_id]
        axes[0].imshow(image)
        axes[0].set_title(image_id)
        axes[1].imshow(np.ma.masked_equal(mask, 255), vmin=0, vmax=1, cmap="gray")
        axes[1].set_title("Original ground truth (ignore masked)")
        for ax, (name, prediction) in zip(axes[2:], panels):
            ax.imshow(prediction, vmin=0, vmax=1, cmap="gray")
            ax.set_title(name)
        for ax in axes:
            ax.axis("off")
        if entries[0][1]["nonscientific_smoke_test"]:
            fig.suptitle("NONSCIENTIFIC SMOKE TEST")
        fig.tight_layout()
        fig.savefig(out / f"example_{index:02d}.png", dpi=160)
        fig.savefig(out / f"example_{index:02d}.pdf")
        plt.close(fig)
