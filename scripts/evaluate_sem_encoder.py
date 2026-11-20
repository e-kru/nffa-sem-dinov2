"""Frozen encoder comparison: configurable existing heads, train/validation only.

See docs/downstream_evaluation.md for the staged CUDA commands and fixed protocols.
This script never invokes SSL or accesses test images/masks/labels.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
os.environ["XFORMERS_DISABLED"] = "1"

import numpy as np
import sklearn
import torch
import torchvision

from src.aachen_ablation.cache import atomic_json, git_info
from src.aachen_ablation.data import digest, sha256_file
from src.aachen_ablation.devices import select_device
from src.config import DATASET_ROOT
from src.downstream.benchmark import (
    ROOT, arrays, classification_records, extract_features, feature_contract, sources,
)
from src.downstream.probes import HEAD_PROFILES, head_configurations, evaluate_probe, fit_logreg, fit_mlp
from src.ssl.upstream import PIN, activate

from src.config import VITS14_CHECKPOINT

REFERENCE = VITS14_CHECKPOINT
LEGACY_HUB = Path(os.environ.get(
    "DINOV2_LEGACY_HUB", REFERENCE.parent.parent / "facebookresearch_dinov2_main",
)).expanduser().resolve()
DEFAULT_HEADS = {"classification": ["logreg", "mlp"], "segmentation": ["linear", "cnn8"]}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=("classification", "segmentation", "both"), default="both")
    p.add_argument("--heads", nargs="+", help="Task-specific heads; with 'both' use the two task-specific options")
    p.add_argument('--head-profile', choices=HEAD_PROFILES, default='legacy',
                   help='residual_v1: residual MLP with 10%% dropout; CNN8 already uses residual blocks and 10%% dropout')
    p.add_argument("--classification-heads", nargs="+", choices=DEFAULT_HEADS["classification"])
    p.add_argument("--segmentation-heads", nargs="+", choices=DEFAULT_HEADS["segmentation"])
    p.add_argument("--encoder-name", choices=("dinov2_vits14", "dinov2_vitl14"), default="dinov2_vits14")
    p.add_argument("--encoder-checkpoint", type=Path)
    p.add_argument("--legacy-hub-dir", type=Path, default=LEGACY_HUB)
    p.add_argument("--stage", choices=("preflight", "extract", "train", "evaluate", "report", "all"), default="preflight")
    p.add_argument("--output-root", type=Path, default=Path("outputs/encoder_evaluation/v1"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--workers", "--num-workers", dest="workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--splits", nargs="+", choices=("A", "B"), default=["A", "B"])
    p.add_argument("--classification-manifest", type=Path, default=Path("outputs/dataset_v3/manifest.csv"))
    p.add_argument("--split-a", type=Path, default=Path("outputs/dataset_v3/splits/split_A_exact.csv"))
    p.add_argument("--split-b", type=Path, default=Path("outputs/dataset_v3/splits/split_B_full.csv"))
    p.add_argument("--image-root", type=Path, default=DATASET_ROOT)
    p.add_argument("--segmentation-manifest", type=Path,
                   default=Path("outputs/aachen_heerlen/splits/group_split_seed42.csv"))
    p.add_argument("--resume", action="store_true", help="Reuse matching completed artifacts; resume extraction/epoch checkpoints")
    p.add_argument("--verified-data-root", type=Path,
                   help="Reuse a previously verified encoder's dataset metadata; no image-content rehashing")
    p.add_argument("--smoke-test", action="store_true", help="FAKE encoder, limited samples/epochs; NONSCIENTIFIC output")
    return p


def selected_heads(args):
    tasks = list(DEFAULT_HEADS) if args.task == "both" else [args.task]
    if args.heads and args.task == "both":
        raise ValueError("--heads requires one --task; use --classification-heads and --segmentation-heads with both")
    for task in set(DEFAULT_HEADS) - set(tasks):
        if getattr(args, task+"_heads"):
            raise ValueError(f"--{task}-heads requires --task {task} or both")
    selected = {}
    for task in tasks:
        explicit = getattr(args, task+"_heads")
        if args.heads and explicit:
            raise ValueError("Specify either --heads or the task-specific head option")
        names = args.heads or explicit or DEFAULT_HEADS[task]
        if len(names) != len(set(names)) or set(names) - set(DEFAULT_HEADS[task]):
            raise ValueError(f"Choose unique {task} heads from {DEFAULT_HEADS[task]}")
        selected[task] = names
    return selected


def software():
    return {"python": sys.version.split()[0], "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__), "numpy": np.__version__,
            "sklearn": sklearn.__version__}


def identity_and_root(args):
    name = getattr(args, "encoder_name", "dinov2_vits14")
    if name == "dinov2_vitl14" and (not args.encoder_checkpoint or args.smoke_test):
        raise ValueError("Large evaluation requires an explicit checkpoint and real features")
    checkpoint = (args.encoder_checkpoint or REFERENCE).resolve()
    sha = sha256_file(checkpoint) if (not args.smoke_test or args.encoder_checkpoint) else "FAKE"
    if not args.encoder_checkpoint and not args.smoke_test and sha != PIN["backbone_sha256"]:
        raise ValueError("Reference checkpoint does not match the pinned Meta release SHA256")
    identity = {"checkpoint_sha256": sha, "upstream_commit": PIN["commit"],
                "model": name, "nonscientific_smoke_test": args.smoke_test}
    if name == "dinov2_vitl14":
        identity["feature_dim"] = 1024
    root = args.output_root.resolve()
    if args.smoke_test:
        root = root / "NONSCIENTIFIC_SMOKE"
    return checkpoint, identity, root / f"{name}_{sha[:16]}"


def verification_contract(checkpoint, args, identity):
    verification_files = ("src/dinov2_encoder.py", "src/downstream/pinned_encoder.py", "src/ssl/upstream.py",
                          "scripts/verify_frozen_encoder.py")
    return {"encoder": identity, "software": software(),
            "sources": {f: sha256_file(ROOT / f) for f in verification_files},
            "legacy_hub_dir": str(args.legacy_hub_dir.resolve()),
            "legacy_sources": {str(f.relative_to(args.legacy_hub_dir)): sha256_file(f)
                               for f in sorted(args.legacy_hub_dir.rglob("*.py"))}}


def verify_loading(root, checkpoint, args, identity, run=False):
    """Acceptance gate: exact offline loading/forward equivalence, no benchmark data."""
    contract = verification_contract(checkpoint, args, identity)
    path = root / "encoder_verification.json"
    if path.exists():
        result = json.loads(path.read_text())
        if result.get("contract") != contract or result.get("status") != "passed":
            raise ValueError("Encoder verification is stale/incompatible; use a new output root")
        return result
    if not run:
        raise ValueError("Run --stage preflight before extraction/training/evaluation")
    if args.smoke_test:
        result = {"status": "passed", "contract": contract,
                  "nonscientific_smoke_test": True, "loading_equivalence_verified": False}
    else:
        activate()
        if not (args.legacy_hub_dir / "hubconf.py").is_file():
            raise FileNotFoundError("Legacy Hub checkout unavailable; specify --legacy-hub-dir")
        outputs = []
        with tempfile.TemporaryDirectory(prefix="sem_encoder_verify_") as temporary:
            for source in ("legacy", "pinned"):
                destination = Path(temporary) / f"{source}.pt"
                command = [sys.executable, "-m", "scripts.verify_frozen_encoder", "--source", source,
                           "--checkpoint", str(checkpoint), "--legacy-hub-dir", str(args.legacy_hub_dir),
                           "--output", str(destination), "--encoder-name", identity["model"]]
                subprocess.run(command, cwd=ROOT, check=True)
                outputs.append(torch.load(destination, map_location="cpu", weights_only=True))
        old, new = outputs
        if old["encoder_state_sha256"] != new["encoder_state_sha256"] or not all(
                o["frozen"] and o["eval"] for o in outputs):
            raise ValueError("Loaded weights/frozen state differ between baseline and pinned loader")
        dim = identity.get("feature_dim", 384)
        errors = {}
        for name, shape in (("global", (1, dim)), ("dense", (1, dim, 48, 64))):
            if any(tuple(o[name].shape) != shape or not torch.isfinite(o[name]).all() for o in outputs):
                raise ValueError(f"Invalid {name} backbone outputs")
            torch.testing.assert_close(old[name], new[name], rtol=0, atol=0)
            errors[name] = float((old[name]-new[name]).abs().max())
        result = {"status": "passed", "contract": contract, "loading_equivalence_verified": True,
                  "encoder_state_sha256": new["encoder_state_sha256"], "max_abs_error": errors,
                  "global_shape": [1, dim], "dense_shape": [1, dim, 48, 64],
                  "legacy_source": old["source"], "pinned_source": new["source"],
                  "device": "cpu", "precision": "float32", "frozen": True, "eval": True}
    atomic_json(path, result)
    return result


def run_classification(args, root, checkpoint, identity, heads, device):
    profile = getattr(args, 'head_profile', 'legacy')
    configurations = head_configurations(profile)
    for split in args.splits:
        manifest = args.classification_manifest.resolve()
        split_file = getattr(args, "split_"+split.lower()).resolve()
        if args.verified_data_root:
            from src.ssl.ablation_data import classification
            records = classification(args.verified_data_root/'classification'/f'split_{split}'/'preflight.json',
                                     manifest, split_file, args.image_root)
        else:
            records = classification_records(manifest, split_file, args.image_root,
                                             args.workers, args.smoke_test, progress=True)
        contract = feature_contract(records, manifest, split_file, args.image_root, identity, args.smoke_test)
        base = root / "classification" / f"split_{split}"
        base.mkdir(parents=True, exist_ok=True)
        preflight = {"contract": contract, "head_configurations": configurations,
                     "seed": args.seed, "workers": args.workers, "test_files_opened": False,
                     "counts": {s: sum(r["split"] == s for r in records) for s in ("train", "val")}}
        if args.verified_data_root:
            from src.ssl.ablation_data import __file__ as reuse_source
            preflight['reused_verified_data'] = dict(root=str(args.verified_data_root.resolve()),
                                                    reader_sha256=sha256_file(reuse_source))
        path = base / "preflight.json"
        if path.exists():
            if json.loads(path.read_text()) != preflight:
                raise ValueError("Classification protocol/data changed; choose a new output root")
        else:
            atomic_json(path, preflight)
        if args.stage == "preflight":
            print(f"Classification split {split}: {preflight['counts']}, heads={heads}", flush=True)
            continue
        cache = base / "features"
        if args.stage in ("extract", "all"):
            extract_features(cache, contract, checkpoint, device, args.workers, args.resume, args.smoke_test)
        if args.stage == "extract":
            continue
        X_train, y_train, X_val, y_val = arrays(cache, contract)
        cache_meta = json.loads((cache / "cache_metadata.json").read_text())
        for head in heads:
            head_config = copy.deepcopy(configurations[head])
            if args.smoke_test and head == "mlp":
                head_config["epochs"] = 2
            config = {"head": head, "head_config": head_config, "seed": args.seed,
                      "head_profile": profile,
                      "workers": args.workers, "device": str(device), "software": software(),
                      "cache_contract_sha256": digest(contract), "split": split,
                      "nonscientific_smoke_test": args.smoke_test}
            output = base / head / f"seed{args.seed}"
            config_file = output / "config.json"
            if config_file.exists():
                if json.loads(config_file.read_text()) != config:
                    raise ValueError(f"Head configuration changed: {output}")
            else:
                if output.exists() and any(output.iterdir()):
                    raise FileExistsError(output)
                output.mkdir(parents=True, exist_ok=True)
                atomic_json(config_file, config)
            if args.stage in ("train", "all"):
                if (not args.resume and not (output / "training_complete.json").exists()
                        and any(p.name != "config.json" for p in output.iterdir())):
                    raise FileExistsError(f"Interrupted head; use --resume: {output}")
                if (output / "training_complete.json").exists():
                    if not args.resume:
                        raise FileExistsError(f"Completed head; use --resume to skip: {output}")
                elif head == "logreg":
                    fit_logreg(output, X_train, y_train, X_val, y_val, head_config, args.seed)
                else:
                    fit_mlp(output, X_train, y_train, X_val, y_val, head_config, args.seed,
                            args.workers, device, args.resume)
            if args.stage in ("evaluate", "all"):
                metrics_path = output / "validation_metrics.json"
                if not metrics_path.exists():
                    result = evaluate_probe(output, head, X_val, y_val, head_config, device)
                    atomic_json(metrics_path, {"task": "classification", "split": split,
                        "partition": "val", "head": head, "seed": args.seed,
                        "encoder": identity, "head_config": head_config, "samples": len(y_val),
                        "encoder_state_sha256": cache_meta["encoder_state_sha256"],
                        "test_evaluated": False, "nonscientific_smoke_test": args.smoke_test, **result})


def run_segmentation(args, root, checkpoint, heads):
    # Delegate every training/evaluation operation to the working original-raster runner.
    from scripts.run_aachen_ablation import main as aachen_main
    command = ["--manifest", str(args.segmentation_manifest.resolve()),
               "--output-root", str(root / "segmentation"),
               "--experiment", "A", "--variants", "A3", "--heads", *heads,
               "--seeds", str(args.seed), "--stage", args.stage,
               "--device", args.device, "--workers", str(args.workers),
               "--encoder-name", getattr(args, "encoder_name", "dinov2_vits14")]
    if not args.smoke_test or args.encoder_checkpoint:
        command += ["--encoder-checkpoint", str(checkpoint), "--pinned-encoder"]
    if args.smoke_test:
        command.append("--smoke-test")
    if args.resume:
        command.append("--resume")
    if args.verified_data_root:
        from src.ssl.ablation_data import segmentation_reference
        command += ["--verified-data", str(segmentation_reference(args.verified_data_root))]
    aachen_main(command)


def report(args, root, identity, heads):
    rows, missing, fingerprints = [], [], set()
    if "classification" in heads:
        for split in args.splits:
            for head in heads["classification"]:
                path = root / "classification" / f"split_{split}" / head / f"seed{args.seed}" / "validation_metrics.json"
                if not path.is_file():
                    missing.append(str(path.relative_to(root)))
                else:
                    metrics = json.loads(path.read_text())
                    if metrics.get("encoder") != identity or metrics.get("partition") != "val":
                        raise ValueError("Classification report encoder/partition mismatch")
                    expected = head_configurations(getattr(args,'head_profile','legacy'))[head]
                    if args.smoke_test and head == 'mlp':
                        expected['epochs'] = 2
                    if metrics.get('head_config') != expected:
                        raise ValueError('Classification report head profile mismatch')
                    fingerprints.add(metrics["encoder_state_sha256"])
                    rows.append({"task": "classification", "split": split, "head": head,
                                 "seed": args.seed, "artifact": str(path),
                                 "metrics": metrics})
    if "segmentation" in heads:
        base = root / "segmentation"
        if args.smoke_test:
            base = base / "NONSCIENTIFIC_SMOKE"
        for head in heads["segmentation"]:
            paths = sorted(base.glob(f"runs/A3/{head}/seed{args.seed}_*/validation_metrics.json"))
            if len(paths) != 1:
                missing.append(f"segmentation A3/{head}/seed{args.seed}: expected one result, found {len(paths)}")
            else:
                config = json.loads((paths[0].parent / "config.json").read_text())
                cache = config["cache_metadata"]
                if cache["contract"]["checkpoint_sha256"] != identity["checkpoint_sha256"] and not args.smoke_test:
                    raise ValueError("Segmentation report checkpoint mismatch")
                fingerprints.add(cache["encoder_state_sha256"])
                rows.append({"task": "segmentation", "head": head, "seed": args.seed,
                             "artifact": str(paths[0]), "metrics": json.loads(paths[0].read_text())})
    if any(r["metrics"].get("test_evaluated") is not False for r in rows):
        raise ValueError("Report requires validation-only artifacts")
    if len(fingerprints) > 1 and not args.smoke_test:
        raise ValueError("Frozen backbone weights differ between tasks/splits")
    result = {"encoder": identity, "partition": "val", "test_evaluated": False,
              "complete": not missing, "missing": missing, "results": rows,
              "encoder_state_fingerprints": sorted(fingerprints),
              "historical_baseline_reproduced": False,
              "note": "Historical test scores are not validation references; baseline reproduction requires manual full runs."}
    destination = root / "reports" / digest(result)[:16]
    destination.mkdir(parents=True, exist_ok=True)
    if not (destination / "report.json").exists():
        atomic_json(destination / "report.json", result)
        lines = ["# Frozen DINOv2 validation report", "", f"Complete: {result['complete']}", "",
                 "All scores are fractions. Threshold tuning uses validation only.", "",
                 "| Task | Split | Head | Threshold protocol | Threshold | Accuracy | Macro-F1 | MA-IoU | mIoU | MA-Precision | MA-Recall |",
                 "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for row in rows:
            m = row["metrics"]
            operating_points = [("n/a", {})] if row["task"] == "classification" else [
                ("fixed", m["validation_at_0_50"]),
                ("validation-selected", m["validation_at_best_threshold"])]
            for protocol, point in operating_points:
                values = [point.get("threshold"), m.get("accuracy"), m.get("macro_f1"),
                          point.get("ma_iou"), point.get("miou"),
                          point.get("ma_precision"), point.get("ma_recall")]
                lines.append(f"| {row['task']} | {row.get('split', 'val')} | {row['head']} | {protocol} | " +
                             " | ".join(f"{v:.6f}" if v is not None else "—" for v in values) + " |")
        if missing:
            lines += ["", "Missing:", "", *[f"- {m}" for m in missing]]
        (destination / "report.md").write_text("\n".join(lines)+"\n")
    print(f"Report: {destination / 'report.md'} (complete={not missing})", flush=True)


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        heads = selected_heads(args)
        if args.workers < 0 or len(args.splits) != len(set(args.splits)):
            raise ValueError("Workers must be nonnegative and splits unique")
        if args.smoke_test:
            args.workers = 0
            if args.verified_data_root:
                raise ValueError('Verified scientific inventories cannot be used for a smoke test')
        torch.set_num_threads(1)
        checkpoint, identity, root = identity_and_root(args)
        root.mkdir(parents=True, exist_ok=True)
        if args.stage == "report":
            report(args, root, identity, heads)
            return
        verify_loading(root, checkpoint, args, identity, run=args.stage in ("preflight", "all"))
        device = torch.device("cpu")
        if args.stage != "preflight":
            device = select_device(args.device)
            if device.type == "cuda":
                if (torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name(device)
                        or torch.cuda.get_device_properties(device).total_memory < 75 * 1024**3):
                    raise ValueError("Expose exactly one A100 80GB via CUDA_VISIBLE_DEVICES")
        provenance = root / "provenance.json"
        if not provenance.exists():
            atomic_json(provenance, {"encoder": identity, "checkpoint": str(checkpoint),
                                     "git": git_info(), "software": software()})
        print(f"Encoder artifacts: {root}", flush=True)
        if "classification" in heads:
            run_classification(args, root, checkpoint, identity, heads["classification"], device)
        if "segmentation" in heads:
            run_segmentation(args, root, checkpoint, heads["segmentation"])
        if args.stage == "all":
            report(args, root, identity, heads)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
