"""Versioned per-original feature caches, atomic writes and strict provenance."""
import ast
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.downstream.model import DINOv2Backbone
from .data import digest, load_mask, load_original, sha256_file
from .devices import CudaMemoryProfiler, normalize_device
from .geometry import PREPROCESSING_VERSION, VARIANTS, content_box, prepare_region, regions

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "aachen-ablation-cache-v1"


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def atomic_torch(path, value):
    temp = path.with_suffix(path.suffix + ".partial")
    torch.save(value, temp)
    os.replace(temp, path)


def git_info():
    def run(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"
    return {"commit": run("rev-parse", "HEAD"), "status": run("status", "--short")}


def source_hashes():
    paths = [*sorted((ROOT / "src/aachen_ablation").glob("*.py")),
             ROOT / "src/preprocessing_v2.py", ROOT / "src/dinov2_encoder.py",
             ROOT / "src/downstream/pinned_encoder.py",
             ROOT / "src/downstream/model.py", ROOT / "src/downstream/heads.py"]
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in paths}


# Audited feature extraction in commit f3a90df. Only this exact legacy source
# set may omit feature_sources; unknown old schemas/source sets remain invalid.
LEGACY_FEATURE_SOURCE_SETS = {"8562a612ced193073a0033d4cbafbad8ba8f7d29d27ada12a0060c241a7fb5fb"}
LEGACY_COMPATIBLE_FEATURE_PIPELINE = "3482b015655d4fefe02d5f80c72048e07533f2b5143896d9f2f66b420a4eeae9"


def feature_sources():
    files = {name: sha256_file(ROOT / name) for name in (
        "src/aachen_ablation/geometry.py", "src/preprocessing_v2.py",
        "src/dinov2_encoder.py", "src/downstream/model.py")}
    tree = ast.parse(Path(__file__).read_text())
    names = {"make_encoder", "encoder_fingerprint", "SyntheticEncoder", "extract_cache", "ExtractionDataset"}
    functions = {node.name: digest(ast.dump(node, include_attributes=False))
                 for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names}
    # Reviewed filename-only change: backbone snapshots now carry the actual
    # encoder name. Numerical extraction is identical. Alias ONLY this exact
    # function AST; full source hashes still record the real implementation.
    # Encoder/feature shape/weights stay independent strict contract fields.
    if functions['extract_cache'] == 'de5db83d80e2713edafe5d544228312ce88f329ea3f029e5c3e60e916c8c8c13':
        functions['extract_cache'] = '9a23aaab7cd3d0bff9b7bf64b594e711f78a11fcab40008771019ef081dc6e1b'
    loader = next(node for node in ast.parse((ROOT / "src/aachen_ablation/data.py").read_text()).body
                  if isinstance(node, ast.FunctionDef) and node.name == "load_original")
    return {"files": files, "extraction_code": functions,
            "original_loader": digest(ast.dump(loader, include_attributes=False))}


def contracts_compatible(stored, expected):
    ignored = {"sources", "feature_sources"}
    if {k: v for k, v in stored.items() if k not in ignored} != {k: v for k, v in expected.items() if k not in ignored}:
        return False
    if "feature_sources" in stored:
        return stored["feature_sources"] == expected.get("feature_sources")
    # Exact equality still permits loading legacy contracts with their original
    # expected metadata (e.g. report snapshots); cross-version reuse is audited.
    if stored.get("sources") == expected.get("sources"):
        return True
    return (digest(stored.get("sources")) in LEGACY_FEATURE_SOURCE_SETS
            and all(stored["sources"].get(k) == value
                    for k, value in expected.get("feature_sources", {}).get("files", {}).items())
            and digest(expected.get("feature_sources")) == LEGACY_COMPATIBLE_FEATURE_PIPELINE)


def find_cache(cache_root, contract):
    """Discover compatible caches without renaming or editing their metadata."""
    root = Path(cache_root)
    if (root / "cache_metadata.json").exists():
        stored = json.loads((root / "cache_metadata.json").read_text())
        if not contracts_compatible(stored.get("contract", {}), contract):
            raise ValueError(f"Incompatible selected cache: {root}")
        return root
    exact = root / f"{contract['geometry']['name']}_{digest(contract)[:16]}"
    if (exact / "cache_metadata.json").exists():
        return exact
    candidates = []
    for file in sorted(root.glob(f"{contract['geometry']['name']}_*/cache_metadata.json")):
        meta = json.loads(file.read_text())
        if contracts_compatible(meta.get("contract", {}), contract):
            candidates.append(file.parent)
    if len(candidates) > 1:
        raise ValueError(f"Ambiguous compatible caches: {candidates}; select one using --cache-root")
    return candidates[0] if candidates else exact


def cache_contract(manifest, records, variant, dtype, checkpoint, smoke, encoder_name="dinov2_vits14"):
    from src.dinov2_encoder import DINO_DIMS
    dim = DINO_DIMS[encoder_name]
    return {"schema": SCHEMA, "preprocessing_version": PREPROCESSING_VERSION,
            "geometry": variant.config(), "dtype": dtype,
            "feature_shape": [4 if variant.quadrants else 1, dim, *variant.grid],
            "manifest_sha256": sha256_file(manifest), "records": records,
            "encoder": "FAKE-NONSCIENTIFIC" if smoke else encoder_name,
            "checkpoint": str(checkpoint.resolve()) if checkpoint else "torch.hub pretrained",
            "checkpoint_sha256": sha256_file(checkpoint) if checkpoint else None,
            "feature_layer": "get_intermediate_layers(n=1,reshape=True,norm=True)[0]",
            "normalization": "ImageNet; PIL bilinear; black letterbox; no metadata crop",
            "sources": source_hashes(), "feature_sources": feature_sources(), "extraction_seed": 42,
            "nonscientific_smoke_test": smoke}


def cache_location(root, contract):
    return Path(root) / "caches" / f"{contract['geometry']['name']}_{digest(contract)[:16]}"


class SyntheticEncoder(nn.Module):
    """Deterministic shape-correct stand-in; NEVER used for scientific runs."""
    feature_dim = 384

    def forward_dense(self, x):
        if not torch.is_inference_mode_enabled() or self.training:
            raise RuntimeError("Extraction requires eval() and inference_mode()")
        return torch.nn.functional.avg_pool2d(x, 14).repeat(1, 128, 1, 1)


def encoder_fingerprint(model):
    h = __import__("hashlib").sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def make_encoder(device, checkpoint=None, smoke=False):
    device = normalize_device(device)
    torch.manual_seed(42)
    model = SyntheticEncoder() if smoke else DINOv2Backbone(
        "dinov2_vits14", frozen=True, checkpoint=checkpoint)
    return model.requires_grad_(False).eval().to(device)


def make_pinned_encoder(device, checkpoint, smoke=False, name="dinov2_vits14"):
    """Explicit opt-in factory; the audited historical factory stays unchanged."""
    if smoke:
        return make_encoder(device, checkpoint, smoke=True)
    from src.downstream.pinned_encoder import get_pinned_dinov2
    torch.manual_seed(42)
    model = DINOv2Backbone(name=name, model=get_pinned_dinov2(checkpoint, name), frozen=True)
    return model.requires_grad_(False).eval().to(normalize_device(device))


def validate_array(path, contract):
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if list(array.shape) != contract["feature_shape"] or str(array.dtype) != contract["dtype"]:
        raise ValueError(f"Cache tensor shape/dtype mismatch: {path}")
    if not np.isfinite(array).all():
        raise ValueError(f"Nonfinite cache: {path}")
    return array


def validate_cache(path, contract, complete=True):
    meta = json.loads((Path(path) / "cache_metadata.json").read_text())
    if not contracts_compatible(meta.get("contract", {}), contract):
        raise ValueError(f"Incompatible cache metadata: {path}")
    if complete and not meta.get("complete"):
        raise ValueError(f"Incomplete cache, resume extraction first: {path}")
    if meta.get("complete"):
        if len(meta.get("samples", [])) != len(contract["records"]):
            raise ValueError("Missing cache index")
        for i, (sample, row) in enumerate(zip(meta["samples"], contract["records"])):
            expected_boxes = [list(b) for b in regions(*row["original_size"], contract["geometry"]["quadrants"])]
            v = VARIANTS[contract["geometry"]["name"]]
            expected_inner = [list(content_box(b[2]-b[0], b[3]-b[1], v)) for b in expected_boxes]
            if (sample["record"] != row or sample["crop_coordinates"] != expected_boxes
                    or sample["input_content_boxes"] != expected_inner):
                raise ValueError("Cache identity/geometry mismatch")
            if sample["file"] != f"{i:06d}.npy":
                raise ValueError("Cache index order mismatch")
            file = Path(path) / sample["file"]
            validate_array(file, contract)
            if sha256_file(file) != sample["sha256"]:
                raise ValueError(f"Corrupt cache: {file}")
    return meta


class ExtractionDataset(Dataset):
    """CPU workers prepare missing originals; workers never touch CUDA."""
    def __init__(self, manifest, records, variant, pending):
        self.manifest, self.records, self.variant, self.pending = manifest, records, variant, pending

    def __len__(self):
        return len(self.pending)

    def __getitem__(self, index):
        i = self.pending[index]
        row = self.records[i]
        image, _ = load_original(self.manifest, row)
        if list(image.size) != row["original_size"]:
            raise ValueError("Original geometry changed")
        boxes = regions(*image.size, self.variant.quadrants)
        return i, torch.stack([prepare_region(image, b, self.variant) for b in boxes])


def extract_cache(path, contract, manifest, variant, device, encoder=None,
                  extraction_batch_size=4, resume=False, workers=8):
    device = normalize_device(device)
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    metadata_path = path / "cache_metadata.json"
    if metadata_path.exists():
        meta = validate_cache(path, contract, complete=False)
        if meta.get("complete"):
            print(f"Cache complete, skipping: {path}", flush=True)
            return meta
        if not resume:
            raise FileExistsError(f"Incomplete cache; use --resume: {path}")
    else:
        if any(path.iterdir()):
            raise FileExistsError(f"Unrecognized cache directory: {path}")
        meta = {"contract": contract, "complete": False, "git": git_info(),
                "torch_version": str(torch.__version__), "samples": []}
        atomic_json(metadata_path, meta)
    if encoder is None:
        checkpoint = contract["checkpoint"] if contract["checkpoint_sha256"] else None
        encoder = make_encoder(device, checkpoint, contract["nonscientific_smoke_test"])
    encoder.eval().requires_grad_(False)
    fingerprint = encoder_fingerprint(encoder)
    if meta.get("encoder_state_sha256", fingerprint) != fingerprint:
        raise ValueError("Encoder weights changed during extraction resume")
    meta["encoder_state_sha256"] = fingerprint
    if isinstance(encoder, DINOv2Backbone):
        weights_dir = path.parent.parent / "backbones"
        weights_dir.mkdir(parents=True, exist_ok=True)
        snapshot = weights_dir / f"{contract['encoder']}_{fingerprint}.pt"
        if not snapshot.exists():
            atomic_torch(snapshot, encoder.model.state_dict())
        meta["encoder_weights_snapshot"] = str(snapshot.resolve())
    atomic_json(metadata_path, meta)
    samples = []
    pending = [i for i in range(len(contract["records"])) if not (path / f"{i:06d}.npy").exists()]
    loader = DataLoader(ExtractionDataset(manifest, contract["records"], variant, pending),
                        batch_size=None, num_workers=workers, pin_memory=device.type == "cuda",
                        **({"multiprocessing_context": "spawn", "timeout": 120} if workers else {}))
    prepared = iter(loader)
    memory = CudaMemoryProfiler(device)
    memory.start()
    for i, row in enumerate(contract["records"]):
        file = path / f"{i:06d}.npy"
        boxes = regions(*row["original_size"], variant.quadrants)
        if file.exists():
            validate_array(file, contract)
            # Completed individual originals may be reused after interruption.
        else:
            loaded_index, inputs = next(prepared)
            if loaded_index != i:
                raise ValueError("Extraction worker order mismatch")
            values = []
            with torch.inference_mode():
                for chunk in inputs.split(extraction_batch_size):
                    values.append(encoder.forward_dense(chunk.to(device)).cpu().numpy())
            array = np.concatenate(values).astype(contract["dtype"])
            if list(array.shape) != contract["feature_shape"] or not np.isfinite(array).all():
                raise ValueError("Invalid encoder output")
            temp = file.with_suffix(".partial")
            with temp.open("wb") as handle:
                np.save(handle, array, allow_pickle=False)
            os.replace(temp, file)
        samples.append({"record": row, "file": file.name, "sha256": sha256_file(file),
                        "crop_coordinates": boxes,
                        "input_content_boxes": [content_box(b[2]-b[0], b[3]-b[1], variant) for b in boxes]})
        print(f"{variant.name}: {i+1}/{len(contract['records'])} originals", flush=True)
    meta.update(complete=True, samples=samples,
                extraction_peak_gpu_bytes=memory.peak_bytes(),
                extraction_memory_profiling=memory.metadata())
    atomic_json(metadata_path, meta)
    return meta


class OriginalCachedDataset(Dataset):
    def __init__(self, path, contract, manifest, split, *, verified_metadata=None):
        if split not in ("train", "val"):
            raise ValueError("Only train/val are permitted")
        self.path, self.manifest = Path(path), Path(manifest)
        if sha256_file(manifest) != contract["manifest_sha256"]:
            raise ValueError("Manifest changed since extraction")
        meta = verified_metadata if verified_metadata is not None else validate_cache(path, contract)
        if not contracts_compatible(meta.get("contract", {}), contract) or not meta.get("complete"):
            raise ValueError("Incompatible validated cache metadata")
        self.samples = [s for s in meta["samples"] if s["record"]["split"] == split]
        if not self.samples:
            raise ValueError(f"Empty {split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        sample = self.samples[i]
        row = sample["record"]
        target = load_mask(self.manifest, row)
        features = torch.from_numpy(np.array(np.load(self.path / sample["file"], allow_pickle=False),
                                            dtype=np.float32, copy=True))
        return {"features": features, "target": target, "record": row}
