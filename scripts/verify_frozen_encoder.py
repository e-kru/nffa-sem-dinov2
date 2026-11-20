"""Offline CPU worker for independent pinned/legacy backbone equivalence checks.

Run each source in a fresh process: Python's dinov2 import cache would otherwise
make a comparison between two different source checkouts meaningless.
"""
import argparse
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", choices=("pinned", "legacy"), required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--legacy-hub-dir", type=Path, required=True)
    p.add_argument("--encoder-name", choices=("dinov2_vits14", "dinov2_vitl14"), default="dinov2_vits14")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    os.environ["XFORMERS_DISABLED"] = "1"
    import sys
    import torch
    from src.dinov2_encoder import load_dinov2_checkpoint
    from src.downstream.pinned_encoder import get_pinned_dinov2
    from src.aachen_ablation.cache import encoder_fingerprint
    from src.aachen_ablation.data import sha256_file
    torch.set_num_threads(1)
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.source == "pinned":
        model = get_pinned_dinov2(args.checkpoint, args.encoder_name)
    else:
        # The cached Hub code used by get_dinov2(), without network or downloads.
        sys.path.insert(0, str(args.legacy_hub_dir.resolve()))
        model = torch.hub.load(str(args.legacy_hub_dir.resolve()), args.encoder_name,
                               source="local", pretrained=False)
        load_dinov2_checkpoint(model, args.checkpoint)
        model.requires_grad_(False).eval()
    import dinov2
    source = Path(dinov2.__file__).resolve().parent
    with torch.inference_mode():
        generator = torch.Generator().manual_seed(73)
        global_features = model(torch.randn(1, 3, 224, 224, generator=generator))
        dense = model.get_intermediate_layers(
            torch.randn(1, 3, 672, 896, generator=generator), n=1, reshape=True, norm=True)[0]
    torch.save({"global": global_features, "dense": dense,
                "encoder_state_sha256": encoder_fingerprint(model),
                "frozen": not any(v.requires_grad for v in model.parameters()),
                "eval": not model.training,
                "source": str(source), "source_files": {
                    str(f.relative_to(source)): sha256_file(f) for f in sorted(source.rglob("*.py"))}}, args.output)


if __name__ == "__main__":
    main()
