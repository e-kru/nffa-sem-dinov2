"""Reconstructed global cosine-similarity audit for cleaned V3 images.

Reads V3 manifest and frozen DINOv2 ViT-S/14 embeddings previously created
with crop_metadata_strip (preprocessing_v2), Resize(256), CenterCrop(224).
Builds top-20 nearest neighbors and all undirected pairs with cosine >= 0.995.
No images are changed and no preprocessing is rerun. Does not overwrite originals.
Original CUDA audit code was only partially recovered, so this is NOT claimed
as its original implementation. Output numerical parity must be checked.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

V2 = Path("outputs/dataset_v2")
V3 = Path("outputs/dataset_v3")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.995)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with (V3 / "manifest.csv").open(newline="") as f:
        included = {
            r["image_id"]: r for r in csv.DictReader(f) if r["included"] == "1"
        }
    ids = sorted(included)
    assert len(ids) == 18286
    idx = {p: i for i, p in enumerate(ids)}
    X = np.zeros((len(ids), 384), dtype=np.float32)
    filled = np.zeros(len(ids), dtype=bool)
    files = sorted((V2 / "features").glob("*.npz"))
    assert files
    for file in files:
        with np.load(file, allow_pickle=True) as data:
            for path, emb in zip(data["paths"], data["X"]):
                path = str(path)
                if "/nffa_sem_100/" in path:
                    path = path.split("/nffa_sem_100/", 1)[1]
                if path in idx:
                    i = idx[path]
                    if filled[i]:
                        raise ValueError(f"Duplicate feature for {path}")
                    X[i] = emb
                    filled[i] = True
    assert filled.all()

    embeddings = F.normalize(torch.from_numpy(X).to(args.device), dim=1)
    n = len(ids)
    neighbor_csv = args.output_dir / "dinov2_neighbors_v3.csv"
    high_csv = args.output_dir / "dinov2_high_similarity_v3.csv"
    n_high = 0
    with neighbor_csv.open("w", newline="") as nf, high_csv.open("w", newline="") as hf:
        neighbor_writer = csv.writer(nf)
        high_writer = csv.writer(hf)
        neighbor_writer.writerow(["image_id", "neighbor_id", "rank", "cosine_similarity"])
        high_writer.writerow(["image_a", "image_b", "cosine_similarity"])
        for start in range(0, n, args.batch_size):
            end = min(n, start + args.batch_size)
            sim = embeddings[start:end] @ embeddings.T
            values, indices = sim.topk(k=args.top_k + 1, dim=1)
            values = values.cpu().numpy()
            indices = indices.cpu().numpy()
            for local_i, i in enumerate(range(start, end)):
                rank = 0
                for j, value in zip(indices[local_i], values[local_i]):
                    if int(j) == i:
                        continue
                    rank += 1
                    neighbor_writer.writerow([ids[i], ids[int(j)], rank, round(float(value), 8)])
                    if rank == args.top_k:
                        break
            # Emit each high-similarity pair once (i < j), even if >top-k.
            for local_i, i in enumerate(range(start, end)):
                matched = torch.where(sim[local_i, i + 1:] >= args.threshold)[0]
                for offset in matched.cpu().tolist():
                    j = i + 1 + offset
                    high_writer.writerow([ids[i], ids[j], round(float(sim[local_i, j].item()), 8)])
                    n_high += 1
            print(f"Processed {end}/{n} images", flush=True)

    print("High-similarity undirected pairs:", n_high)
    print("Original count was 338; verify any difference before using reconstructed output.")
    print("Saved:", neighbor_csv, high_csv)


if __name__ == "__main__":
    main()
