from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.spatial import cKDTree

from .embed import build_batch, load_atoms
from .io import atomic_savez, read_manifest


def site_model_args(device: str) -> SimpleNamespace:
    return SimpleNamespace(
        embedding_layer="dMaSIF",
        search=False,
        site=True,
        single_protein=True,
        use_mesh=False,
        n_layers=3,
        radius=12.0,
        emb_dims=16,
        in_channels=16,
        atom_dims=6,
        orientation_units=16,
        unet_hidden_channels=8,
        post_units=8,
        resolution=1.0,
        distance=1.05,
        variance=0.1,
        sup_sampling=100,
        curvature_scales=[1.0, 2.0, 3.0, 5.0, 10.0],
        dropout=0.0,
        no_chem=False,
        no_geom=False,
        k=40,
        device=device,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Attach dMaSIF-site probabilities to existing search embeddings."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--embedding-dir", required=True, type=Path)
    parser.add_argument("--dmasif-repo", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("--shard-index must be in [0, --num-shards)")

    sys.path.insert(0, str(args.dmasif_repo.resolve()))
    from model import dMaSIF  # type: ignore

    net = dMaSIF(site_model_args(args.device))
    checkpoint = torch.load(str(args.checkpoint), map_location=args.device)
    net.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=True)
    net = net.to(args.device).eval()
    rows = read_manifest(args.manifest)[args.shard_index :: args.num_shards]
    failures = args.embedding_dir / "_site_failures.tsv"

    with failures.open("a") as failure_handle:
        for offset in range(0, len(rows), args.batch_size):
            structures = []
            for row in rows[offset : offset + args.batch_size]:
                npz_path = args.embedding_dir / (row["id"] + ".npz")
                if not npz_path.exists():
                    failure_handle.write("{}\tmissing_embedding\n".format(row["id"]))
                    continue
                with np.load(str(npz_path)) as existing:
                    if "site_score" in existing:
                        continue
                try:
                    xyz, types = load_atoms(Path(row["path"]))
                    structures.append((row["id"], xyz, types))
                except Exception as exc:
                    failure_handle.write("{}\tload\t{}\n".format(row["id"], exc))
            if not structures:
                continue

            with torch.no_grad():
                output = net(build_batch(structures, args.device), P2=None)["P1"]
            site_xyz = output["xyz"].detach().cpu().numpy()
            site_score = (
                torch.sigmoid(output["iface_preds"].squeeze(-1)).detach().cpu().numpy()
            )
            batch = output["batch"].detach().cpu().numpy()

            for i, (item_id, _, _) in enumerate(structures):
                mask = batch == i
                npz_path = args.embedding_dir / (item_id + ".npz")
                arrays = dict(np.load(str(npz_path)))
                distances, indices = cKDTree(site_xyz[mask]).query(arrays["xyz"], k=1)
                arrays["site_score"] = site_score[mask][indices].astype(np.float32)
                arrays["site_nn_dist"] = distances.astype(np.float32)
                atomic_savez(npz_path, **arrays)
            print("[{}/{}]".format(min(offset + args.batch_size, len(rows)), len(rows)))


if __name__ == "__main__":
    main()
