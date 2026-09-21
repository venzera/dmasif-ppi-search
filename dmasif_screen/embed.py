from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import torch
from Bio.PDB import MMCIFParser, PDBParser

from .io import atomic_savez, read_manifest


ATOM_TYPES = {"C": 0, "H": 1, "O": 2, "N": 3, "S": 4, "SE": 5}


def model_args(device: str) -> SimpleNamespace:
    return SimpleNamespace(
        embedding_layer="dMaSIF",
        search=True,
        site=False,
        single_protein=False,
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
        sup_sampling=20,
        curvature_scales=[1.0, 2.0, 3.0, 5.0, 10.0],
        dropout=0.0,
        no_chem=False,
        no_geom=False,
        k=40,
        device=device,
    )


def load_atoms(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    suffix = path.suffix.lower()
    parser = MMCIFParser(QUIET=True) if suffix in {".cif", ".mmcif"} else PDBParser(QUIET=True)
    structure = parser.get_structure("structure", str(path))
    coords: List[np.ndarray] = []
    atom_indices: List[int] = []
    for atom in structure.get_atoms():
        element = (atom.element or "").strip().upper()
        if element not in ATOM_TYPES:
            continue
        coords.append(atom.get_coord())
        atom_indices.append(ATOM_TYPES[element])
    if not coords:
        raise ValueError("no supported atoms in {}".format(path))
    xyz = np.asarray(coords, dtype=np.float32)
    types = np.zeros((len(atom_indices), len(ATOM_TYPES)), dtype=np.float32)
    types[np.arange(len(atom_indices)), atom_indices] = 1.0
    return xyz, types


def build_batch(structures, device: str):
    xyz = np.concatenate([item[1] for item in structures], axis=0)
    types = np.concatenate([item[2] for item in structures], axis=0)
    batch = np.concatenate(
        [np.full(len(item[1]), i, dtype=np.int64) for i, item in enumerate(structures)]
    )
    atoms = torch.from_numpy(xyz).to(device)
    return {
        "atoms": atoms,
        "atom_xyz": atoms,
        "atomtypes": torch.from_numpy(types).to(device),
        "batch_atoms": torch.from_numpy(batch).to(device),
    }


def run_batch(net, structures, device: str, max_points: int, min_points: int, rng):
    with torch.no_grad():
        output = net(build_batch(structures, device), P2=None)["P1"]
    xyz = output["xyz"].detach().cpu().numpy()
    emb1 = output["embedding_1"].detach().cpu().numpy()
    emb2 = output["embedding_2"].detach().cpu().numpy()
    batch = output["batch"].detach().cpu().numpy()
    results = {}
    for i, (item_id, _, _) in enumerate(structures):
        indices = np.flatnonzero(batch == i)
        if len(indices) < min_points:
            results[item_id] = None
            continue
        if max_points and len(indices) > max_points:
            indices = rng.choice(indices, size=max_points, replace=False)
        results[item_id] = (xyz[indices], emb1[indices], emb2[indices])
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dMaSIF-search surface embeddings.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dmasif-repo", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-points", type=int, default=2000)
    parser.add_argument("--min-points", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("--shard-index must be in [0, --num-shards)")
    sys.path.insert(0, str(args.dmasif_repo.resolve()))
    from model import dMaSIF  # type: ignore

    architecture = model_args(args.device)
    net = dMaSIF(architecture)
    checkpoint = torch.load(str(args.checkpoint), map_location=args.device)
    state = checkpoint.get("model_state_dict", checkpoint)
    net.load_state_dict(state, strict=True)
    net = net.to(args.device).eval()

    rows = read_manifest(args.manifest)[args.shard_index :: args.num_shards]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [row for row in rows if not (args.output_dir / (row["id"] + ".npz")).exists()]
    rng = np.random.default_rng(args.seed)
    failures = args.output_dir / "_embed_failures.tsv"
    start = time.time()

    with failures.open("a") as failure_handle:
        for offset in range(0, len(rows), args.batch_size):
            batch_rows = rows[offset : offset + args.batch_size]
            structures = []
            for row in batch_rows:
                try:
                    xyz, types = load_atoms(Path(row["path"]))
                    structures.append((row["id"], xyz, types))
                except Exception as exc:
                    failure_handle.write("{}\tload\t{}\n".format(row["id"], exc))
            if not structures:
                continue
            try:
                results = run_batch(
                    net, structures, args.device, args.max_points, args.min_points, rng
                )
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                torch.cuda.empty_cache()
                results = {}
                for structure in structures:
                    try:
                        results.update(
                            run_batch(
                                net,
                                [structure],
                                args.device,
                                args.max_points,
                                args.min_points,
                                rng,
                            )
                        )
                    except Exception as single_exc:
                        failure_handle.write(
                            "{}\tforward\t{}\n".format(structure[0], single_exc)
                        )
            for item_id, result in results.items():
                if result is None:
                    failure_handle.write("{}\ttoo_few_points\n".format(item_id))
                    continue
                xyz, emb1, emb2 = result
                atomic_savez(
                    args.output_dir / (item_id + ".npz"),
                    xyz=xyz.astype(np.float32),
                    embedding_1=emb1.astype(np.float32),
                    embedding_2=emb2.astype(np.float32),
                )
            done = min(offset + args.batch_size, len(rows))
            print("[{}/{}] {:.1f}s".format(done, len(rows), time.time() - start), flush=True)


if __name__ == "__main__":
    main()
