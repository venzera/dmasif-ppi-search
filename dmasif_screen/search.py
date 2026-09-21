from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np

from .io import interface_mask, save_json


def load_targets(directory: Path, site_threshold: float):
    emb1_parts: List[np.ndarray] = []
    emb2_parts: List[np.ndarray] = []
    target_ids: List[str] = []
    for path in sorted(directory.glob("*.npz")):
        with np.load(str(path)) as data:
            mask = interface_mask(data, site_threshold)
            if not mask.any():
                continue
            emb1 = np.asarray(data["embedding_1"][mask], dtype=np.float32)
            emb2 = np.asarray(data["embedding_2"][mask], dtype=np.float32)
        emb1_parts.append(emb1)
        emb2_parts.append(emb2)
        target_ids.extend([path.stem] * len(emb1))
    if not emb1_parts:
        raise SystemExit("no target points survived filtering")
    return np.concatenate(emb1_parts), np.concatenate(emb2_parts), np.asarray(target_ids)


def background_stats(
    query_files: List[Path],
    target_emb1: np.ndarray,
    target_emb2: np.ndarray,
    site_threshold: float,
    n_samples: int,
    rng,
) -> dict:
    chosen = rng.choice(query_files, size=min(50, len(query_files)), replace=False)
    q1_parts: List[np.ndarray] = []
    q2_parts: List[np.ndarray] = []
    for path in chosen:
        with np.load(str(path)) as data:
            mask = interface_mask(data, site_threshold)
            if mask.any():
                q1_parts.append(np.asarray(data["embedding_1"][mask], dtype=np.float32))
                q2_parts.append(np.asarray(data["embedding_2"][mask], dtype=np.float32))
    if not q1_parts:
        raise SystemExit("no sampled query points survived filtering")
    q1 = np.concatenate(q1_parts)
    q2 = np.concatenate(q2_parts)
    qi = rng.integers(0, len(q1), size=n_samples)
    ti = rng.integers(0, len(target_emb1), size=n_samples)
    score_a = np.einsum("ij,ij->i", q1[qi], target_emb2[ti])
    score_b = np.einsum("ij,ij->i", q2[qi], target_emb1[ti])
    std_a = float(score_a.std())
    std_b = float(score_b.std())
    if not np.isfinite(std_a) or not np.isfinite(std_b) or std_a <= 0 or std_b <= 0:
        raise ValueError(
            "background descriptor standard deviation must be finite and positive"
        )
    return {
        "mean_a": float(score_a.mean()),
        "std_a": std_a,
        "mean_b": float(score_b.mean()),
        "std_b": std_b,
        "site_threshold": site_threshold,
        "n_samples": n_samples,
    }


def make_index(vectors: np.ndarray, gpu: bool):
    index = faiss.IndexFlatIP(vectors.shape[1])
    if gpu:
        resources = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(resources, 0, index)
    index.add(vectors)
    return index


def search_query(
    path: Path,
    index_emb2,
    index_emb1,
    target_ids: np.ndarray,
    stats: dict,
    site_threshold: float,
    topk: int,
    z_threshold: float,
) -> List[dict]:
    with np.load(str(path)) as data:
        mask = interface_mask(data, site_threshold)
        q1 = np.asarray(data["embedding_1"][mask], dtype=np.float32)
        q2 = np.asarray(data["embedding_2"][mask], dtype=np.float32)
    if not len(q1):
        return []
    k = min(topk, len(target_ids))
    dot_a, idx_a = index_emb2.search(q1, k)
    dot_b, idx_b = index_emb1.search(q2, k)
    z_a = (dot_a - stats["mean_a"]) / stats["std_a"]
    z_b = (dot_b - stats["mean_b"]) / stats["std_b"]
    aggregate: Dict[str, dict] = {}
    for z_values, dot_values, indices in ((z_a, dot_a, idx_a), (z_b, dot_b, idx_b)):
        selected = z_values > z_threshold
        for z, dot, index in zip(
            z_values[selected], dot_values[selected], indices[selected]
        ):
            target_id = str(target_ids[index])
            row = aggregate.setdefault(
                target_id,
                {"target_id": target_id, "max_z": float("-inf"), "max_dot": float("-inf"), "n_matches": 0},
            )
            row["n_matches"] += 1
            if z > row["max_z"]:
                row["max_z"] = float(z)
                row["max_dot"] = float(dot)
    for row in aggregate.values():
        row["n_matches_frac"] = row["n_matches"] / float(len(q1))
        row["sigmoid_logit"] = float(
            1.0 / (1.0 + np.exp(-np.clip(row["max_dot"], -80, 80)))
        )
    return sorted(
        aggregate.values(), key=lambda row: (row["max_z"], row["n_matches"]), reverse=True
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="FAISS retrieval over dMaSIF embeddings.")
    parser.add_argument("--query-embeddings", required=True, type=Path)
    parser.add_argument("--target-embeddings", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--site-threshold", type=float, default=0.5)
    parser.add_argument("--z-threshold", type=float, default=3.0)
    parser.add_argument("--topk-points", type=int, default=32)
    parser.add_argument(
        "--max-targets-per-query",
        type=int,
        default=500,
        help="retrieval rows retained per query; 0 keeps all",
    )
    parser.add_argument("--background-samples", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", action="store_true", help="use a FAISS GPU index")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    query_files = sorted(args.query_embeddings.glob("*.npz"))
    if not query_files:
        raise SystemExit("no query embeddings")
    target_emb1, target_emb2, target_ids = load_targets(
        args.target_embeddings, args.site_threshold
    )
    rng = np.random.default_rng(args.seed)
    stats = background_stats(
        query_files,
        target_emb1,
        target_emb2,
        args.site_threshold,
        args.background_samples,
        rng,
    )
    save_json(args.output_dir / "background.json", stats)
    index_emb2 = make_index(target_emb2, args.gpu)
    index_emb1 = make_index(target_emb1, args.gpu)
    columns = (
        "target_id",
        "max_z",
        "max_dot",
        "sigmoid_logit",
        "n_matches",
        "n_matches_frac",
    )
    for number, path in enumerate(query_files, start=1):
        output = args.output_dir / (path.stem + ".tsv")
        if output.exists():
            continue
        rows = search_query(
            path,
            index_emb2,
            index_emb1,
            target_ids,
            stats,
            args.site_threshold,
            args.topk_points,
            args.z_threshold,
        )
        if args.max_targets_per_query:
            rows = rows[: args.max_targets_per_query]
        with output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        print("[{}/{}] {}: {} targets".format(number, len(query_files), path.stem, len(rows)))


if __name__ == "__main__":
    main()
