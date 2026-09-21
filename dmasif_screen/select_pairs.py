from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List

from .io import read_fasta, write_tsv


COLUMNS = (
    "pair_id",
    "query_id",
    "target_id",
    "rank_query",
    "selection_score",
    "max_z",
    "n_matches",
    "n_correspondences",
    "n_ransac_inliers",
    "n_icp_correspondences",
    "fitness",
    "inlier_rmse",
    "ransac_score",
    "query_length",
    "target_length",
    "query_sequence",
    "target_sequence",
)


def number(value, default=float("-inf")):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter geometrically supported pairs and export top-N sequences."
    )
    parser.add_argument("--rescored-dir", required=True, type=Path)
    parser.add_argument("--query-fasta", required=True, type=Path)
    parser.add_argument("--target-fasta", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--top-n-per-query", type=int, default=1)
    parser.add_argument("--top-n-total", type=int, default=0, help="0 keeps all selected queries")
    parser.add_argument(
        "--rank-by",
        choices=("ransac", "dmasif", "rrf"),
        default="rrf",
        help="RRF combines within-query dMaSIF and RANSAC ranks",
    )
    parser.add_argument("--rrf-k", type=float, default=10.0)
    parser.add_argument("--min-inliers", type=int, default=1)
    parser.add_argument("--min-fitness", type=float, default=0.0)
    parser.add_argument("--max-rmsd", type=float, default=float("inf"))
    args = parser.parse_args()

    query_sequences = read_fasta(args.query_fasta)
    target_sequences = read_fasta(args.target_fasta)
    selected: List[dict] = []
    for path in sorted(args.rescored_dir.glob("*.tsv")):
        query_id = path.stem
        with path.open(newline="") as handle:
            rows = [
                row
                for row in csv.DictReader(handle, delimiter="\t")
                if row.get("status") == "ok"
                and int(
                    number(
                        row.get("n_ransac_inliers", row.get("n_inliers")),
                        0,
                    )
                )
                >= args.min_inliers
                and number(row.get("fitness"), 0) >= args.min_fitness
                and number(row.get("inlier_rmse")) <= args.max_rmsd
            ]
        dmasif_order = sorted(rows, key=lambda row: number(row.get("max_z")), reverse=True)
        ransac_order = sorted(
            rows, key=lambda row: number(row.get("ransac_score")), reverse=True
        )
        dmasif_rank = {row["target_id"]: rank for rank, row in enumerate(dmasif_order, 1)}
        ransac_rank = {row["target_id"]: rank for rank, row in enumerate(ransac_order, 1)}
        for row in rows:
            if args.rank_by == "ransac":
                row["_selection_score"] = number(row.get("ransac_score"))
            elif args.rank_by == "dmasif":
                row["_selection_score"] = number(row.get("max_z"))
            else:
                row["_selection_score"] = (
                    1.0 / (args.rrf_k + dmasif_rank[row["target_id"]])
                    + 1.0 / (args.rrf_k + ransac_rank[row["target_id"]])
                )
        rows.sort(key=lambda row: row["_selection_score"], reverse=True)
        for rank, row in enumerate(rows[: args.top_n_per_query], start=1):
            target_id = row["target_id"]
            if query_id not in query_sequences or target_id not in target_sequences:
                continue
            selected.append(
                {
                    "pair_id": "{}__{}".format(query_id, target_id),
                    "query_id": query_id,
                    "target_id": target_id,
                    "rank_query": rank,
                    "selection_score": "{:.8g}".format(row["_selection_score"]),
                    "max_z": row.get("max_z", ""),
                    "n_matches": row.get("n_matches", ""),
                    "n_correspondences": row.get("n_correspondences", ""),
                    "n_ransac_inliers": row.get(
                        "n_ransac_inliers", row.get("n_inliers", "")
                    ),
                    "n_icp_correspondences": row.get(
                        "n_icp_correspondences", row.get("n_inliers", "")
                    ),
                    "fitness": row.get("fitness", ""),
                    "inlier_rmse": row.get("inlier_rmse", ""),
                    "ransac_score": row.get("ransac_score", ""),
                    "query_length": len(query_sequences[query_id]),
                    "target_length": len(target_sequences[target_id]),
                    "query_sequence": query_sequences[query_id],
                    "target_sequence": target_sequences[target_id],
                }
            )
    selected.sort(key=lambda row: number(row["selection_score"]), reverse=True)
    if args.top_n_total:
        selected = selected[: args.top_n_total]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(args.output_dir / "pairs.tsv", selected, COLUMNS)

    fasta_dir = args.output_dir / "pair_fastas"
    fasta_dir.mkdir(exist_ok=True)
    with (args.output_dir / "pairs.fasta").open("w") as combined:
        for row in selected:
            pair_id = row["pair_id"]
            text = (
                ">{}|A|{}\n{}\n>{}|B|{}\n{}\n".format(
                    pair_id,
                    row["query_id"],
                    row["query_sequence"],
                    pair_id,
                    row["target_id"],
                    row["target_sequence"],
                )
            )
            combined.write(text)
            (fasta_dir / (safe_name(pair_id) + ".fasta")).write_text(text)
    print("selected {} pairs -> {}".format(len(selected), args.output_dir))


if __name__ == "__main__":
    main()
