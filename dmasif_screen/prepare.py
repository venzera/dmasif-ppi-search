from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from .io import discover_structures, read_fasta, write_tsv


def build_manifest(
    structure_dir: Path,
    fasta_path: Path,
    output_path: Path,
    allow_missing: bool,
    recursive: bool,
) -> None:
    structures = discover_structures(structure_dir, recursive=recursive)
    sequences = read_fasta(fasta_path)

    structure_only = sorted(set(structures) - set(sequences))
    fasta_only = sorted(set(sequences) - set(structures))
    if (structure_only or fasta_only) and not allow_missing:
        raise SystemExit(
            "structure/FASTA identifiers do not match: "
            "{} structure-only, {} FASTA-only. File stems must equal FASTA IDs. "
            "Use --allow-missing to retain their intersection.".format(
                len(structure_only), len(fasta_only)
            )
        )

    shared = sorted(set(structures) & set(sequences))
    if not shared:
        raise SystemExit("no matching structure stems and FASTA identifiers")
    rows: List[Dict[str, object]] = [
        {
            "id": item_id,
            "path": str(structures[item_id]),
            "sequence_length": len(sequences[item_id]),
        }
        for item_id in shared
    ]
    write_tsv(output_path, rows, ("id", "path", "sequence_length"))
    print(
        "{}: {} matched; {} structure-only; {} FASTA-only".format(
            output_path, len(shared), len(structure_only), len(fasta_only)
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate structure/FASTA identifiers and write query/target manifests."
    )
    parser.add_argument("--query-structures", required=True, type=Path)
    parser.add_argument("--target-structures", required=True, type=Path)
    parser.add_argument("--query-fasta", required=True, type=Path)
    parser.add_argument("--target-fasta", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="keep only IDs present in both the structure directory and FASTA",
    )
    parser.add_argument("--no-recursive", action="store_true")
    args = parser.parse_args()

    manifests = args.work_dir / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    build_manifest(
        args.query_structures,
        args.query_fasta,
        manifests / "query.tsv",
        args.allow_missing,
        not args.no_recursive,
    )
    build_manifest(
        args.target_structures,
        args.target_fasta,
        manifests / "target.tsv",
        args.allow_missing,
        not args.no_recursive,
    )


if __name__ == "__main__":
    main()
