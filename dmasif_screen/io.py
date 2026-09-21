from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple

import numpy as np


STRUCTURE_SUFFIXES = {".pdb", ".cif", ".mmcif", ".ent"}


def read_fasta(path: Path) -> Dict[str, str]:
    records: Dict[str, str] = {}
    current_id = None
    chunks: List[str] = []

    def store() -> None:
        if current_id is None:
            return
        sequence = "".join(chunks).replace(" ", "").upper()
        if not sequence:
            raise ValueError("empty FASTA sequence for {!r}".format(current_id))
        if current_id in records:
            raise ValueError("duplicate FASTA identifier {!r}".format(current_id))
        records[current_id] = sequence

    with path.open() as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                store()
                current_id = line[1:].split()[0]
                if not current_id:
                    raise ValueError("empty FASTA header in {}".format(path))
                chunks = []
            elif current_id is None:
                raise ValueError("sequence before first FASTA header in {}".format(path))
            else:
                chunks.append(line)
    store()
    if not records:
        raise ValueError("no FASTA records in {}".format(path))
    return records


def discover_structures(root: Path, recursive: bool = True) -> Dict[str, Path]:
    iterator = root.rglob("*") if recursive else root.glob("*")
    structures: Dict[str, Path] = {}
    for path in sorted(iterator):
        if not path.is_file() or path.suffix.lower() not in STRUCTURE_SUFFIXES:
            continue
        structure_id = path.stem
        if structure_id in structures:
            raise ValueError(
                "duplicate structure identifier {!r}: {} and {}".format(
                    structure_id, structures[structure_id], path
                )
            )
        structures[structure_id] = path.resolve()
    if not structures:
        raise ValueError("no PDB/mmCIF structures under {}".format(root))
    return structures


def read_manifest(path: Path) -> List[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {"id", "path"}
    if not rows:
        raise ValueError("empty manifest: {}".format(path))
    if not required.issubset(rows[0]):
        raise ValueError("{} must contain columns: id, path".format(path))
    return rows


def write_tsv(path: Path, rows: Iterable[Mapping[str, object]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def interface_mask(data: Mapping[str, np.ndarray], threshold: float) -> np.ndarray:
    n_points = data["xyz"].shape[0]
    if threshold <= 0:
        return np.ones(n_points, dtype=bool)
    if "site_score" not in data:
        raise ValueError(
            "embedding has no site_score; run score-sites or set --site-threshold 0"
        )
    return np.asarray(data["site_score"]) > threshold


def load_embedding(path: Path, threshold: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(str(path)) as data:
        mask = interface_mask(data, threshold)
        return (
            np.asarray(data["xyz"][mask], dtype=np.float32),
            np.asarray(data["embedding_1"][mask], dtype=np.float32),
            np.asarray(data["embedding_2"][mask], dtype=np.float32),
        )


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(
        prefix=path.stem + ".", suffix=".npz", dir=str(path.parent), delete=False
    )
    temporary.close()
    try:
        np.savez_compressed(temporary.name, **arrays)
        os.replace(temporary.name, str(path))
    finally:
        if os.path.exists(temporary.name):
            os.unlink(temporary.name)
