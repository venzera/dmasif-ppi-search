import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dmasif_screen.select_pairs import main


class SelectPairsTests(unittest.TestCase):
    def test_selects_highest_rrf_pair_and_exports_fasta(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rescored = root / "rescored"
            rescored.mkdir()
            query_fasta = root / "queries.fasta"
            target_fasta = root / "targets.fasta"
            output = root / "selected"
            query_fasta.write_text(">q1\nAAAA\n")
            target_fasta.write_text(">t1\nBBBB\n>t2\nCCCC\n")
            columns = [
                "target_id",
                "status",
                "max_z",
                "n_matches",
                "n_correspondences",
                "n_inliers",
                "fitness",
                "inlier_rmse",
                "ransac_score",
            ]
            with (rescored / "q1.tsv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
                writer.writeheader()
                writer.writerow(
                    dict(
                        target_id="t1",
                        status="ok",
                        max_z=8,
                        n_matches=5,
                        n_correspondences=8,
                        n_inliers=5,
                        fitness=0.5,
                        inlier_rmse=0.5,
                        ransac_score=3.3,
                    )
                )
                writer.writerow(
                    dict(
                        target_id="t2",
                        status="ok",
                        max_z=4,
                        n_matches=3,
                        n_correspondences=5,
                        n_inliers=3,
                        fitness=0.3,
                        inlier_rmse=0.8,
                        ransac_score=1.7,
                    )
                )
            argv = [
                "select_pairs",
                "--rescored-dir",
                str(rescored),
                "--query-fasta",
                str(query_fasta),
                "--target-fasta",
                str(target_fasta),
                "--output-dir",
                str(output),
                "--top-n-per-query",
                "1",
            ]
            with patch.object(sys, "argv", argv):
                main()
            with (output / "pairs.tsv").open() as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual([row["pair_id"] for row in rows], ["q1__t1"])
            self.assertIn(">q1__t1|A|q1", (output / "pairs.fasta").read_text())


if __name__ == "__main__":
    unittest.main()
