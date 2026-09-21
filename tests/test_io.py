import tempfile
import unittest
from pathlib import Path

import numpy as np

from dmasif_screen.io import discover_structures, interface_mask, read_fasta


class IoTests(unittest.TestCase):
    def test_read_fasta_uses_first_header_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.fasta"
            path.write_text(">query_1 description\nACD\nEF\n>target_2\nGG\n")
            self.assertEqual(
                read_fasta(path), {"query_1": "ACDEF", "target_2": "GG"}
            )

    def test_duplicate_fasta_identifier_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.fasta"
            path.write_text(">same\nAA\n>same\nBB\n")
            with self.assertRaises(ValueError):
                read_fasta(path)

    def test_discover_structures_is_recursive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "nested").mkdir()
            (root / "query.pdb").write_text("")
            (root / "nested" / "target.cif").write_text("")
            self.assertEqual(set(discover_structures(root)), {"query", "target"})

    def test_interface_mask_requires_scores_when_enabled(self):
        data = {"xyz": np.zeros((2, 3), dtype=np.float32)}
        np.testing.assert_array_equal(interface_mask(data, 0), [True, True])
        with self.assertRaises(ValueError):
            interface_mask(data, 0.5)


if __name__ == "__main__":
    unittest.main()
