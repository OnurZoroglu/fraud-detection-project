"""Checks that SQL features and PCA components are joined on the right raw transaction.

Run from the project root:
    python -m unittest discover -s tests

The integration tests need data/creditcard.csv, data/features.csv and
data/features_with_graph_ids.csv (gitignored) and are skipped without them.
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data_prep import (  # noqa: E402
    FEATURES_PATH,
    RAW_DATA_PATH,
    check_alignment,
    load_dataset,
    load_dataset_with_graph_features,
    load_raw,
)

GRAPH_IDS_PATH = "data/features_with_graph_ids.csv"
HAVE_DATA = all(os.path.exists(p) for p in (RAW_DATA_PATH, FEATURES_PATH))
HAVE_GRAPH_DATA = HAVE_DATA and all(os.path.exists(p) for p in (GRAPH_IDS_PATH, "data/graph_features.csv"))


def _toy_raw():
    # Two transactions share time_seconds=0, like most rows in the real dataset.
    return pd.DataFrame({
        "transaction_id": [1, 2, 3],
        "time_seconds": [0.0, 0.0, 1.0],
        "amount": [10.0, 99.0, 5.0],
        "is_fraud": [0, 1, 0],
        "v1": [0.1, 0.2, 0.3],
    })


class CheckAlignmentTest(unittest.TestCase):
    def test_accepts_correct_join(self):
        raw = _toy_raw()
        check_alignment(raw.copy(), raw, ["v1"])

    def test_rejects_rows_swapped_within_same_timestamp(self):
        raw = _toy_raw()
        df = raw.copy()
        df.loc[[0, 1], ["amount", "is_fraud"]] = df.loc[[1, 0], ["amount", "is_fraud"]].to_numpy()
        with self.assertRaisesRegex(ValueError, "amount mismatch.*label mismatch"):
            check_alignment(df, raw, ["v1"])

    def test_rejects_missing_pca_components(self):
        raw = _toy_raw()
        df = raw.copy()
        df.loc[2, "v1"] = np.nan
        with self.assertRaisesRegex(ValueError, "missing PCA components"):
            check_alignment(df, raw, ["v1"])

    def test_rejects_dropped_rows(self):
        raw = _toy_raw()
        with self.assertRaisesRegex(ValueError, "expected 3 rows"):
            check_alignment(raw.iloc[:2].copy(), raw, ["v1"])


@unittest.skipUnless(HAVE_DATA, "raw/feature CSVs not present")
class RealDataAlignmentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw, cls.v_columns = load_raw()

    def assert_fully_aligned(self, df):
        """Every row's PCA components must equal those of its own raw transaction."""
        self.assertEqual(len(df), len(self.raw))
        self.assertTrue(df["transaction_id"].is_unique)
        ref = self.raw.set_index("transaction_id").loc[df["transaction_id"]]
        np.testing.assert_array_equal(df[self.v_columns].to_numpy(), ref[self.v_columns].to_numpy())
        np.testing.assert_allclose(df["amount"].to_numpy(), ref["amount"].to_numpy())
        np.testing.assert_array_equal(df["is_fraud"].to_numpy(), ref["is_fraud"].to_numpy())

    def test_features_csv_ids_are_raw_row_numbers(self):
        features = pd.read_csv(FEATURES_PATH)
        self.assertEqual(sorted(features["transaction_id"]), list(range(1, len(self.raw) + 1)))

    def test_load_dataset(self):
        df, _ = load_dataset()
        self.assert_fully_aligned(df)

    @unittest.skipUnless(HAVE_GRAPH_DATA, "graph CSVs not present")
    def test_load_dataset_with_graph_features(self):
        df, _ = load_dataset_with_graph_features()
        self.assert_fully_aligned(df)


if __name__ == "__main__":
    unittest.main()
