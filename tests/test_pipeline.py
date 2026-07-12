import unittest

import numpy as np
import pandas as pd
import torch

from src.models.components.adaptive_patch import AdaptivePatchTransformer
from src.models.transformer import PositionalEncoding, TransformerModel
from src.windows import prepare_windows


class WindowProtocolTests(unittest.TestCase):
    def test_targets_are_temporally_isolated(self):
        for horizon in (90, 365):
            prepared = prepare_windows("data/processed", 90, horizon, 1)
            split = prepared.split
            train_target_end = split.train_starts[-1] + split.input_len + split.horizon
            val_target_start = split.val_starts[0] + split.input_len
            val_target_end = val_target_start + split.horizon
            test_target_start = split.test_starts[0] + split.input_len

            self.assertLessEqual(train_target_end, val_target_start)
            self.assertLessEqual(val_target_end, test_target_start)

    def test_processed_target_is_kwh_and_has_no_missing_days(self):
        frame = pd.concat(
            [pd.read_csv("data/processed/train.csv"), pd.read_csv("data/processed/test.csv")],
            ignore_index=True,
        )
        dates = pd.to_datetime(frame["Date"])
        self.assertEqual(len(frame), (dates.max() - dates.min()).days + 1)
        self.assertFalse(frame.isna().any().any())
        self.assertGreater(frame["Global_active_power"].min(), 0)
        self.assertLess(frame["Global_active_power"].mean(), 100)


class ModelTests(unittest.TestCase):
    def test_transformer_applies_position_once(self):
        model = TransformerModel(n_features=2, d_model=4, n_heads=2, n_layers=1, horizon=3)
        x = torch.randn(1, 5, 2)
        embedded = model.embed(x)
        expected = PositionalEncoding(4)(embedded)
        self.assertTrue(torch.allclose(model.pos(embedded), expected))

    def test_adaptive_patch_preserves_expected_shapes(self):
        module = AdaptivePatchTransformer(d_model=6, n_heads=2, n_layers=1)
        x = torch.randn(2, 30, 6)
        self.assertEqual(module(x, patch_size=1).shape, (2, 30, 6))
        self.assertEqual(module(x, patch_size=3).shape, (2, 10, 6))


if __name__ == "__main__":
    unittest.main()
