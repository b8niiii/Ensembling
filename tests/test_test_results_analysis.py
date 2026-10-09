"""Synthetic checks for whole-bag pairing and micro-F1 aggregation."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from analyze_test_results import (
    bootstrap_metrics, confusion_counts, metrics_from_counts, paired_intervals,
    validate_counts,
)


class PairedBagAnalysisTests(unittest.TestCase):
    def test_multilabel_counts_and_empty_invalid_prediction(self):
        self.assertEqual(confusion_counts(["a", "b", "b"], ["a", "c", "c"]), (1, 1, 1))
        self.assertEqual(confusion_counts(["a", "b"], []), (0, 0, 2))
        self.assertEqual(confusion_counts([], ["a"]), (0, 1, 0))

    def test_micro_f1_is_not_mean_bag_f1(self):
        counts = np.array([[1, 0, 0], [0, 0, 9]])
        micro = metrics_from_counts(counts.sum(axis=0))[2]
        macro = metrics_from_counts(counts)[:, 2].mean()
        self.assertAlmostEqual(micro, 2 / 11)
        self.assertAlmostEqual(macro, 0.5)

    def test_identical_systems_have_exactly_zero_paired_differences(self):
        counts = np.repeat(np.array([[2, 1, 1], [0, 2, 0], [1, 0, 2]])[:, None, :], 3, axis=1)
        report = paired_intervals(counts, replicates=100, seed=7)
        for contrast in report["paired_f1_differences"]:
            self.assertEqual(contrast["ci95_percentage_points"], [0.0, 0.0])
            self.assertFalse(contrast["ci_excludes_zero"])

    def test_same_bag_indices_for_every_system_and_whole_label_counts(self):
        counts = np.array([[[2, 0, 1], [1, 1, 2], [0, 0, 3]],
                           [[0, 1, 0], [0, 0, 0], [0, 2, 0]],
                           [[1, 0, 0], [1, 0, 0], [0, 0, 1]]])
        draws = bootstrap_metrics(counts, replicates=11, seed=18, batch_size=4)
        rng = np.random.default_rng(18)
        expected = []
        for _ in range(11):
            sampled = rng.integers(0, len(counts), size=len(counts))
            expected.append(metrics_from_counts(counts[sampled].sum(axis=0)))
        np.testing.assert_array_equal(draws, expected)

    def test_negative_bags_and_invalid_positive_bags_are_not_dropped(self):
        counts = np.array([[[0, 1, 0], [0, 0, 0], [0, 0, 0]],
                           [[1, 0, 1], [0, 0, 2], [0, 0, 2]]])
        report = paired_intervals(counts, replicates=80, seed=19)
        self.assertEqual(report["bags"], 2)
        self.assertEqual(report["system_intervals"][2]["micro_f1"]["estimate"], 0.0)

    def test_reject_mismatched_gold_support_or_missing_system(self):
        with self.assertRaises(ValueError):
            validate_counts(np.array([[[1, 0, 0], [0, 0, 0], [1, 0, 0]]]))
        with self.assertRaises(ValueError):
            validate_counts(np.ones((4, 2, 3), dtype=int))

    def test_batch_size_does_not_change_replicates(self):
        counts = np.repeat(np.array([[2, 1, 1], [0, 2, 0], [1, 0, 2]])[:, None, :], 3, axis=1)
        np.testing.assert_array_equal(bootstrap_metrics(counts, replicates=25, seed=8, batch_size=1),
                                      bootstrap_metrics(counts, replicates=25, seed=8, batch_size=7))

    def test_zero_denominator_is_finite(self):
        np.testing.assert_array_equal(metrics_from_counts([0, 0, 0]), np.zeros(3))


if __name__ == "__main__":
    unittest.main()
