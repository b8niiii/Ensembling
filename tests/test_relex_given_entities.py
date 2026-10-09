"""Checks for supplied-entity direction, span alignment, and cached-score filtering."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
import torch
from relex_given_entities import align_supplied_sentence, extract_directed_scores, labels_at_threshold


class MinimalWordProcessor:
    """A whitespace splitter exposing the boundary case used by the adapter."""
    config = SimpleNamespace(max_len=100)

    def prepare_batch(self, texts, labels, relations):
        text = texts[0]
        tokens, starts, ends = [], [], []
        position = 0
        for token in text.split():
            start = text.index(token, position)
            tokens.append(token)
            starts.append(start)
            ends.append(start + len(token))
            position = ends[-1]
        return {"valid_to_orig_idx": [0], "tokens": [tokens],
                "start_token_map": [starts], "end_token_map": [ends], "input_x": [{}]}

    def prepare_base_input(self, tokens):
        return [{"tokens": tokens[0]}]


class GivenEntitiesTests(unittest.TestCase):
    def test_directed_scores_do_not_use_stronger_reverse_relation(self):
        output = SimpleNamespace(
            rel_idx=torch.tensor([[[1, 0], [0, 1]]]),
            rel_mask=torch.tensor([[True, True]]),
            rel_logits=torch.tensor([[[8.0, -8.0], [-8.0, 8.0]]]),
        )
        scores = extract_directed_scores(output, {1: "parent", 2: "child"},
                                         {"parent": "parent_label", "child": "child_label"})
        self.assertEqual(labels_at_threshold(scores, .5), ["child_label"])

    def test_multiple_relations_and_strict_threshold_can_be_rescored(self):
        saved = {"capital": .8, "contains": .6, "other": .5}
        self.assertEqual(labels_at_threshold(saved, .5), ["capital", "contains"])
        self.assertEqual(labels_at_threshold(saved, .7), ["capital"])
        self.assertEqual(labels_at_threshold(saved, .9), [])

    def test_exact_mentions_inside_possessive_and_punctuation_are_preserved(self):
        text = "John's son Peter."
        prepared, indices = align_supplied_sentence(
            MinimalWordProcessor(), text, [0, 4], [11, 16], ["person"], ["has child"],
        )
        self.assertEqual(prepared["tokens"], [["John", "'s", "son", "Peter", "."]])
        self.assertEqual(indices.tolist(), [[[0, 0], [3, 3]]])
        self.assertTrue(prepared["boundary_refined"])
        self.assertEqual(text[0:4], "John")
        self.assertEqual(text[11:16], "Peter")

    def test_head_tail_order_is_preserved_when_head_occurs_second(self):
        text = "Paris France"
        _, indices = align_supplied_sentence(
            MinimalWordProcessor(), text, [6, 12], [0, 5], ["location"], ["capital"],
        )
        self.assertEqual(indices.tolist(), [[[1, 1], [0, 0]]])

    def test_invalid_and_identical_spans_are_rejected(self):
        for head, tail in (([-1, 5], [6, 12]), ([0, 5], (0, 5))):
            with self.assertRaises(ValueError):
                align_supplied_sentence(MinimalWordProcessor(), "Paris France", head, tail,
                                        ["location"], ["capital"])


if __name__ == "__main__":
    unittest.main()
