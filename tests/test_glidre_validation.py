"""Direction, complete scores, cache integrity and fresh-annotation report tests.

Only synthetic tensors, stub processors and temporary result files are used.
No pretrained model is loaded and no checkpoint forward pass is performed.
"""

import hashlib
import itertools
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
import numpy as np
import torch
from glidre_given_entities import align_words, extract_scores, prepare_sentence, UntruncatedTokenizer, wire_native_biencoder
import validation_model_comparison as comparison


def splitter(text):
    return ((m.group(), m.start(), m.end()) for m in re.finditer(r"\S+", text))


class StubProcessor:
    words_splitter = staticmethod(splitter)

    def collate_raw_batch(self, items, names, class_to_ids, id_to_classes):
        words = items[0]["tokenized_text"]
        spans = [(m["mentions"][0]["start"], m["mentions"][0]["end"]) for m in items[0]["mentions"]]
        pairs = [[[a], [b]] for a, b in itertools.product(spans, repeat=2)]
        return {"tokens": [words], "rel_idx": torch.tensor([pairs]),
                "seq_length": torch.tensor([[len(words)]]), "id_to_classes": id_to_classes}

    def collate_fn(self, raw, prepare_labels, prepare_entities):
        length = len(raw["tokens"][0])
        return {"input_ids": torch.zeros((1, length), dtype=torch.long),
                "attention_mask": torch.ones((1, length), dtype=torch.long),
                "words_mask": torch.arange(1, length + 1).unsqueeze(0)}


class GLiDREAdapterTests(unittest.TestCase):
    def test_attention_tuple_bridge_and_cached_label_parity(self):
        tokens = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
        attention = torch.arange(9, dtype=torch.float32).reshape(1, 1, 3, 3)
        labels = torch.arange(96, dtype=torch.float32).reshape(24, 4)
        class Encoder:
            def __call__(self, *args, **kwargs): return tokens, attention, labels
            def encode_text(self, *args, **kwargs): return tokens, attention
        def word_extractor(token_values, attention_values, words_mask, attention_mask, batch_size, max_length, hidden_size, lengths):
            torch.testing.assert_close(attention_values, attention)
            self.assertEqual(tuple(lengths.shape), (1,))
            return token_values, attention_values, torch.ones((1, 3), dtype=torch.bool)
        core = SimpleNamespace(config=SimpleNamespace(labels_encoder='BGE', has_rnn=True, post_fusion_schema=''),
                               token_rep_layer=Encoder())
        wire_native_biencoder(core, word_extractor)
        arguments = {'input_ids': torch.zeros((1, 3), dtype=torch.long),
                     'attention_mask': torch.ones((1, 3), dtype=torch.long),
                     'text_lengths': torch.tensor([[3]]), 'words_mask': torch.tensor([[1, 2, 3]])}
        original = core.get_representations(**arguments)
        cached = core.get_representations(**arguments, labels_embeddings=labels)
        self.assertEqual(len(original), 5)
        for left, right in zip(original, cached, strict=True):
            torch.testing.assert_close(left, right)
        torch.testing.assert_close(original[-1], attention)

    def test_exact_character_spans_and_reverse_text_order(self):
        aligned = align_words(splitter, "Peter met John's son.", [10, 14], [0, 5])
        self.assertEqual(aligned["word_spans"], [(2, 2), (0, 0)])
        self.assertTrue(aligned["boundary_refined"])
        self.assertEqual(aligned["tokens"], ["Peter", "met", "John", "'s", "son."])

    def test_directed_complete_scores_exclude_reverse_and_self_pairs(self):
        spans = [(3, 3), (0, 0)]
        pairs = torch.tensor([[[[a], [b]] for a, b in itertools.product(spans, repeat=2)]])
        logits = torch.full((1, 4, 24), -8.0)
        logits[0, 2, :] = 8.0  # Stronger reverse-pair scores must be ignored.
        logits[0, 1, 5] = 0.0  # Exact .5 is excluded by the strict cutoff.
        logits[0, 1, 7] = 8.0
        names = {f"name{i}": f"r{i}" for i in range(24)}
        scores = extract_scores(logits, pairs, spans, {i+1: f"name{i}" for i in range(24)}, names)
        self.assertEqual(len(scores), 24)
        self.assertEqual([k for k, v in scores.items() if v > .5], ["r7"])
        self.assertEqual(scores["r5"], .5)

    def test_missing_scores_and_nonfinite_logits_are_rejected(self):
        pairs = torch.tensor([[[[[0, 0]], [[0, 0]]], [[[0, 0]], [[1, 1]]],
                               [[[1, 1]], [[0, 0]]], [[[1, 1]], [[1, 1]]]]])
        names = {f"n{i}": f"r{i}" for i in range(24)}
        inverse = {i+1: f"n{i}" for i in range(24)}
        with self.assertRaises(RuntimeError):
            extract_scores(torch.zeros((1, 4, 23)), pairs, [(0, 0), (1, 1)], inverse, names)
        logits = torch.zeros((1, 4, 24)); logits[0, 1, 0] = float('nan')
        with self.assertRaises(RuntimeError):
            extract_scores(logits, pairs, [(0, 0), (1, 1)], inverse, names)

    def test_preparation_rejects_missing_word_and_never_calls_ner(self):
        processor = StubProcessor()
        model = SimpleNamespace(data_processor=processor, device=torch.device('cpu'),
                                config=SimpleNamespace(max_len=2048,
                                    encoder_config=SimpleNamespace(vocab_size=1000),
                                    labels_encoder_config=SimpleNamespace(max_position_embeddings=512)))
        aligned, inputs, _ = prepare_sentence(model, 'Paris France', [6, 12], [0, 5],
                                              [f'n{i}' for i in range(24)], cached_labels=torch.zeros((24, 2)))
        self.assertEqual(aligned['word_spans'], [(1, 1), (0, 0)])
        self.assertIn('labels_embeddings', inputs)
        original = processor.collate_fn
        def truncated(*args, **kwargs):
            values = original(*args, **kwargs)
            values['words_mask'][0, -1] = 0
            return values
        with patch.object(processor, 'collate_fn', side_effect=truncated):
            with self.assertRaises(RuntimeError):
                prepare_sentence(model, 'Paris France', [6, 12], [0, 5], [f'n{i}' for i in range(24)])

    def test_tokenizer_proxy_disables_truncation(self):
        class Tokenizer:
            def __call__(self, *args, **kwargs): return kwargs
        self.assertFalse(UntruncatedTokenizer(Tokenizer())('text', truncation=True)['truncation'])

    def test_invalid_character_boundaries_are_rejected(self):
        for head, tail in (([-1, 5], [6, 12]), ([0, 5], [0, 5]), ([.1, 5], [6, 12])):
            with self.assertRaises(ValueError):
                align_words(splitter, 'Paris France', head, tail)


class ComparisonTests(unittest.TestCase):
    def test_fixed_grid_counts_and_labelwise_majority(self):
        self.assertEqual(len(comparison.configuration_inventory(['base', 'large'])), 24)
        self.assertEqual(len(comparison.configuration_inventory(['base', 'large', 'glidre'])), 90)
        arrays = {'base': {.5: np.array([[True, False]])}, 'large': {.5: np.array([[False, True]])},
                  'glidre': {.5: np.array([[True, True]])}}
        config = {'members': list(arrays), 'cutoffs': [.5]*3, 'rule': 'majority_2_of_3'}
        np.testing.assert_array_equal(comparison.combine_predictions(config, arrays), [[True, True]])

    def test_scorer_counts_positive_labels_and_empty_bags(self):
        gold = np.array([[True, False], [False, False]])
        actual = np.array([[True, True], [True, False]])
        result = comparison.array_metrics(gold, actual)
        self.assertEqual((result['TP'], result['FP'], result['FN']), (1, 2, 0))
        self.assertEqual(result['negative_bags_with_fp'], 1)
        self.assertEqual(result['micro_f1'], .5)

    def test_reports_reread_annotations_without_changing_score_caches(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); relex = root/'data/relex'; glidre = root/'data/glidre'
            labels = [f'r{i:02d}' for i in range(24)]
            bags = [{'pair_ids': [f'h{i}', f't{i}'], 'head': 'Paris', 'tail': 'France',
                     'full_bag_size': 1, 'selected_indices_zero_based': [0], 'sentences': ['Paris France'],
                     'entity_positions': [{'head': [0, 5], 'tail': [6, 12]}],
                     'ds_positive_labels': [labels[0]] if i == 0 else []} for i in range(2)]
            manifest = [{**b, 'sample_id': f'C{i+1:02d}'} for i,b in enumerate(bags)]
            comparison.save_json(relex/'manifest.json', bags)
            comparison.save_json(root/'data/pilot_runs/reviewed_validation_30_encoder_v1_20261001/manifest.json', manifest)
            frozen = {'fingerprint': 'original'}
            settings = {'source_run_fingerprint': 'original', 'source_manifest_sha256': comparison.file_hash(relex/'manifest.json')}
            fingerprint = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
            comparison.save_json(glidre/'settings.json', {'settings':settings, 'fingerprint':fingerprint})
            comparison.save_json(glidre/'model_runtime.json', {'sessions':[{'label_encoding_wall_s': .2}]})
            comparison.save_json(glidre/'offline_preflight.json', {'run_fingerprint':fingerprint, 'success':True, 'wall_s':.1})
            with patch.multiple(comparison, ROOT=root, RELEX_RUN=relex, GLIDRE_RUN=glidre):
                paths=[]
                for model in ('base', 'large', 'glidre'):
                    for i, bag in enumerate(bags):
                        scores = {label: .1 for label in labels}
                        if i == 0: scores[labels[0]] = .5 if model == 'base' else .8
                        row = {'model':comparison.MODEL_SPECS[model]['name'], 'status':'valid',
                               'pair_ids':bag['pair_ids'], 'run_fingerprint':fingerprint if model == 'glidre' else 'original',
                               'full_bag_size':1,'selected_indices_zero_based':[0], 'sentences_processed':1,
                               'wall_s':1.0, 'relation_score_max':scores,
                               'relations': sorted(k for k,v in scores.items() if v>.5),
                               'evidence':[{'selected_sentence_index':0, 'source_record_index':0,
                                            'head_char_span':[0,5], 'tail_char_span':[6,12], 'relation_scores':scores}]}
                        path=comparison.bag_path(glidre if model == 'glidre' else relex,model,bag)
                        comparison.save_json(path,row);paths.append(path)
                initial_hashes = [comparison.file_hash(p) for p in paths]
                reference={'C01':[labels[0]], 'C02':[]}
                def fresh(*args):
                    digest=hashlib.sha256(json.dumps(reference,sort_keys=True).encode()).hexdigest()
                    return reference, {'version':'confirmed-test','labels_sha256':digest}
                with patch.object(comparison, 'frozen_inputs', return_value=(bags, frozen, labels)), \
                        patch('reviewed_validation_reference.load_reviewed_reference', side_effect=fresh):
                    first,_=comparison.build_comparison(save=False)
                    reference['C02']=[labels[1]]
                    second,_=comparison.build_comparison(save=False)
                self.assertEqual(len(first['rows']),180)
                self.assertNotEqual(first['reviewed_reference_metadata'],second['reviewed_reference_metadata'])
                old=next(r for r in first['rows'] if r['reference']=='reviewed_30_text' and r['configuration']=='single|large@0.5')
                new=next(r for r in second['rows'] if r['reference']=='reviewed_30_text' and r['configuration']=='single|large@0.5')
                self.assertEqual((old['FN'],new['FN']),(0,1))
                self.assertAlmostEqual(first['timing']['inference_s_by_model']['glidre'],2.2)
                self.assertEqual(initial_hashes,[comparison.file_hash(p) for p in paths])
                bad=json.loads(paths[0].read_text());bad['relation_score_max'][labels[0]]=.9
                with self.assertRaises(RuntimeError):
                    comparison.validate_result(bad,bags[0],'base','original',labels)


if __name__ == '__main__':
    unittest.main()
