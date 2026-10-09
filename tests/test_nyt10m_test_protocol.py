"""Synthetic test-protocol fixtures; no pretrained inference or real API calls."""
import ast
import io
import json
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
import nyt10m_test_protocol as protocol
import nyt10m_test_api as api
import nyt10m_test_encoders as encoders


def record(index, label="NA"):
    return {"text": f"Head Tail sentence {index}.",
            "h": {"id": "H", "name": "Head", "pos": [0, 4]},
            "t": {"id": "T", "name": "Tail", "pos": [5, 9]}, "relation": label}


class ReconstructionTests(unittest.TestCase):
    def test_reconstructs_multiple_labels_without_repeating_evidence(self):
        inputs, gold, audit = protocol.reconstruct_test([record(0, "A"), record(0, "B"), record(0, "A")], {"A", "B"})
        self.assertEqual(len(inputs[0]["sentences"]), 1)
        self.assertEqual(gold[inputs[0]["bag_id"]]["selected_positive_labels"], ["A", "B"])
        self.assertEqual(inputs[0]["selected_source_lines"], [[1, 2, 3]])
        self.assertEqual(audit["extra_label_serialization_rows"], 2)

    def test_cap_excludes_labels_only_on_omitted_sentences(self):
        records = [record(i, "B" if i == 1 else "A" if i == 0 else "NA") for i in range(21)]
        inputs, references, audit = protocol.reconstruct_test(records, {"A", "B"}, cap=2)
        bag = inputs[0]
        self.assertEqual(bag["selected_indices_zero_based"], [0, 20])
        self.assertEqual(references[bag["bag_id"]]["selected_positive_labels"], ["A"])
        self.assertEqual(references[bag["bag_id"]]["omitted_positive_labels_for_audit_only"], ["B"])
        self.assertEqual(audit["bags_with_changed_reference"], 1)

    def test_selection_does_not_depend_on_labels(self):
        rows = [record(i, "NA") for i in range(43)]
        inputs, _, _ = protocol.reconstruct_test(rows, {"A"})
        for r in rows:
            r["relation"] = "A"
        changed, _, _ = protocol.reconstruct_test(rows, {"A"})
        self.assertEqual(inputs, changed)

    def test_empty_manual_list_is_authoritative_and_missing_list_is_error(self):
        r = {**record(0, "A"), "anno_relation_list": []}
        inputs, references, _ = protocol.reconstruct_test([r], {"A"}, encoding="anno_relation_list")
        self.assertEqual(references[inputs[0]["bag_id"]]["selected_positive_labels"], [])
        with self.assertRaises(ValueError):
            protocol.reconstruct_test([record(0, "A")], {"A"}, encoding="anno_relation_list")

    def test_direction_is_ordered_not_text_order(self):
        forward = record(0, "A")
        backward = {**forward, "h": forward["t"], "t": forward["h"], "relation": "B"}
        inputs, gold, _ = protocol.reconstruct_test([forward, backward], {"A", "B"})
        self.assertEqual(len(inputs), 2)
        for bag in inputs:
            expected = ["A"] if bag["pair_ids"] == ["H", "T"] else ["B"]
            self.assertEqual(gold[bag["bag_id"]]["selected_positive_labels"], expected)

    def test_identical_text_with_different_mentions_remains_distinct(self):
        text = "Head Tail Head"
        r = {**record(0, "A"), "text": text}
        other = {**r, "h": {"id": "H", "name": "Head", "pos": [10, 14]}, "relation": "B"}
        inputs, _, _ = protocol.reconstruct_test([r, other], {"A", "B"})
        self.assertEqual(len(inputs[0]["sentences"]), 2)

    def test_unknown_labels_bad_spans_and_conflicting_na_are_rejected(self):
        for rows in ([record(0, "unknown")],
                     [{**record(0), "h": {"id": "H", "name": "Head", "pos": [1, 5]}}],
                     [record(0, "A"), record(0, "NA")]):
            with self.assertRaises(ValueError):
                protocol.reconstruct_test(rows, {"A"})


class ScorerTests(unittest.TestCase):
    def test_constructed_scoring_contract(self):
        self.assertTrue(protocol.scorer_self_check()["success"])

    def test_missing_and_transport_results_cannot_be_scored(self):
        for results in ({}, {"x": {"status": "http_error", "relations": []}},
                        {"x": {"status": "failed", "relations": []}}):
            with self.assertRaises(ValueError):
                protocol.score_predictions({"x": ["A"]}, results, {"A"})

    def test_invalid_negative_is_failure_not_success(self):
        r = protocol.score_predictions({"x": []}, {"x": {"status": "invalid", "relations": []}}, {"A"})
        self.assertEqual(r["invalid_rate"], 1)
        self.assertEqual(r["successful_exact_match_bags"], 0)

    def test_strict_cutoffs_finite_inventory_and_labelwise_vote(self):
        self.assertEqual(protocol.labels_above({"A": .7, "B": .71}, .7, {"A", "B"}), {"B"})
        for scores in ({"A": float("nan")}, {"A": 1.1}, {"A": True}, {}):
            with self.assertRaises(ValueError):
                protocol.labels_above(scores, .7, {"A"})
        self.assertEqual(protocol.majority({"base": ["A", "A"], "large": ["B"], "glidre": ["B"]}), {"B"})
        with self.assertRaises(ValueError):
            protocol.majority({"base": ["A"], "large": ["A"]})

    def test_matches_existing_scorer_on_valid_multilabel_cases(self):
        import numpy as np
        from validation_model_comparison import array_metrics
        g = np.array([[1, 1, 0], [0, 0, 0], [0, 1, 0]], dtype=bool)
        p = np.array([[1, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=bool)
        labels = ["A", "B", "C"]
        gold = {str(i): [labels[j] for j in range(3) if g[i, j]] for i in range(3)}
        pred = {str(i): {"status": "valid", "relations": [labels[j] for j in range(3) if p[i, j]]} for i in range(3)}
        actual = protocol.score_predictions(gold, pred, labels)
        prior = array_metrics(g, p)
        for field in ("TP", "FP", "FN", "precision", "recall", "micro_f1"):
            self.assertEqual(actual[field], prior[field])


class APIContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        inputs, references, audit = protocol.reconstruct_test([record(0, "A")], {"A", "B"})
        self.context = {"run": Path(self.temporary.name), "fingerprint": "synthetic",
                        "manifest": inputs, "references": references, "audit": audit,
                        "settings": {"label_descriptions": {"A": "first relation", "B": "second relation"}}}
        self.context_patch = patch.object(api, "verify_context")
        self.context_patch.start()
        self.addCleanup(self.context_patch.stop)
        self.root_patch = patch.object(api, "ROOT", Path(self.temporary.name))
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        api.configure(self.context, run_api=True, keep_awake=False)

    def response(self, *, content='{"relations":["A"]}', finish="stop"):
        return {"model": "deepseek-flash", "choices": [{"message": {"content": content}, "finish_reason": finish}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 12, "total_tokens": 112}}

    def test_prompt_exactly_matches_completed_v2(self):
        notebook = json.loads((protocol.ROOT / "source/DeepSeek_flash_validation.ipynb").read_text())
        source = ''.join(notebook['cells'][6]['source'])
        first = ast.parse(source).body[0]
        namespace = {"LABEL_DESCRIPTIONS": api.LABEL_DESCRIPTIONS, "ALLOWED": api.ALLOWED}
        exec(compile(ast.Module(body=[first], type_ignores=[]), '<v2-prompt>', 'exec'), namespace)
        self.assertEqual(api.SYSTEM_PROMPT, namespace["SYSTEM_PROMPT"])

    def test_request_only_contains_selected_evidence_and_low_controls(self):
        bag = {**self.context["manifest"][0], "gold": ["B"], "distant_labels": ["B"]}
        payload = api.request_for(bag, "thinking_low")
        self.assertEqual(payload["reasoning_effort"], "low")
        self.assertEqual(payload["max_tokens"], 4096)
        user = json.loads(payload["messages"][1]["content"])
        self.assertEqual(set(user), {"head", "tail", "sentences"})
        self.assertEqual(set(user["sentences"][0]), {"text", "head_span", "tail_span"})

    def test_parser_duplicates_unknown_nested_na_and_truncation(self):
        self.assertEqual(api.parse_answer('{"relations":["A","A"]}', "stop")["duplicates"], 1)
        for content, reason in [('{"relations":[["A"]]}', 'stop'),
                                ('{"relations":["NA"]}', 'stop'),
                                ('{"relations":["unknown"]}', 'stop'),
                                ('{"relations":[],"relations":[]}', 'stop'),
                                ('{"relations":["A"]}', 'length'), ('', 'stop')]:
            self.assertEqual(api.parse_answer(content, reason)["status"], "invalid")

    def test_resume_reuses_completed_response_without_paying_again(self):
        with patch.object(api, "credential", return_value="synthetic-key"), \
             patch.object(api.urllib.request, "urlopen", side_effect=lambda *a, **k: io.BytesIO(json.dumps(self.response()).encode())) as network:
            api.run_experiment()
            api.run_experiment()
            self.assertEqual(network.call_count, 1)
        self.assertEqual(api.report_api()["rows"][0]["micro_f1"], 1)

    def test_http_error_is_pending_and_can_resume(self):
        error = urllib.error.HTTPError("https://synthetic", 503, "Unavailable", {}, io.BytesIO(b"temporary"))
        with patch.object(api, "credential", return_value="synthetic-key"), \
             patch.object(api.urllib.request, "urlopen", side_effect=error), \
             patch.object(api, "MAX_HTTP_RETRIES", 0):
            api.run_experiment()
        with self.assertRaises(RuntimeError):
            api.report_api()
        with patch.object(api, "credential", return_value="synthetic-key"), \
             patch.object(api.urllib.request, "urlopen", side_effect=lambda *a, **k: io.BytesIO(json.dumps(self.response()).encode())) as network:
            api.run_experiment()
            self.assertEqual(network.call_count, 1)
        self.assertEqual(api.status_and_cost()["attempts"], 2)

    def test_uncertain_transport_attempt_prevents_silent_duplicate_call(self):
        with patch.object(api, "credential", return_value="synthetic-key"), \
             patch.object(api.urllib.request, "urlopen", side_effect=TimeoutError("synthetic timeout")) as network:
            with self.assertRaises(TimeoutError):
                api.run_experiment()
            with self.assertRaises(api.RunStopped):
                api.run_experiment()
            self.assertEqual(network.call_count, 1)

    def test_budget_stops_before_network_submission(self):
        api.MAX_BUDGET_USD = .0000001
        with patch.object(api, "credential", return_value="synthetic-key"), \
             patch.object(api.urllib.request, "urlopen") as network:
            with self.assertRaises(api.RunStopped):
                api.run_experiment()
            network.assert_not_called()

    def test_invalid_answer_is_cached_without_generation_retry(self):
        with patch.object(api, "credential", return_value="synthetic-key"), \
             patch.object(api.urllib.request, "urlopen", side_effect=lambda *a, **k: io.BytesIO(json.dumps(self.response(finish="length")).encode())) as network:
            api.run_experiment()
            api.run_experiment()
            self.assertEqual(network.call_count, 1)
        row = api.report_api()["rows"][0]
        self.assertEqual((row["invalid"], row["FN"]), (1, 1))


class EncoderCacheTests(unittest.TestCase):
    def test_cache_maxima_and_boundaries_are_verified(self):
        with tempfile.TemporaryDirectory() as temporary:
            inputs, _, _ = protocol.reconstruct_test([record(0, "A")], {"A", "B"})
            bag = inputs[0]
            context = {"run": Path(temporary), "fingerprint": "synthetic",
                       "settings": {"label_descriptions": {"A": "first", "B": "second"}}}
            row = {"fingerprint": "synthetic", "member": "large", "input_sha256": protocol.digest(bag),
                   "pair_ids": bag["pair_ids"], "status": "valid", "relations": ["B"],
                   "wall_s": 1., "total_attempt_wall_s": 1.,
                   "relation_score_max": {"A": .7, "B": .8},
                   "evidence": [{"selected_sentence_index": 0, "source_lines": [1],
                                 "head_char_span": [0, 4], "tail_char_span": [5, 9],
                                 "relation_scores": {"A": .7, "B": .8}}]}
            path = encoders.cache_path(context, "large", bag)
            protocol.atomic_json(path, row)
            self.assertEqual(encoders.read_encoder_result(context, "large", bag)["relations"], ["B"])
            row["evidence"][0]["relation_scores"]["A"] = .9
            protocol.atomic_json(path, row)
            with self.assertRaises(RuntimeError):
                encoders.read_encoder_result(context, "large", bag)


if __name__ == "__main__":
    unittest.main()
