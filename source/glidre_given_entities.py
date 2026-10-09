"""GLiDRE inference on exact supplied head/tail spans and all relation labels.

The checkpoint's relation layers are used directly. No entity detector, entity
type annotations, pretrained-weight changes, or relation gold enters inference.
GLiDRE is imported only by its isolated worker, not by the notebook kernel.
"""

from __future__ import annotations

import math
from types import MethodType

import torch


def wire_native_biencoder(core, extract_word_embeddings):
    """Repair the author's mismatched tuple plumbing, without changing layers.

    The pinned BiEncoder returns (text, attention, labels), its word extractor
    returns (words, word attention, mask), and SpanPairModel expects five
    representations. The released BaseModel still unpacks the old GLiNER
    two/four-item interfaces. This bridge connects the existing functions and
    preserves the attention values needed by the pretrained ATLOP head.
    """
    if not core.config.labels_encoder or core.config.has_rnn is not True:
        raise RuntimeError("Unexpected checkpoint representation architecture")
    if core.config.post_fusion_schema:
        raise RuntimeError("This compatibility bridge was audited without post fusion")

    def representations(self, input_ids=None, attention_mask=None,
                        labels_embeddings=None, labels_input_ids=None,
                        labels_attention_mask=None, text_lengths=None,
                        words_mask=None, **kwargs):
        if labels_embeddings is None:
            tokens, attention, labels = self.token_rep_layer(
                input_ids, attention_mask, labels_input_ids, labels_attention_mask, **kwargs)
        else:
            tokens, attention = self.token_rep_layer.encode_text(input_ids, attention_mask, **kwargs)
            labels = labels_embeddings
        batch_size, _, hidden_size = tokens.shape
        lengths = text_lengths.reshape(-1)
        words, word_attention, mask = extract_word_embeddings(
            tokens, attention, words_mask, attention_mask, batch_size,
            lengths.max(), hidden_size, lengths)
        labels = labels.unsqueeze(0).expand(batch_size, -1, -1).to(words.dtype)
        label_mask = torch.ones(labels.shape[:-1], dtype=attention_mask.dtype, device=attention_mask.device)
        # The pinned author's bi-encoder path does not invoke its allocated RNN.
        # Do not add a layer invocation or change the released computation here.
        return labels, label_mask, words, mask, word_attention

    core.get_representations = MethodType(representations, core)


class UntruncatedTokenizer:
    """Disable the native processor's implicit text truncation."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def __call__(self, *args, **kwargs):
        kwargs["truncation"] = False
        return self.tokenizer(*args, **kwargs)


def align_words(splitter, text, head_pos, tail_pos):
    """Refine words at supplied character boundaries, retaining head/tail order."""
    spans = []
    for position in (head_pos, tail_pos):
        if len(position) != 2 or any(type(v) is not int for v in position):
            raise ValueError("Entity spans require two integer character offsets")
        start, end = position
        if not 0 <= start < end <= len(text):
            raise ValueError("Invalid supplied character span")
        spans.append((start, end))
    if spans[0] == spans[1]:
        raise ValueError("The two supplied mentions must have distinct spans")
    boundaries = {v for span in spans for v in span}
    tokens, starts, ends = [], [], []
    refined = False
    for token, start, end in splitter(text):
        cuts = [start, *sorted(v for v in boundaries if start < v < end), end]
        refined |= len(cuts) > 2
        for left, right in zip(cuts, cuts[1:]):
            tokens.append(text[left:right])
            starts.append(left)
            ends.append(right)
    if not tokens:
        raise ValueError("Empty sentence")
    start_lookup = {v: i for i, v in enumerate(starts)}
    end_lookup = {v: i for i, v in enumerate(ends)}
    word_spans = []
    for start, end in spans:
        if start not in start_lookup or end not in end_lookup:
            raise ValueError("Supplied mention cannot be aligned exactly")
        word_spans.append((start_lookup[start], end_lookup[end]))
    return {"tokens": tokens, "starts": starts, "ends": ends,
            "word_spans": word_spans, "boundary_refined": refined}


def prepare_sentence(model, text, head_pos, tail_pos, label_names, *, cached_labels=None):
    """Use the native pair processor; verify that every input word survived."""
    aligned = align_words(model.data_processor.words_splitter, text, head_pos, tail_pos)
    if len(aligned["tokens"]) > model.config.max_len:
        raise ValueError("Sentence exceeds the declared full-text word guard")
    names = list(label_names)
    if len(names) != 24 or len(set(names)) != 24:
        raise ValueError("Exactly 24 unique positive relation names are required")
    mentions = [{"id": i, "mentions": [{"start": start, "end": end}]}
                for i, (start, end) in enumerate(aligned["word_spans"])]
    mapping = {name: i + 1 for i, name in enumerate(names)}
    inverse = {v: k for k, v in mapping.items()}
    raw = model.data_processor.collate_raw_batch(
        [{"tokenized_text": aligned["tokens"], "mentions": mentions}], names,
        class_to_ids=mapping, id_to_classes=inverse,
    )
    if raw["tokens"] != [aligned["tokens"]]:
        raise RuntimeError("GLiDRE preprocessing changed or truncated supplied text")
    inputs = model.data_processor.collate_fn(
        raw, prepare_labels=False, prepare_entities=cached_labels is None,
    )
    inputs.update(text_lengths=raw["seq_length"], rel_idx=raw["rel_idx"])
    if cached_labels is not None:
        inputs["labels_embeddings"] = cached_labels
    expected = set(range(1, len(aligned["tokens"]) + 1))
    observed = set(inputs["words_mask"][0].tolist()) - {0}
    if observed != expected:
        raise RuntimeError("Transformer tokenization omitted one or more supplied words")
    if int(raw["rel_idx"].max()) >= len(aligned["tokens"]):
        raise RuntimeError("Supplied entity lies outside the complete encoded sentence")
    if int(inputs["input_ids"].max()) >= model.config.encoder_config.vocab_size:
        raise RuntimeError("Text tokenizer does not match checkpoint vocabulary")
    # The label encoder has absolute positions; its short descriptions must fit.
    if "labels_input_ids" in inputs:
        if inputs["labels_input_ids"].shape[1] > model.config.labels_encoder_config.max_position_embeddings:
            raise ValueError("Relation names exceed the label encoder context")
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}
    aligned["subtokens"] = int(inputs["attention_mask"].sum())
    return aligned, inputs, raw


def directed_row(rel_idx, word_spans):
    """Locate the exact directed pair; reverse and self pairs are never merged."""
    target = torch.tensor([[word_spans[0]], [word_spans[1]]], device=rel_idx.device)
    matches = (rel_idx[0] == target).all(dim=-1).all(dim=-1).all(dim=-1).nonzero().flatten()
    if len(matches) != 1:
        raise RuntimeError("Expected exactly one supplied head-to-tail relation row")
    return int(matches[0])


def extract_scores(logits, rel_idx, word_spans, id_to_names, name_to_id):
    """Return every directed sigmoid score, including those below the cutoff."""
    if logits.shape != (1, 4, 24):
        raise RuntimeError(f"Unexpected GLiDRE relation tensor shape: {tuple(logits.shape)}")
    if set(id_to_names) != set(range(1, 25)):
        raise RuntimeError("Invalid native GLiDRE label index inventory")
    row = directed_row(rel_idx, word_spans)
    values = torch.sigmoid(logits[0, row]).detach().cpu().tolist()
    scores = {name_to_id[id_to_names[i + 1]]: float(value) for i, value in enumerate(values)}
    if len(scores) != 24 or set(scores) != set(name_to_id.values()):
        raise RuntimeError("Incomplete GLiDRE positive-label inventory")
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in scores.values()):
        raise RuntimeError("Non-finite GLiDRE score")
    return scores


@torch.inference_mode()
def encode_relation_names(model, names):
    """Precompute the fixed label embeddings once, using the native label encoder."""
    tokens = model.data_processor.labels_tokenizer(
        list(names), return_tensors="pt", padding=True, truncation=False,
    )
    if tokens["input_ids"].shape[1] > model.config.labels_encoder_config.max_position_embeddings:
        raise ValueError("Label encoder context would be exceeded")
    return model.model.token_rep_layer.encode_labels(
        tokens["input_ids"].to(model.device), tokens["attention_mask"].to(model.device),
    ).detach()


@torch.inference_mode()
def score_sentence(model, text, head_pos, tail_pos, name_to_id, cached_labels):
    """Score the supplied pair through the unchanged pretrained relation layers."""
    aligned, inputs, raw = prepare_sentence(
        model, text, head_pos, tail_pos, name_to_id, cached_labels=cached_labels,
    )
    output = model.model(**inputs)
    return extract_scores(output.logits, inputs["rel_idx"], aligned["word_spans"],
                          raw["id_to_classes"], name_to_id), aligned


@torch.inference_mode()
def verify_native_parity(model, name_to_id, cached_labels):
    """Check raw-score, cached-label and multi-label-decoder parity offline."""
    differences = []
    for text, head_pos, tail_pos in (
        ("Paris is the capital of France", [24, 30], [0, 5]),
        ("Paris is the capital of France", [0, 5], [24, 30]),
    ):
        mentions = [{"id": i, "mentions": [{"start": p[0], "end": p[1], "value": text[p[0]:p[1]]}]}
                    for i, p in enumerate((head_pos, tail_pos))]
        native_inputs, native_raw = model.prepare_model_inputs([text], list(name_to_id), [mentions])
        native = model.model(**native_inputs).logits
        aligned, inputs, raw = prepare_sentence(
            model, text, head_pos, tail_pos, name_to_id, cached_labels=cached_labels,
        )
        output = model.model(**inputs).logits
        torch.testing.assert_close(native, output, rtol=1e-5, atol=1e-6)
        if not torch.equal(native_raw["rel_idx"], raw["rel_idx"]):
            raise AssertionError("Supplied pair indices differ from native preprocessing")
        scores = extract_scores(output, inputs["rel_idx"], aligned["word_spans"],
                                raw["id_to_classes"], name_to_id)
        decoded = model.decoder.decode(native_raw["tokens"], native_raw["id_to_classes"],
                                       native_raw["rel_idx"], native, threshold=.5, multi_label=True)[0]
        expected_spans = [[span] for span in aligned["word_spans"]]
        decoded_ids = {name_to_id[r["relation_type"]] for r in decoded
                       if r["entity_1"] == expected_spans[0] and r["entity_2"] == expected_spans[1]}
        if decoded_ids != {k for k, v in scores.items() if v > .5}:
            raise AssertionError("Native multi-label decoder and cached score cutoff disagree")
        differences.append(float((native - output).abs().max()))
    return {"native_relation_logit_max_difference": max(differences),
            "cached_label_parity": True, "directed_pair_checks": 2,
            "native_attention_tuple_bridge": True,
            "native_multilabel_cutoff_parity": True,
            "entity_recognition": False, "relation_scores_per_sentence": 24}
