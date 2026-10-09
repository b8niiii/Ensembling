"""Score supplied, directed entity spans with the pinned GLiNER-relex checkpoints.

This inference adapter bypasses entity classification and entity-score selection.
It reuses the encoder, span representations, and relation head without changing
weights. It is restricted to the two verified architectures in GLiNER 0.2.29.
"""

from __future__ import annotations

import importlib.metadata
import math
from types import MethodType

import torch


GLINER_VERSION = "0.2.29"
SUPPORTED_MODES = {"token_level", "markerV0"}


def _supplied_span_representations(
    self, words_embeddings, words_mask, prompts_embeddings,
    span_idx=None, span_mask=None, labels=None, threshold=None,
    return_embeddings=False,
):
    """Pool exactly the supplied head and tail; never score or select entities."""
    if labels is not None:
        raise RuntimeError("The supplied-entity adapter is inference-only")
    given = getattr(self, "_supplied_entity_indices", None)
    if given is None or given.shape != (1, 2, 2):
        raise RuntimeError("Exactly two supplied entity spans are required")
    given = given.to(words_embeddings.device)
    if int(given.min()) < 0 or int(given.max()) >= words_embeddings.shape[1]:
        raise ValueError("Supplied span lies outside the encoded word sequence")

    if self.config.span_mode == "token_level":
        # This checkpoint's pretrained span layer accepts an arbitrary span list.
        representations = self.span_rep_layer(words_embeddings, given)
    elif self.config.span_mode == "markerV0":
        # The pretrained marker layer reshapes in max_width-sized groups. Padding
        # preserves that shape; only the two real entity representations are used.
        width = self.config.max_width
        padded = given.new_zeros((1, width, 2))
        padded[:, :2] = given
        representations = self.span_rep_layer(words_embeddings, padded)
        representations = representations.reshape(1, width, -1)[:, :2]
    else:
        raise RuntimeError("Unsupported span representation")

    mask = torch.ones((1, 2), dtype=torch.bool, device=words_embeddings.device)
    # No entity logits are produced. Relation decoding uses the supplied indices.
    result = (None, representations, mask, given)
    return (*result, None) if return_embeddings else result


def install_supplied_span_adapter(model):
    """Install a per-instance inference method; leave model weights unchanged."""
    if importlib.metadata.version("gliner") != GLINER_VERSION:
        raise RuntimeError(f"Supplied-span adapter requires gliner=={GLINER_VERSION}")
    if model.config.span_mode not in SUPPORTED_MODES:
        raise RuntimeError(f"Unsupported checkpoint span mode: {model.config.span_mode}")
    if hasattr(model.model, "relations_rep_layer"):
        raise RuntimeError("An adjacency-filtering architecture is not supported")
    if not hasattr(model.model, "pair_rep_layer"):
        raise RuntimeError("The verified pair-representation relation head is required")
    if getattr(model.model, "_supplied_span_adapter_installed", False):
        raise RuntimeError("Supplied-span adapter is already installed")
    original = model.model.represent_spans
    model.model.represent_spans = MethodType(_supplied_span_representations, model.model)
    model.model._supplied_span_adapter_installed = True
    return original


def align_supplied_sentence(model, text, head_pos, tail_pos, entity_labels, relation_names):
    """Retain exact mention boundaries, splitting model words only when required."""
    character_spans = []
    for position in (head_pos, tail_pos):
        if len(position) != 2 or any(type(value) is not int for value in position):
            raise ValueError("Entity positions must contain two integer offsets")
        start, end = position
        if not 0 <= start < end <= len(text):
            raise ValueError("Invalid half-open character span")
        character_spans.append({"start": start, "end": end})
    if character_spans[0] == character_spans[1]:
        raise ValueError("Head and tail must have distinct mention spans")
    prepared = model.prepare_batch(
        [text], entity_labels, relations=relation_names,
    )
    if prepared["valid_to_orig_idx"] != [0]:
        raise ValueError("Empty or invalid sentence")
    # The public input_spans converter silently drops non-word-aligned spans.
    # Refine only intersected words at the supplied character offsets instead.
    boundaries = {value for span in character_spans for value in span.values()}
    tokens, starts, ends = [], [], []
    refined = False
    for token, start, end in zip(prepared["tokens"][0], prepared["start_token_map"][0],
                                  prepared["end_token_map"][0], strict=True):
        cuts = [start, *sorted(value for value in boundaries if start < value < end), end]
        if len(cuts) == 2:
            tokens.append(token)
            starts.append(start)
            ends.append(end)
        else:
            refined = True
            for left, right in zip(cuts, cuts[1:]):
                tokens.append(text[left:right])
                starts.append(left)
                ends.append(right)
    start_lookup = {value: i for i, value in enumerate(starts)}
    end_lookup = {value: i for i, value in enumerate(ends)}
    word_spans = []
    for span in character_spans:
        if span["start"] not in start_lookup or span["end"] not in end_lookup:
            raise ValueError("Supplied entity boundary does not align after word refinement")
        word_spans.append((start_lookup[span["start"]], end_lookup[span["end"]]))
    prepared.update(tokens=[tokens], start_token_map=[starts], end_token_map=[ends],
                    word_input_spans=[word_spans], boundary_refined=refined)
    if refined:
        prepared["input_x"] = model.prepare_base_input([tokens])
    for character, (start, end) in zip(character_spans, word_spans, strict=True):
        if (start > end or prepared["start_token_map"][0][start] != character["start"] or
                prepared["end_token_map"][0][end] != character["end"]):
            raise ValueError("Entity offsets changed during tokenization")
    if len(prepared["tokens"][0]) > model.config.max_len:
        raise ValueError("Sentence exceeds checkpoint word limit; truncation is not permitted")
    return prepared, torch.tensor([word_spans], dtype=torch.long)


def prepare_supplied_sentence(model, text, head_pos, tail_pos, entity_labels, relation_names):
    """Collate the aligned sentence and preserve supplied head/tail index order."""
    prepared, indices = align_supplied_sentence(
        model, text, head_pos, tail_pos, entity_labels, relation_names,
    )
    batch = model.collate_batch(
        prepared["input_x"], prepared["entity_types"],
        relation_types=prepared["relation_types"],
    )
    if int(indices.max()) >= int(batch["text_lengths"][0]):
        raise ValueError("An entity was truncated from the collated input")
    return prepared, batch, indices


def extract_directed_scores(output, class_mapping, name_to_label):
    """Read every relation score for supplied head index 0 to tail index 1."""
    pairs = output.rel_idx[0]
    valid = output.rel_mask[0].bool()
    selected = ((pairs[:, 0] == 0) & (pairs[:, 1] == 1) & valid).nonzero().flatten()
    if selected.numel() != 1:
        raise RuntimeError("Expected exactly one supplied head-to-tail relation row")
    probabilities = torch.sigmoid(output.rel_logits[0, int(selected[0])]).cpu().tolist()
    if isinstance(class_mapping, list):
        class_mapping = class_mapping[0]
    if set(class_mapping) != set(range(1, len(probabilities) + 1)):
        raise RuntimeError("Unexpected relation-class mapping")
    scores = {name_to_label[class_mapping[i + 1]]: float(probability)
              for i, probability in enumerate(probabilities)}
    if set(scores) != set(name_to_label.values()):
        raise RuntimeError("Incomplete positive-relation score inventory")
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in scores.values()):
        raise RuntimeError("Non-finite or out-of-range relation score")
    return scores


@torch.inference_mode()
def score_supplied_sentence(model, text, head_pos, tail_pos, entity_labels, name_to_label):
    """Run supplied-span relation inference without any entity-score threshold."""
    if not getattr(model.model, "_supplied_span_adapter_installed", False):
        raise RuntimeError("Supplied-span adapter is not installed")
    prepared, batch, indices = prepare_supplied_sentence(
        model, text, head_pos, tail_pos, entity_labels, list(name_to_label),
    )
    model.model._supplied_entity_indices = indices
    try:
        # The library's default entity-threshold argument is inert: the adapter
        # never calls its entity scorer, entity selector, or public NER decoder.
        output = model.run_batch(batch)
        if output.logits is not None or not torch.equal(output.entity_spans.cpu(), indices):
            raise RuntimeError("Supplied entities were not retained exactly")
        scores = extract_directed_scores(output, batch["rel_id_to_classes"], name_to_label)
    finally:
        del model.model._supplied_entity_indices
    return scores, prepared["word_input_spans"][0]


def labels_at_threshold(scores, threshold):
    """Apply one strict cutoff independently to all positive relation labels."""
    if not 0 <= threshold <= 1:
        raise ValueError("Relation threshold must be between zero and one")
    return sorted(label for label, value in scores.items() if value > threshold)


@torch.inference_mode()
def verify_supplied_span_adapter(model, entity_labels, name_to_label):
    """Check native relation-layer parity and prove that NER selection is unused."""
    text = "Paris is the capital of France."
    head, tail = [text.index("France"), text.index("France") + 6], [0, 5]
    _, batch, indices = prepare_supplied_sentence(
        model, text, head, tail, entity_labels, list(name_to_label),
    )
    mask = torch.ones((1, 2), dtype=torch.bool)
    native = model.model.represent_spans
    selector = model.model.select_span_target_embedding
    if model.config.span_mode == "token_level":
        reference = model.run_batch(batch, span_idx=indices, span_mask=mask)
    else:
        # Keep native span representations and relation weights, but force the
        # supplied mentions so the reference itself has no entity-selection bias.
        def force_native_spans(self, span_rep, span_scores, span_mask, span_labels=None,
                               threshold=None, top_k=None, span_idx=None):
            selected = []
            for entity in indices[0].to(span_idx.device):
                matches = ((span_idx[0] == entity).all(-1) & span_mask[0].bool()).nonzero().flatten()
                if matches.numel() != 1:
                    raise RuntimeError("Supplied span is missing from the native span grid")
                selected.append(int(matches[0]))
            flat = span_rep.reshape(1, -1, span_rep.shape[-1])
            return flat[:, selected], mask.to(flat.device), indices.to(flat.device)
        model.model.select_span_target_embedding = MethodType(force_native_spans, model.model)
        try:
            reference = model.run_batch(batch)
        finally:
            model.model.select_span_target_embedding = selector

    install_supplied_span_adapter(model)
    scorer = getattr(model.model, "scorer", None)
    scorer_forward = scorer.forward if scorer is not None else None
    def forbidden(*args, **kwargs):
        raise AssertionError("Entity recognition or entity-score selection was called")
    model.model.select_span_target_embedding = forbidden
    if scorer is not None:
        scorer.forward = forbidden
    model.model._supplied_entity_indices = indices
    try:
        output = model.run_batch(batch)
        torch.testing.assert_close(output.rel_logits, reference.rel_logits, rtol=1e-5, atol=1e-6)
        if output.logits is not None or not torch.equal(output.entity_spans.cpu(), indices):
            raise AssertionError("Entity recognition was not fully bypassed")
        scores = extract_directed_scores(output, batch["rel_id_to_classes"], name_to_label)
        max_difference = float((output.rel_logits - reference.rel_logits).abs().max())
    finally:
        del model.model._supplied_entity_indices
        model.model.select_span_target_embedding = selector
        if scorer is not None:
            scorer.forward = scorer_forward
        model.model.represent_spans = native
        del model.model._supplied_span_adapter_installed
    return {"native_relation_logit_max_difference": max_difference,
            "ner_logits_produced": False, "entity_selector_called": False,
            "relations_scored": len(scores), "synthetic_head": "France", "synthetic_tail": "Paris"}
