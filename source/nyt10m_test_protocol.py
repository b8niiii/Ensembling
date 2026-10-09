"""Shared selected-evidence NYT10m test protocol; no inference on import.

The released manual NYT10m test serializes multiple labels as repeated rows.
Reconstruct each exact sentence/mention unit before label-blind cap selection.
References are kept separate from the input-only manifest sent to models.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "nyt10m_selected_evidence_test_v1_20261007"
TEST_SHA256 = "c703fcfef492bda6bb7b8a244eb4755415f04a86f53ab96ea16eb2ff9d431ad9"
CAP = 20
CUTOFFS = {"base": 0.5, "large": 0.7, "glidre": 0.7}
MODEL_IDS = {
    "base": {"repo": "knowledgator/gliner-relex-base-v1.0",
             "revision": "e6a880049a19c5cc222a7a479c32e84b0d8cdd9a",
             "weights_sha256": "7186a83eb61b067bfc9fdfbe542a07c69cd39a446be0cfec170bd83f8a5b12b5"},
    "large": {"repo": "knowledgator/gliner-relex-large-v1.0",
              "revision": "4aedc9226a5ac9e2f6b5ea3e91c1ee577c88a290",
              "weights_sha256": "7c5bd751e1b24e4254d70fe4355a986cd65400676ce3735f7752429fcc26960a"},
    "glidre": {"repo": "cea-list-ia/glidre_large",
               "revision": "d1ba1cf0f41d2849f1695c6acc67546700d721ba",
               "weights_sha256": "dc2fb3cf239ba898161fa9a9d6c27d8f4f51714b409d094f58c343ea12c52225"},
}
SOURCE_FILES = (
    "source/nyt10m_test_protocol.py", "source/nyt10m_test_encoders.py",
    "source/nyt10m_test_api.py", "source/run_full_validation_relex_given_entities.py",
    "source/run_full_validation_glidre.py", "source/relex_given_entities.py",
    "source/glidre_given_entities.py", "source/run_extended_gliner_relex.py",
    "source/validation_model_comparison.py",
)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(canonical(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_once(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise RuntimeError(f"Frozen experiment changed; use a new version: {path}")
    else:
        atomic_json(path, value)


def selected_indices(n, cap=CAP):
    if type(n) is not int or n < 1 or type(cap) is not int or cap < 2:
        raise ValueError("Positive sentence count and cap >= 2 required")
    if n <= cap:
        return list(range(n))
    den = cap - 1
    indices = [(2 * i * (n - 1) + den) // (2 * den) for i in range(cap)]
    if len(set(indices)) != cap or indices[0] != 0 or indices[-1] != n - 1:
        raise RuntimeError("Invalid evenly spaced selection")
    return indices


def positive_labels(labels, allowed):
    if not isinstance(labels, list) or any(not isinstance(x, str) for x in labels):
        raise ValueError("Annotation must be a list of relation IDs")
    if set(labels) - (set(allowed) | {"NA"}):
        raise ValueError("Unknown annotation label")
    if "NA" in labels and set(labels) - {"NA"}:
        raise ValueError("Conflicting NA and positive annotation")
    return set(labels) - {"NA"}


def reconstruct_test(records, allowed, *, cap=CAP, encoding="released_relation_rows"):
    """Pure reconstruction: annotation labels cannot influence selection indices.

    Equality requires the same ordered IDs, exact text and both character spans.
    Distinct mentions in identical text are separate evidence units. Repeated rows
    representing several labels of one unit are merged, preserving every label.
    """
    if encoding not in {"released_relation_rows", "anno_relation_list"}:
        raise ValueError("Specify the verified manual annotation encoding")
    groups = defaultdict(dict)
    counts = Counter()
    for line_number, row in enumerate(records, 1):
        if not isinstance(row, dict) or not isinstance(row.get("text"), str):
            raise ValueError(f"Invalid record {line_number}")
        text = row["text"]
        pair = tuple(row[role]["id"] for role in ("h", "t"))
        if any(not isinstance(x, str) or not x for x in pair):
            raise ValueError("Invalid ordered entity IDs")
        positions = {}
        for role, name in (("h", "head"), ("t", "tail")):
            entity = row[role]
            pos = entity["pos"]
            if (not isinstance(entity.get("name"), str) or
                    not isinstance(pos, (tuple, list)) or len(pos) != 2 or
                    any(type(x) is not int for x in pos)):
                raise ValueError("Malformed supplied entity")
            start, end = pos
            if not 0 <= start < end <= len(text) or text[start:end] != entity["name"]:
                raise ValueError("Supplied mention does not match original text")
            positions[name] = list(pos)
        if positions["head"] == positions["tail"]:
            raise ValueError("Identical supplied mentions")
        if encoding == "anno_relation_list":
            if "anno_relation_list" not in row:
                raise ValueError("Missing manual annotations; no distant-label fallback")
            labels = positive_labels(row["anno_relation_list"], allowed)
        else:
            if "anno_relation_list" in row:
                raise ValueError("Unexpected mixed annotation serialization")
            labels = positive_labels([row["relation"]], allowed)
        key = (text, tuple(positions["head"]), tuple(positions["tail"]))
        unit = groups[pair].setdefault(key, {
            "text": text, "positions": positions, "head": row["h"]["name"],
            "tail": row["t"]["name"], "labels": set(), "source_lines": [],
            "had_na": False,
        })
        if (unit["had_na"] and labels) or (unit["labels"] and not labels):
            raise ValueError("Conflicting manual annotations on the same sentence/mentions")
        unit["had_na"] |= not labels
        unit["labels"].update(labels)
        unit["source_lines"].append(line_number)
        counts["source_records"] += 1
        counts["records_with_anno_relation_list"] += "anno_relation_list" in row
    manifest, references = [], {}
    for pair in sorted(groups):
        units = list(groups[pair].values())  # First occurrence order, without label sorting.
        indices = selected_indices(len(units), cap)
        chosen = [units[i] for i in indices]
        full = set().union(*(u["labels"] for u in units))
        selected = set().union(*(u["labels"] for u in chosen))
        slug = digest(list(pair))[:24]
        bag = {
            "bag_id": slug, "pair_ids": list(pair), "head": chosen[0]["head"],
            "tail": chosen[0]["tail"], "full_bag_size": len(units),
            "source_record_count": sum(len(u["source_lines"]) for u in units),
            "selected_indices_zero_based": indices,
            "selected_source_lines": [u["source_lines"] for u in chosen],
            "sentences": [u["text"] for u in chosen],
            "entity_positions": [u["positions"] for u in chosen],
        }
        manifest.append(bag)
        references[slug] = {
            "pair_ids": list(pair), "selected_positive_labels": sorted(selected),
            "selected_sentence_labels": [sorted(u["labels"]) for u in chosen],
            "full_bag_positive_labels_for_audit_only": sorted(full),
            "omitted_positive_labels_for_audit_only": sorted(full - selected),
        }
        counts["bags"] += 1
        counts["distinct_sentence_mentions"] += len(units)
        counts["selected_sentence_mentions"] += len(chosen)
        counts["bags_shortened_by_cap"] += len(units) > cap
        counts["full_positive_bag_facts"] += len(full)
        counts["selected_positive_bag_facts"] += len(selected)
        counts["bags_with_changed_reference"] += full != selected
        counts["omitted_positive_bag_facts"] += len(full - selected)
    counts["extra_label_serialization_rows"] = counts["source_records"] - counts["distinct_sentence_mentions"]
    return manifest, references, dict(counts)


def label_descriptions():
    frozen = read_json(ROOT / "results/full_validation_settings.json")
    values = frozen["settings"]
    expected = hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()
    if expected != frozen["fingerprint"]:
        raise RuntimeError("Invalid published validation settings")
    for name, relative in (
        ("runner_sha256", "source/run_full_validation_relex_given_entities.py"),
        ("adapter_sha256", "source/relex_given_entities.py"),
        ("label_module_sha256", "source/run_extended_gliner_relex.py"),
    ):
        if file_hash(ROOT / relative) != values[name]:
            raise RuntimeError(f"Historical dependency changed: {relative}")
    labels = values["label_name"]
    rel2id = read_json(ROOT / "data/nyt10m/nyt10m_rel2id.json")
    if len(labels) != 24 or set(labels) != set(rel2id) - {"NA"}:
        raise RuntimeError("Unexpected relation inventory")
    return labels


def prepare_experiment():
    """Verify the release and persist one shared frozen specification, without inference."""
    source = ROOT / "data/nyt10m/nyt10m_test.txt"
    if file_hash(source) != TEST_SHA256:
        raise RuntimeError("Test bytes differ from the verified official manual release")
    descriptions = label_descriptions()
    with source.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    manifest, references, audit = reconstruct_test(rows, descriptions)
    # Independent fact-set parity with OpenNRE's released relation-row branch.
    official_facts = {(r["h"]["id"], r["t"]["id"], r["relation"])
                      for r in rows if r["relation"] != "NA"}
    reconstructed_facts = {(*tuple(r["pair_ids"]), label)
                           for r in references.values()
                           for label in r["full_bag_positive_labels_for_audit_only"]}
    if official_facts != reconstructed_facts:
        raise RuntimeError("Manual fact reconstruction differs from the OpenNRE loader")
    if (audit["source_records"], audit["distinct_sentence_mentions"], audit["bags"],
            audit["full_positive_bag_facts"]) != (11086, 9744, 5174, 3899):
        raise RuntimeError("Release reconstruction disagrees with the published manual test")
    from nyt10m_test_api import system_prompt, API_SETTINGS
    # Freeze tokenizer/config files separately from the declared weight hashes.
    # The offline encoder preflights verify the weight bytes and helper receipt.
    metadata_paths = []
    for member, spec in MODEL_IDS.items():
        directory = ROOT / "data/model_cache" / ("models--" + spec["repo"].replace("/", "--")) / "snapshots" / spec["revision"]
        config = "glidre_config.json" if member == "glidre" else "gliner_config.json"
        for name in (config, "tokenizer.json", "tokenizer_config.json", "spm.model"):
            metadata_paths.append(directory / name)
    receipt = ROOT / "data/model_cache/glidre_runtime/5c3bee6698b4365224576459f9c8d48c9a6d3b6d/assets.json"
    metadata_paths.append(receipt)
    missing = [str(p) for p in metadata_paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("Model metadata missing; prefetch assets in the SLM notebook first: " + ", ".join(missing))
    settings = {
        "version": VERSION, "test_sha256": TEST_SHA256, "cap": CAP,
        "selection": "exact sentence+ordered mentions; first occurrence; evenly spaced first/last inclusive",
        "manual_encoding": "verified released relation rows; not distant labels",
        "gold_rule": "union of positive manual labels of selected sentence/mention units only",
        "input_manifest_sha256": digest(manifest), "reference_sha256": digest(references),
        "cutoffs": CUTOFFS, "encoder_acceptance": "strict score > cutoff",
        "encoder_sentence_aggregation": "per-relation maximum over selected sentences",
        "ensemble": "label-wise two-of-three vote", "models": MODEL_IDS,
        "glidre_runtime": {"author_code_revision": "5c3bee6698b4365224576459f9c8d48c9a6d3b6d",
                           "isolated_gliner": "0.2.13", "full_text_word_guard": 2048,
                           "label_tokenizer_revision": "d4aa6901d3a41ba39fb536a557fa166f842b0e09"},
        "primary_systems": ["large_0.7", "majority_0.5_0.7_0.7", "deepseek_low"],
        "label_descriptions": descriptions, "api": API_SETTINGS,
        "api_system_prompt_sha256": digest(system_prompt(descriptions)),
        "invalid_rule": "empty predictions for positive F1; invalid counted separately; no success credit",
        "metrics": ["precision", "recall", "micro_f1", "total_wall_s", "mean_wall_s", "invalid_rate"],
        "source_hashes": {p: file_hash(ROOT / p) for p in SOURCE_FILES},
        "model_metadata_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in metadata_paths},
        "runtime": {p: importlib.metadata.version(p) for p in
                    ("torch", "gliner", "transformers", "huggingface_hub", "tokenizers", "safetensors", "numpy")},
        "official_sources": {
            "paper": "https://aclanthology.org/2021.findings-acl.112/",
            "download": "https://raw.githubusercontent.com/thunlp/OpenNRE/master/benchmark/download_nyt10m.sh",
            "test_url": "https://thunlp.oss-cn-qingdao.aliyuncs.com/opennre/benchmark/nyt10m/nyt10m_test.txt",
            "loader": "https://github.com/thunlp/OpenNRE/blob/master/opennre/framework/data_loader.py",
        },
    }
    fingerprint = digest(settings)
    run = ROOT / "data/test_runs" / VERSION
    wrapper = {"fingerprint": fingerprint, "settings": settings}
    for filename, value in (("settings.json", wrapper), ("manifest.json", manifest),
                            ("references.json", references), ("data_audit.json", audit)):
        write_once(run / filename, value)
    write_once(ROOT / "results/test_comparison_spec.json", wrapper)
    write_once(ROOT / "results/test_data_protocol_audit.json", {"fingerprint": fingerprint, **audit,
               "test_sha256": TEST_SHA256, "manual_field": "relation",
               "annotation_serialization": "one row per sentence/mention/manual-label combination",
               "selected_reference_only": True, "official_full_fact_union_parity": True,
               "reference_is_test_best_threshold_optimized": False})
    print(f"Prepared {len(manifest):,} test bags / {audit['selected_sentence_mentions']:,} selected sentences; {fingerprint[:16]}")
    return {"run": run, "fingerprint": fingerprint, "manifest": manifest,
            "references": references, "audit": audit, "settings": settings}


def verify_context(context):
    stored = read_json(context["run"] / "settings.json")
    if stored["fingerprint"] != context["fingerprint"] or digest(stored["settings"]) != context["fingerprint"]:
        raise RuntimeError("Frozen specification mismatch")
    for relative, expected in stored["settings"]["source_hashes"].items():
        if file_hash(ROOT / relative) != expected:
            raise RuntimeError(f"Source changed after freezing: {relative}")
    for relative, expected in stored["settings"]["model_metadata_sha256"].items():
        if file_hash(ROOT / relative) != expected:
            raise RuntimeError(f"Model configuration/tokenizer metadata changed: {relative}")
    if (digest(read_json(context["run"] / "manifest.json")) != stored["settings"]["input_manifest_sha256"] or
            digest(read_json(context["run"] / "references.json")) != stored["settings"]["reference_sha256"]):
        raise RuntimeError("Frozen inputs/references changed")
    if (digest(context["manifest"]) != stored["settings"]["input_manifest_sha256"] or
            digest(context["references"]) != stored["settings"]["reference_sha256"] or
            digest(context["settings"]) != context["fingerprint"]):
        raise RuntimeError("In-memory settings/inputs/references changed")


def labels_above(scores, cutoff, allowed):
    if set(scores) != set(allowed):
        raise ValueError("Incomplete relation-score inventory")
    if not 0 <= cutoff <= 1 or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or
            not math.isfinite(v) or not 0 <= v <= 1 for v in scores.values()):
        raise ValueError("Invalid relation scores/cutoff")
    return {label for label, value in scores.items() if value > cutoff}


def majority(member_predictions):
    if set(member_predictions) != set(CUTOFFS):
        raise ValueError("All three members are required")
    votes = Counter(label for labels in member_predictions.values() for label in set(labels))
    return {label for label, count in votes.items() if count >= 2}


def score_predictions(references, results, allowed):
    """Fixed-threshold multilabel micro scoring; missing/transport work blocks scoring."""
    if set(references) != set(results):
        raise ValueError("Incomplete or extra prediction inventory; no partial final score")
    total = Counter()
    for bag_id, gold_labels in references.items():
        row = results[bag_id]
        if row.get("status") not in {"valid", "valid_empty", "invalid"}:
            raise ValueError("Transport/compute failures are unresolved, not model predictions")
        gold = set(gold_labels)
        valid = row["status"] != "invalid"
        raw = row.get("relations")
        if not isinstance(raw, list) or any(not isinstance(x, str) for x in raw):
            raise ValueError("Malformed prediction inventory")
        pred = set(raw) if valid else set()
        if (not valid and raw) or (gold | pred) - set(allowed):
            raise ValueError("Unknown gold/predicted relation or nonempty invalid prediction")
        total["TP"] += len(gold & pred)
        total["FP"] += len(pred - gold)
        total["FN"] += len(gold - pred)
        total["invalid"] += not valid
        total["successful_exact_match_bags"] += valid and gold == pred
        total["reference_empty_bags"] += not gold
        total["positive_predictions_on_reference_empty_bags"] += not gold and bool(pred)
    tp, fp, fn = (total[k] for k in ("TP", "FP", "FN"))
    n = len(references)
    return {**dict(total), "bags": n, "TP": tp, "FP": fp, "FN": fn,
            "invalid": total["invalid"], "invalid_rate": total["invalid"] / n if n else 0.,
            "precision": tp / (tp + fp) if tp + fp else 0.,
            "recall": tp / (tp + fn) if tp + fn else 0.,
            "micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.}


def selected_gold(context):
    return {k: row["selected_positive_labels"] for k, row in context["references"].items()}


def scorer_self_check():
    """Constructed cases: false positives, omissions, NA, invalids and duplicates."""
    allowed = {"A", "B", "C"}
    result = score_predictions({"x": ["A", "B"]}, {"x": {"status": "valid", "relations": ["A", "C"]}}, allowed)
    assert (result["TP"], result["FP"], result["FN"], result["micro_f1"]) == (1, 1, 1, .5)
    for gold, pred, status, counts in [(["A"], [], "invalid", (0, 0, 1)),
                                      ([], [], "invalid", (0, 0, 0)),
                                      ([], [], "valid_empty", (0, 0, 0)),
                                      ([], ["A"], "valid", (0, 1, 0)),
                                      (["A"], ["A", "A"], "valid", (1, 0, 0))]:
        r = score_predictions({"x": gold}, {"x": {"status": status, "relations": pred}}, allowed)
        assert (r["TP"], r["FP"], r["FN"]) == counts
        assert r.get("successful_exact_match_bags", 0) == int(status != "invalid" and set(gold) == set(pred))
    assert labels_above({"A": .7, "B": .700001}, .7, {"A", "B"}) == {"B"}
    assert majority({"base": ["A", "A"], "large": ["B"], "glidre": ["B"]}) == {"B"}
    return {"success": True, "scope": "synthetic scoring/threshold/voting cases; no model or API calls"}
