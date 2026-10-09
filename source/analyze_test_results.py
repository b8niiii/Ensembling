"""Analyse frozen test caches; this module cannot load models or call an API.

The public count archive contains sufficient statistics for paired bag bootstrap
inference. Each row keeps one bag's complete multilabel TP/FP/FN contribution.
Raw-cache extraction is optional and reads the original experiment only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform

import numpy as np

SYSTEMS = ("large_0.7", "majority_0.5_0.7_0.7", "deepseek_low")
NAMES = ("GLiNER-relex large", "Three-member majority", "DeepSeek low")
CONTRASTS = ((1, 0), (2, 0), (2, 1))
METRICS = ("precision", "recall", "micro_f1")
VERSION = "paired_bag_analysis_v1_20261008"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def confusion_counts(gold, prediction):
    """Count distinct directed relations for one bag; NA is an empty gold set."""
    gold, prediction = set(gold), set(prediction)
    return len(gold & prediction), len(prediction - gold), len(gold - prediction)


def metrics_from_counts(counts):
    """Compute micro metrics from totals, with zero for a zero denominator."""
    counts = np.asarray(counts)
    tp, fp, fn = np.moveaxis(counts, -1, 0)
    numerators = np.stack((tp, tp, 2 * tp), axis=-1).astype(np.float64)
    denominators = np.stack((tp + fp, tp + fn, 2 * tp + fp + fn), axis=-1)
    return np.divide(numerators, denominators, out=np.zeros_like(numerators), where=denominators != 0)


def validate_counts(counts):
    counts = np.asarray(counts)
    if (counts.ndim != 3 or counts.shape[0] == 0 or counts.shape[1:] != (3, 3)
            or counts.dtype.kind not in "iu" or np.any(counts < 0)):
        raise ValueError("Expected nonnegative integer counts [bags, 3 systems, TP/FP/FN].")
    if not np.all(counts[:, :, 0] + counts[:, :, 2] == (counts[:, :1, 0] + counts[:, :1, 2])):
        raise ValueError("Paired systems must have identical gold support in every bag.")
    return counts


def bootstrap_metrics(counts, *, replicates=20_000, seed=20261008, batch_size=64):
    """Ordinary paired bootstrap: all systems receive the SAME sampled bags.

    A multilabel bag is never split into individual labels or sentence rows.
    Negative bags and invalid API answers remain in the sampling population.
    """
    counts = validate_counts(counts).astype(np.int64, copy=False)
    if type(replicates) is not int or replicates < 2 or type(batch_size) is not int or batch_size < 1:
        raise ValueError("At least two replicates and a positive batch size are required.")
    rng = np.random.default_rng(seed)
    samples = np.empty((replicates, 3, 3), dtype=np.float64)
    for start in range(0, replicates, batch_size):
        stop = min(start + batch_size, replicates)
        indices = rng.integers(0, len(counts), size=(stop - start, len(counts)))
        totals = counts[indices].sum(axis=1)
        samples[start:stop] = metrics_from_counts(totals)
    return samples


def paired_intervals(counts, *, replicates=20_000, seed=20261008):
    counts = validate_counts(counts)
    point = metrics_from_counts(counts.sum(axis=0))
    draws = bootstrap_metrics(counts, replicates=replicates, seed=seed)
    system_intervals = []
    for i, system in enumerate(SYSTEMS):
        row = {"system": system}
        for j, metric in enumerate(METRICS):
            row[metric] = {"estimate": float(point[i, j]),
                           "ci95": np.quantile(draws[:, i, j], [0.025, 0.975], method="linear").tolist()}
        system_intervals.append(row)
    differences = []
    for a, b in CONTRASTS:
        delta = 100 * (draws[:, a, 2] - draws[:, b, 2])
        lo, hi = np.quantile(delta, [0.025, 0.975], method="linear").tolist()
        differences.append({"system_a": SYSTEMS[a], "system_b": SYSTEMS[b],
                            "f1_difference_percentage_points": float(100 * (point[a, 2] - point[b, 2])),
                            "ci95_percentage_points": [lo, hi], "ci_excludes_zero": bool(lo > 0 or hi < 0)})
    return {"method": "paired whole-bag percentile bootstrap; micro metrics recomputed from summed TP/FP/FN",
            "replicates": replicates, "seed": seed, "bags": len(counts), "confidence_level": 0.95,
            "rng": "numpy.default_rng / PCG64", "quantile_method": "linear",
            "multiple_comparisons": "Three prespecified contrasts; nominal intervals, no familywise adjustment.",
            "scope": "Sampling uncertainty conditional on fixed saved predictions, references and frozen configurations."
                     " Assumes approximately exchangeable bags; shared entities/articles may create dependence."
                     " Does not measure repeated API-generation, training, annotation, model-selection or timing uncertainty.",
            "system_intervals": system_intervals, "paired_f1_differences": differences}


def load_public_analysis(root):
    root = Path(root)
    report = read_json(root / "results/test_statistical_analysis_20261008.json")
    spec = read_json(root / "results/test_comparison_spec.json")
    from nyt10m_test_protocol import digest
    if digest(spec["settings"]) != spec["fingerprint"] or report["fingerprint"] != spec["fingerprint"]:
        raise RuntimeError("Public analysis differs from the frozen experiment.")
    archive = root / report["count_archive"]["path"]
    if sha256(archive) != report["count_archive"]["sha256"]:
        raise RuntimeError("Public sufficient-statistic archive changed.")
    with np.load(archive, allow_pickle=False) as data:
        counts = validate_counts(data["counts"])
        if data["systems"].tolist() != list(SYSTEMS):
            raise RuntimeError("Archive system order changed.")
    actual = metrics_from_counts(counts.sum(axis=0))
    expected = [[r[m] for m in METRICS] for r in report["primary_rows"]]
    if not np.allclose(actual, expected, atol=1e-14, rtol=0):
        raise RuntimeError("Public metrics differ from archived bag contributions.")
    return report, counts


def extract_saved_results(root, run):
    """Validate caches and export counts. Neither preparation nor inference runs."""
    from nyt10m_test_protocol import canonical, digest, labels_above, majority
    from nyt10m_test_encoders import read_encoder_result
    import nyt10m_test_api as api

    root, run = Path(root), Path(run)
    spec = read_json(root / "results/test_comparison_spec.json")
    settings, fingerprint = spec["settings"], spec["fingerprint"]
    if digest(settings) != fingerprint or read_json(run / "settings.json") != spec:
        raise RuntimeError("Frozen run/specification mismatch.")
    for filename, expected in settings["source_hashes"].items():
        if sha256(root / filename) != expected:
            raise RuntimeError(f"Frozen inference source changed: {filename}")
    manifest, references = read_json(run / "manifest.json"), read_json(run / "references.json")
    if digest(manifest) != settings["input_manifest_sha256"] or digest(references) != settings["reference_sha256"]:
        raise RuntimeError("Input/reference identity mismatch.")
    ids = [b["bag_id"] for b in manifest]
    if len(set(ids)) != len(ids) or set(ids) != set(references):
        raise RuntimeError("Duplicate or unpaired bag IDs.")
    allowed = settings["label_descriptions"]
    # The frozen parser is used unchanged; assigning its inventory does not configure
    # requests, credentials or execution. The request runner is never invoked.
    if getattr(api, "RUN_API", False):
        raise RuntimeError("Analysis requires API execution disabled.")
    api.ALLOWED = set(allowed)
    prompt = api.system_prompt(allowed)
    if digest(prompt) != settings["api_system_prompt_sha256"]:
        raise RuntimeError("Frozen prompt identity mismatch.")
    context = {"run": run, "settings": settings, "fingerprint": fingerprint}
    ledger = {}
    for path in sorted((run / "api/attempts").glob("*.json")):
        row = read_json(path)
        if (row["attempt_id"] in ledger or row.get("inference_fingerprint") != fingerprint
                or row["state"] not in {"completed", "acknowledged_unknown"}):
            raise RuntimeError("Duplicate, unbound or unresolved journal entry.")
        ledger[row["attempt_id"]] = row
    counts = np.zeros((len(manifest), 3, 3), dtype=np.int64)
    invalid = np.zeros((len(manifest), 3), dtype=np.uint8)
    per_relation = {label: Counter() for label in allowed}
    transitions, examples = Counter(), {}
    member_seconds = Counter()
    seen_attempts = set()
    files = [run / "settings.json", run / "manifest.json", run / "references.json"]
    files += sorted((run / "api/attempts").glob("*.json"))
    for i, bag in enumerate(manifest):
        gold_row = references[bag["bag_id"]]
        gold = set(gold_row["selected_positive_labels"])
        if gold_row["pair_ids"] != bag["pair_ids"] or gold - set(allowed):
            raise RuntimeError("Gold direction or inventory changed.")
        members = {}
        for member in ("base", "large", "glidre"):
            row = read_encoder_result(context, member, bag)
            if row is None or row["status"] != "valid":
                raise RuntimeError("Complete valid encoder caches required.")
            members[member] = labels_above(row["relation_score_max"], settings["cutoffs"][member], allowed)
            member_seconds[member] += row["total_attempt_wall_s"]
            files.append(run / "encoders" / member / "bags" / (bag["bag_id"] + ".json"))
        large, vote = members["large"], majority(members)
        path = run / "api/bags/thinking_low" / (bag["bag_id"] + ".json")
        row = read_json(path)
        evidence = {"head": bag["head"], "tail": bag["tail"], "sentences": [
            {"text": text, "head_span": pos["head"], "tail_span": pos["tail"]}
            for text, pos in zip(bag["sentences"], bag["entity_positions"], strict=True)]}
        payload = {"model": settings["api"]["model"],
                   "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": canonical(evidence)}],
                   "response_format": settings["api"]["response_format"], "stream": False,
                   "max_tokens": settings["api"]["max_tokens"], "thinking": settings["api"]["thinking"],
                   "reasoning_effort": settings["api"]["reasoning_effort"]}
        if (row.get("inference_fingerprint") != fingerprint or row.get("pair_ids") != bag["pair_ids"]
                or row.get("request_sha256") != digest(payload) or row.get("mode") != "thinking_low"):
            raise RuntimeError("Unpaired or altered API result.")
        choice = row["response"]["choices"][0]
        parsed = api.parse_answer(choice["message"].get("content"), choice.get("finish_reason"))
        if any(row.get(k) != v for k, v in parsed.items()):
            raise RuntimeError("Stored answer differs from frozen strict parser.")
        attempt = ledger[row["attempt_id"]]
        if (attempt["state"] != "completed" or attempt["response"] != row["response"]
                or attempt["request_sha256"] != row["request_sha256"] or attempt["pair_ids"] != bag["pair_ids"]
                or row["attempt_id"] in seen_attempts):
            raise RuntimeError("API completion journal mismatch.")
        if attempt["usage"] != api.normalized_usage(row["response"]["usage"]):
            raise RuntimeError("API usage differs from recorded response.")
        seen_attempts.add(row["attempt_id"])
        files.append(path)
        api_prediction = set(parsed["relations"])
        for j, prediction in enumerate((large, vote, api_prediction)):
            counts[i, j] = confusion_counts(gold, prediction)
        invalid[i, 2] = parsed["status"] == "invalid"
        removed, added = large - vote, vote - large
        transitions["changed_bags"] += large != vote
        categories = {"removed_true_positives": removed & gold,
                      "removed_false_positives": removed - gold,
                      "added_true_positives": added & gold,
                      "added_false_positives": added - gold}
        for category, labels in categories.items():
            transitions[category] += len(labels)
            for label in labels:
                per_relation[label][category] += 1
                if category == "removed_true_positives":
                    if label in members["base"] or label in members["glidre"]:
                        raise RuntimeError("Lost large TP should have no supporting second vote.")
                elif category == "added_true_positives":
                    if label not in members["base"] or label not in members["glidre"]:
                        raise RuntimeError("Recovered TP should have both non-large votes.")
                target = (category == "removed_true_positives" and label == "/location/location/contains"
                          or category == "added_true_positives" and label == "/business/person/company")
                if target and category not in examples and len(bag["sentences"]) == 1:
                    examples[category] = {"bag_id": bag["bag_id"], "head": bag["head"], "tail": bag["tail"],
                                          "relation": label, "selected_sentences": bag["sentences"],
                                          "gold": sorted(gold), "member_predictions": {k: sorted(v) for k, v in members.items()}}
        for label in gold:
            per_relation[label]["manual_support"] += 1
    for member in member_seconds:
        path = run / "encoders" / member / "runtime.json"
        member_seconds[member] += sum(r["label_encoding_wall_s"] for r in read_json(path))
        files.append(path)
    if seen_attempts != {k for k, v in ledger.items() if v["state"] == "completed"}:
        raise RuntimeError("Extra completed API attempts require explicit accounting.")
    summary = read_json(root / "results/test_comparison_summary.json")
    audit = read_json(root / "results/test_results_audit_20261008.json")
    if summary["fingerprint"] != fingerprint or audit["fingerprint"] != fingerprint:
        raise RuntimeError("Unbound published comparison.")
    for j, published in enumerate(summary["rows"]):
        totals = counts[:, j].sum(axis=0)
        if published["system"] != SYSTEMS[j] or totals.tolist() != [published[k] for k in ("TP", "FP", "FN")]:
            raise RuntimeError("Recomputed test counts differ from publication.")
        if int(invalid[:, j].sum()) != published["invalid"]:
            raise RuntimeError("Invalid-answer accounting mismatch.")
    if dict(transitions) != audit["vote_changes_relative_to_large"]:
        raise RuntimeError("Transition counts differ from existing diagnostic audit.")
    local_times = [member_seconds["large"], sum(member_seconds.values())]
    for seconds, published in zip(local_times, summary["rows"][:2], strict=True):
        if not np.isclose(seconds, published["total_wall_s"], atol=1e-7, rtol=0):
            raise RuntimeError("Local accumulated timing mismatch.")
    account = audit["accounting"]
    for key, actual in (("attempts", len(ledger)), ("request_wall_s_including_attempts", sum(r["wall_s"] for r in ledger.values())),
                        ("token_charge_USD_estimate", sum(r["cost_usd_estimate"] for r in ledger.values() if r["state"] == "completed")),
                        ("unknown_charge_reserve_USD", sum(r["reserve_usd"] for r in ledger.values() if r["state"] == "acknowledged_unknown"))):
        if not np.isclose(account[key], actual, atol=1e-7, rtol=0):
            raise RuntimeError(f"API accounting mismatch: {key}")
    inventory_hash = digest([{ "path": str(p.relative_to(run)), "sha256": sha256(p)} for p in sorted(files)])
    diagnostics = {"relative_to": SYSTEMS[0], "ensemble": SYSTEMS[1], **dict(transitions),
                   "mechanism": "Large-only labels are rejected (one vote); labels absent from large are added only when both base and GLiDRE agree.",
                   "per_relation": [{"relation": label, **{k: per_relation[label][k] for k in
                        ("manual_support", "removed_true_positives", "added_true_positives", "removed_false_positives", "added_false_positives")}}
                                    for label in sorted(allowed)], "examples": examples}
    costs = {"local_cpu_inference_seconds": {SYSTEMS[0]: local_times[0], SYSTEMS[1]: local_times[1],
              "member_seconds": dict(member_seconds), "scope": summary["rows"][0]["timing_scope"]},
             "api_request_seconds": {"total": account["request_wall_s_including_attempts"],
                                     "scope": summary["rows"][2]["timing_scope"]},
             "api_expenditure_USD": {"completed_response_charge_estimate": account["token_charge_USD_estimate"],
                 "unknown_attempt_reserve": account["unknown_charge_reserve_USD"],
                 "usage_plus_reserve": account["token_charge_USD_estimate"] + account["unknown_charge_reserve_USD"],
                 "note": "Reconstructed from recorded tariff/cache accounting; not an invoice; reserve is not a known charge."},
             "peak_local_memory_bytes": {SYSTEMS[0]: None, SYSTEMS[1]: None},
             "memory_note": "Peak memory was not recorded. Checkpoint size is not runtime memory; provider memory is unavailable.",
             "local_monetary_cost_USD": None, "local_energy_cost": None,
             "comparison_note": "CPU inference and API network/service latency have different measurement scopes; neither is equivalent hardware compute."}
    return counts, invalid, {"analysis_version": VERSION, "fingerprint": fingerprint,
        "reference": summary["reference"], "primary_rows": summary["rows"], "ensemble_diagnostics": diagnostics,
        "cost_dimensions": costs, "provenance": {"saved_files_checked": len(files),
            "saved_file_inventory_sha256": inventory_hash, "source_hashes_unchanged": True,
            "input_manifest_sha256": settings["input_manifest_sha256"], "reference_sha256": settings["reference_sha256"],
            "new_predictions": 0, "new_API_calls": 0}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--run", type=Path, required=True, help="Existing frozen run directory; read only")
    parser.add_argument("--replicates", type=int, default=20_000)
    args = parser.parse_args()
    counts, invalid, report = extract_saved_results(args.root, args.run)
    archive = args.root / "results/test_bag_counts_20261008.npz"
    np.savez_compressed(archive, counts=counts, invalid=invalid, systems=np.asarray(SYSTEMS))
    report["count_archive"] = {"path": str(archive.relative_to(args.root)), "sha256": sha256(archive),
        "shape": list(counts.shape), "axes": ["paired bag in frozen manifest order", "primary system", "TP, FP, FN"],
        "content": "Sufficient statistics only; no text, names or entity IDs. Includes all bags and invalid outputs."}
    report["uncertainty"] = paired_intervals(counts, replicates=args.replicates)
    report["analysis_runtime"] = {"python": platform.python_version(), "numpy": np.__version__}
    report["provenance"]["analysis_source_sha256"] = sha256(__file__)
    write_json(args.root / "results/test_statistical_analysis_20261008.json", report)
    print(json.dumps(report["uncertainty"]["paired_f1_differences"], indent=2))


if __name__ == "__main__":
    main()
