"""Compare unchanged GLiNER caches and a separate GLiDRE validation run.

No checkpoint imports or inference occur here. Confirmed annotations are read
fresh; annotation hashes version reports independently of inference caches.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEX_RUN = ROOT / "data/pilot_runs/relex_full_validation_given_entities_v1_20261004"
GLIDRE_RUN = ROOT / "data/pilot_runs/glidre_full_validation_given_entities_v1_20261006"
MODEL_SPECS = {
    "base": {"slug": "gliner_relex_base", "name": "GLiNER-relex-base"},
    "large": {"slug": "gliner_relex_large", "name": "GLiNER-relex-large"},
    "glidre": {"slug": "glidre_large", "name": "GLiDRE-large"},
}
CUTOFFS = (.5, .7, .9)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def frozen_inputs():
    """Read only validation inputs and verify the original inference sources."""
    settings = read_json(RELEX_RUN / "settings.json")
    frozen = settings["settings"]
    digest = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()
    if digest != settings["fingerprint"]:
        raise RuntimeError("Invalid original relex fingerprint")
    for key, relative in {
        "runner_sha256": "source/run_full_validation_relex_given_entities.py",
        "adapter_sha256": "source/relex_given_entities.py",
        "label_module_sha256": "source/run_extended_gliner_relex.py",
    }.items():
        if file_hash(ROOT / relative) != frozen[key]:
            raise RuntimeError(f"Frozen inference source changed: {relative}")
    manifest = read_json(RELEX_RUN / "manifest.json")
    if (len(manifest) != 36266 or len({tuple(b["pair_ids"]) for b in manifest}) != 36266
            or sum(len(b["sentences"]) for b in manifest) != 44751):
        raise RuntimeError("Unexpected frozen validation input inventory")
    labels = sorted(frozen["label_name"])
    if len(labels) != 24:
        raise RuntimeError("Expected the frozen 24 positive relation IDs")
    # Verify manifest evidence against the original validation bytes, without
    # reading train or test or consulting model predictions.
    validation = ROOT / "data/nyt10m/nyt10m_val.txt"
    if file_hash(validation) != frozen["validation_sha256"]:
        raise RuntimeError("Validation source differs from the original inference")
    grouped = defaultdict(list)
    with validation.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                record = json.loads(line)
                grouped[(record["h"]["id"], record["t"]["id"])].append(record)
    for bag in manifest:
        records = grouped[tuple(bag["pair_ids"])]
        count = len(records)
        indices = list(range(count)) if count <= 20 else [
            (2 * i * (count - 1) + 19) // 38 for i in range(20)]
        selected = [records[i] for i in indices]
        if (bag["full_bag_size"] != count or bag["selected_indices_zero_based"] != indices
                or bag["head"] != records[0]["h"]["name"] or bag["tail"] != records[0]["t"]["name"]
                or bag["sentences"] != [r["text"] for r in selected]
                or bag["entity_positions"] != [{"head": r["h"]["pos"], "tail": r["t"]["pos"]} for r in selected]
                or bag["ds_positive_labels"] != sorted({r["relation"] for r in records} - {"NA"})
                or set(bag["ds_positive_labels"]) - set(labels)):
            raise RuntimeError("Frozen manifest differs from the original validation records")
    return manifest, settings, labels


def bag_path(run, model, bag):
    token = hashlib.sha256("|".join(bag["pair_ids"]).encode()).hexdigest()[:20]
    return Path(run) / MODEL_SPECS[model]["slug"] / "bags" / (token + ".json")


def validate_result(row, bag, model, fingerprint, labels):
    """Validate complete sentence scores, exact supplied spans and bag maxima."""
    if (row.get("model") != MODEL_SPECS[model]["name"]
            or row.get("pair_ids") != bag["pair_ids"]
            or row.get("run_fingerprint") != fingerprint):
        raise RuntimeError(f"Mismatched saved {model} result for {bag['pair_ids']}")
    if row.get("status") != "valid":
        return False
    if (row.get("selected_indices_zero_based") != bag["selected_indices_zero_based"]
            or row.get("full_bag_size") != bag["full_bag_size"]
            or row.get("sentences_processed") != len(bag["sentences"])
            or len(row.get("evidence", [])) != len(bag["sentences"])):
        raise RuntimeError("Saved result has different or incomplete selected evidence")
    maxima = {label: 0.0 for label in labels}
    for i, evidence in enumerate(row["evidence"]):
        if (evidence.get("selected_sentence_index") != i
                or evidence.get("source_record_index") != bag["selected_indices_zero_based"][i]
                or evidence.get("head_char_span") != bag["entity_positions"][i]["head"]
                or evidence.get("tail_char_span") != bag["entity_positions"][i]["tail"]):
            raise RuntimeError("Saved sentence identity or directed mention spans differ")
        values = evidence.get("relation_scores", {})
        if set(values) != set(labels) or any(
            type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
            for v in values.values()
        ):
            raise RuntimeError("Invalid sentence score inventory")
        for label in labels:
            maxima[label] = max(maxima[label], values[label])
    if row.get("relation_score_max") != maxima:
        raise RuntimeError("Bag maxima do not reproduce the saved sentence scores")
    if row.get("relations") != sorted(k for k, v in maxima.items() if v > .5):
        raise RuntimeError("Primary saved relation set disagrees with strict cutoff")
    if type(row.get("wall_s")) not in (int, float) or not math.isfinite(row["wall_s"]) or row["wall_s"] < 0:
        raise RuntimeError("Invalid inference timing")
    attempts = row.get("total_attempt_wall_s", row["wall_s"])
    if type(attempts) not in (int, float) or not math.isfinite(attempts) or attempts < row["wall_s"]:
        raise RuntimeError("Invalid accumulated attempt timing")
    return True


def configuration_inventory(models):
    """Fixed grid: singles, all two-member unions/intersections, and 2-of-3 vote."""
    configs = []
    for model in models:
        configs.extend({"rule": "single", "members": [model], "cutoffs": [c]}
                       for c in CUTOFFS)
    for pair in itertools.combinations(models, 2):
        for cutoffs in itertools.product(CUTOFFS, repeat=2):
            configs.extend({"rule": rule, "members": list(pair), "cutoffs": list(cutoffs)}
                           for rule in ("intersection", "union"))
    if len(models) == 3:
        configs.extend({"rule": "majority_2_of_3", "members": list(models), "cutoffs": list(c)}
                       for c in itertools.product(CUTOFFS, repeat=3))
    for config in configs:
        config["configuration"] = config["rule"] + "|" + ";".join(
            f"{m}@{c:g}" for m, c in zip(config["members"], config["cutoffs"], strict=True))
    return configs


def combine_predictions(config, thresholded):
    import numpy as np
    votes = [thresholded[m][c] for m, c in zip(config["members"], config["cutoffs"], strict=True)]
    if config["rule"] == "single":
        return votes[0]
    if config["rule"] == "intersection":
        return np.logical_and.reduce(votes)
    if config["rule"] == "union":
        return np.logical_or.reduce(votes)
    if config["rule"] == "majority_2_of_3" and len(votes) == 3:
        return sum(v.astype(np.uint8) for v in votes) >= 2
    raise ValueError("Unknown combination rule")


def array_metrics(gold, prediction):
    import numpy as np
    tp, fp, fn = (int(np.sum(gold & prediction)), int(np.sum(~gold & prediction)),
                  int(np.sum(gold & ~prediction)))
    negatives = ~gold.any(axis=1)
    return {"bags": len(gold), "TP": tp, "FP": fp, "FN": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "exact_match_bags": int(np.sum(np.all(gold == prediction, axis=1))),
            "negative_bags": int(negatives.sum()),
            "negative_bags_with_fp": int(np.sum(negatives & prediction.any(axis=1)))}


def build_comparison(*, require_glidre=True, model_run_overrides=None, save=True):
    """Read complete caches; keep full distant and reviewed-text references separate."""
    import numpy as np
    from reviewed_validation_reference import load_reviewed_reference
    manifest, frozen, labels = frozen_inputs()
    reviewed_manifest = read_json(ROOT / "data/pilot_runs/reviewed_validation_30_encoder_v1_20261001/manifest.json")
    reviewed_gold, reference_metadata = load_reviewed_reference(ROOT, reviewed_manifest)
    index = {tuple(b["pair_ids"]): i for i, b in enumerate(manifest)}
    reviewed_indices = [index[tuple(b["pair_ids"])] for b in reviewed_manifest]
    for item, i in zip(reviewed_manifest, reviewed_indices, strict=True):
        if any(item[key] != manifest[i][key] for key in ("head", "tail", "sentences")):
            raise RuntimeError("Reviewed texts differ from the inference evidence")
    runs = {"base": RELEX_RUN, "large": RELEX_RUN, "glidre": GLIDRE_RUN}
    runs.update(model_run_overrides or {})
    models = ["base", "large"]
    glidre_settings = None
    if (GLIDRE_RUN / "settings.json").is_file():
        glidre_settings = read_json(GLIDRE_RUN / "settings.json")
        if hashlib.sha256(json.dumps(glidre_settings["settings"], sort_keys=True).encode()).hexdigest() != glidre_settings["fingerprint"]:
            raise RuntimeError("Invalid GLiDRE inference fingerprint")
        if (glidre_settings["settings"]["source_run_fingerprint"] != frozen["fingerprint"]
                or glidre_settings["settings"]["source_manifest_sha256"] != file_hash(RELEX_RUN / "manifest.json")):
            raise RuntimeError("GLiDRE inputs differ from the frozen comparison inputs")
        models.append("glidre")
    elif require_glidre:
        raise RuntimeError("GLiDRE has no prepared experiment; run its enabled stages first")
    arrays, wall, cost, cache_hashes, reviewed_rows = {}, {}, {}, {}, {}
    for model in models:
        fingerprint = glidre_settings["fingerprint"] if model == "glidre" else frozen["fingerprint"]
        values, seconds = np.empty((len(manifest), 24)), []
        digest, detail = hashlib.sha256(), {}
        incomplete = False
        for i, bag in enumerate(manifest):
            path = bag_path(runs[model], model, bag)
            if not path.is_file():
                if model == "glidre" and not require_glidre:
                    incomplete = True
                    break
                raise RuntimeError(f"Incomplete {model} cache: {path}")
            raw = path.read_bytes()
            row = json.loads(raw)
            if not validate_result(row, bag, model, fingerprint, labels):
                if model == "glidre" and not require_glidre:
                    incomplete = True
                    break
                raise RuntimeError(f"Failed or incomplete {model} result: {path}")
            values[i] = [row["relation_score_max"][label] for label in labels]
            seconds.append(float(row.get("total_attempt_wall_s", row["wall_s"])))
            digest.update(str(path.relative_to(ROOT)).encode())
            digest.update(hashlib.sha256(raw).digest())
            if i in reviewed_indices:
                detail[i] = row
        if incomplete:
            print("GLiDRE is incomplete and execution is disabled; reporting only the two complete relex members.", flush=True)
            continue
        arrays[model], wall[model], reviewed_rows[model] = values, np.asarray(seconds), detail
        cost[model], cache_hashes[model] = float(sum(seconds)), digest.hexdigest()
    models = list(arrays)
    runtime = read_json(GLIDRE_RUN / "model_runtime.json") if "glidre" in models else {}
    preflight = read_json(GLIDRE_RUN / "offline_preflight.json") if "glidre" in models else {}
    if "glidre" in models and (preflight.get("success") is not True
            or preflight.get("run_fingerprint") != glidre_settings["fingerprint"]):
        raise RuntimeError("Complete GLiDRE scores require a matching successful compatibility preflight")
    label_setup_s = sum(s["label_encoding_wall_s"] for s in runtime.get("sessions", []))
    if "glidre" in models:
        cost["glidre"] += label_setup_s
    ds = np.array([[label in b["ds_positive_labels"] for label in labels] for b in manifest], dtype=bool)
    reviewed = np.array([[label in reviewed_gold[b["sample_id"]] for label in labels]
                         for b in reviewed_manifest], dtype=bool)
    thresholded = {m: {c: arrays[m] > c for c in CUTOFFS} for m in models}
    configs = configuration_inventory(models)
    rows, cohort_rows, per_relation = [], [], []
    cohorts = {"distant_NA": ~ds.any(axis=1), "distant_positive": ds.any(axis=1),
               "singleton": np.array([b["full_bag_size"] == 1 for b in manifest]),
               "size_2_to_20": np.array([2 <= b["full_bag_size"] <= 20 for b in manifest]),
               "size_above_20": np.array([b["full_bag_size"] > 20 for b in manifest])}
    for config in configs:
        pred = combine_predictions(config, thresholded)
        info = {**config, "members_count": len(config["members"]),
                "total_member_inference_s": sum(cost[m] for m in config["members"])}
        for reference, gold, predicted in (
            ("all_validation_distant_labels", ds, pred),
            ("reviewed_30_text", reviewed, pred[reviewed_indices]),
        ):
            rows.append({**info, "reference": reference, **array_metrics(gold, predicted)})
        if all(c == .5 for c in config["cutoffs"]):
            for name, mask in cohorts.items():
                cohort_rows.append({**info, "cohort": name, **array_metrics(ds[mask], pred[mask])})
            for j, label in enumerate(labels):
                per_relation.append({"configuration": config["configuration"], "relation": label,
                                     **array_metrics(ds[:, j:j+1], pred[:, j:j+1])})
    bags = []
    primary = [c for c in configs if all(t == .5 for t in c["cutoffs"])]
    diagnostic = [c for c in configs if c["rule"] == "single"] + [c for c in primary if c["rule"] != "single"]
    for item, i in zip(reviewed_manifest, reviewed_indices, strict=True):
        gold = set(reviewed_gold[item["sample_id"]])
        predictions = {}
        for config in diagnostic:
            # Combine only this bag for diagnostic views, without storing full predictions.
            small = {m: {c: v[i:i+1] for c, v in thresholded[m].items()} for m in models}
            predicted = combine_predictions(config, small)[0]
            actual = {labels[j] for j in np.flatnonzero(predicted)}
            predictions[config["configuration"]] = {
                "relations": sorted(actual), "TP": sorted(actual & gold),
                "FP": sorted(actual - gold), "FN": sorted(gold - actual)}
        contributions = []
        for left, right in itertools.combinations(models, 2):
            a = {labels[j] for j in np.flatnonzero(thresholded[left][.5][i])}
            b = {labels[j] for j in np.flatnonzero(thresholded[right][.5][i])}
            contributions.append({"left": left, "right": right,
                                  "left_only_TP": sorted((a - b) & gold),
                                  "left_only_FP": sorted((a - b) - gold),
                                  "right_only_TP": sorted((b - a) & gold),
                                  "right_only_FP": sorted((b - a) - gold)})
        bags.append({**{k: item[k] for k in ("sample_id", "pair_ids", "head", "tail", "sentences")},
                     "confirmed_relations": sorted(gold),
                     "distant_relations": manifest[i]["ds_positive_labels"],
                     "predictions": predictions, "member_contributions_at_0_5": contributions,
                     "scores": {m: {"bag_maxima": reviewed_rows[m][i]["relation_score_max"],
                                    "evidence": reviewed_rows[m][i]["evidence"]} for m in models}})
    result = {"analysis_version": "relex_glidre_fixed_grid_v1_20261006",
              "source_run_fingerprint": frozen["fingerprint"],
              "glidre_run_fingerprint": glidre_settings["fingerprint"] if "glidre" in models else None,
              "reviewed_reference_metadata": reference_metadata, "models": models,
              "configurations_per_reference": len(configs), "cutoffs": list(CUTOFFS),
              "rows": rows, "primary_cohorts": cohort_rows, "primary_per_relation": per_relation,
              "score_cache_sha256_by_model": cache_hashes,
              "model_run_directories": {m: str(Path(runs[m]).relative_to(ROOT)) for m in models},
              "timing": {"inference_s_by_model": cost, "glidre_label_setup_s": label_setup_s,
                         "scope": "Full-validation saved bag timings; GLiDRE includes previous failed attempts and session label encoding. Loading/preflight excluded. Ensemble cost sums all members.",
                         "bag_mean_s": {m: float(v.mean()) for m, v in wall.items()},
                         "bag_p95_s": {m: float(np.quantile(v, .95)) for m, v in wall.items()}},
              "reference_warning": "Distant validation labels and selected reused reviewed-30 annotations are separate development references. No automatic model selection or official test evaluation.",
              "glidre_runtime": runtime, "glidre_preflight": preflight}
    if save:
        name = "validation_relex_glidre_comparison" if "glidre" in models else "validation_relex_current_annotations"
        destination = ROOT / "results" / f"{name}_{reference_metadata['labels_sha256'][:12]}.json"
        if destination.exists() and read_json(destination) != result:
            archive = ROOT / "results/archive" / f"{destination.stem}_{file_hash(destination)[:12]}.json"
            if not archive.exists():
                archive.parent.mkdir(parents=True, exist_ok=True)
                archive.write_bytes(destination.read_bytes())
        save_json(destination, result)
        result["published_path"] = str(destination.relative_to(ROOT))
    return result, bags
