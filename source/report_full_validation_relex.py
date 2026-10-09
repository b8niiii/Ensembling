"""Analysis-only report for the frozen supplied-entity full-validation experiment.

Scores cached outputs without loading checkpoints or modifying inference code,
settings, preflight, or per-bag results. NYT10m train-pair membership is excluded
from reporting because it does not identify these checkpoints' training inputs.
The selected text-reviewed 30 remain a distinct diagnostic reference.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

from run_full_validation_relex_given_entities import (
    ADAPTER, MODELS, PREVIOUS_RUN, REL2ID, RELATION_THRESHOLD, ROOT, RUN,
    bag_result_path, save_json, score, sha256, valid_saved,
)
from reviewed_validation_reference import load_reviewed_reference

REPORT_VERSION = "validation_cohorts_v3_annotations_20261006"
ARCHIVE = RUN / "report_archive_before_annotation_revision_20261006"


def read_frozen_inputs() -> tuple[list[dict], dict]:
    """Require the original inference fingerprint and unchanged implementation."""
    settings = json.loads((RUN / "settings.json").read_text(encoding="utf-8"))
    frozen = settings["settings"]
    fingerprint = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()
    if fingerprint != settings["fingerprint"]:
        raise RuntimeError("Frozen settings fingerprint is invalid")
    implementation = {
        "runner_sha256": ROOT / "source/run_full_validation_relex_given_entities.py",
        "adapter_sha256": ADAPTER,
        "label_module_sha256": ROOT / "source/run_extended_gliner_relex.py",
    }
    for key, path in implementation.items():
        if sha256(path) != frozen[key]:
            raise RuntimeError(f"Frozen inference implementation changed: {path}")
    if list(MODELS) != frozen["models"] or RELATION_THRESHOLD != frozen["primary_relation_threshold"]:
        raise RuntimeError("Model definitions or primary cutoff differ from frozen inference")
    if sha256(REL2ID) != frozen["rel2id_sha256"]:
        raise RuntimeError("Relation inventory differs from frozen inference")
    manifest = json.loads((RUN / "manifest.json").read_text(encoding="utf-8"))
    if len(manifest) != frozen["manifest_count"]:
        raise RuntimeError("Manifest size differs from frozen inference")
    return manifest, settings


def archive_existing_report(reference_metadata: dict) -> None:
    """Preserve the previous report once; leave all prediction files untouched."""
    summary_path = RUN / "summary.json"
    if not summary_path.is_file():
        return
    previous = json.loads(summary_path.read_text(encoding="utf-8"))
    old_metadata = previous.get("report_metadata", {})
    if (old_metadata.get("version") == REPORT_VERSION
            and old_metadata.get("reviewed_reference") == reference_metadata):
        return
    reference_hash = old_metadata.get("reviewed_reference", {}).get("labels_sha256", "original")
    archive = ARCHIVE / reference_hash[:16]
    archive.mkdir(parents=True, exist_ok=True)
    for name in ("summary.json", "disagreements.json", "per_relation_distant_scores.csv"):
        source_path = RUN / name
        destination = archive / name
        if source_path.is_file() and not destination.exists():
            shutil.copy2(source_path, destination)


def report() -> None:
    """Score full validation distant labels and reviewed 30 separately."""
    manifest, settings = read_frozen_inputs()
    fingerprint = settings["fingerprint"]
    from relex_given_entities import labels_at_threshold
    predictions, score_cache, wall_s = {}, {}, {}
    for spec in MODELS:
        by_pair, by_pair_scores, total_wall = {}, {}, 0.0
        for bag in manifest:
            path = bag_result_path(spec, bag)
            if not valid_saved(path, spec, bag, fingerprint):
                raise RuntimeError(f"Incomplete result for {spec['name']} and {bag['pair_ids']}")
            row = json.loads(path.read_text(encoding="utf-8"))
            pair = tuple(bag["pair_ids"])
            by_pair_scores[pair] = row["relation_score_max"]
            by_pair[pair] = labels_at_threshold(by_pair_scores[pair], RELATION_THRESHOLD)
            total_wall += row["wall_s"]
        predictions[spec["slug"]] = by_pair
        score_cache[spec["slug"]] = by_pair_scores
        wall_s[spec["slug"]] = round(total_wall, 2)

    base = predictions["gliner_relex_base"]
    large = predictions["gliner_relex_large"]
    pairs = [tuple(b["pair_ids"]) for b in manifest]
    systems = {
        "relex_base": base,
        "relex_large": large,
        "intersection": {p: sorted(set(base[p]) & set(large[p])) for p in pairs},
        "union": {p: sorted(set(base[p]) | set(large[p])) for p in pairs},
    }
    ds_gold = {tuple(b["pair_ids"]): b["ds_positive_labels"] for b in manifest}
    cohorts = {
        "all_validation_distant_labels": manifest,
        "distant_NA_only": [b for b in manifest if not b["ds_positive_labels"]],
        "distant_positive": [b for b in manifest if b["ds_positive_labels"]],
        "singleton_bags": [b for b in manifest if b["full_bag_size"] == 1],
        "bags_with_2_to_20_records": [b for b in manifest if 2 <= b["full_bag_size"] <= 20],
        "bags_above_20_records": [b for b in manifest if b["full_bag_size"] > 20],
    }
    summary = {
        "report_metadata": {
            "version": REPORT_VERSION,
            "script_sha256": sha256(Path(__file__)),
            "excluded_cohorts": ["pair_seen_in_train", "pair_not_seen_in_train"],
            "exclusion_reason": "NYT10m split membership does not establish checkpoint training exposure",
            "reviewed_reference_role": "Separate text-annotation diagnostic on a selected 30-bag sample; not a representative benchmark",
        },
        "run_fingerprint": fingerprint,
        "inference_mode": "supplied_dataset_spans_no_entity_recognition",
        "primary_relation_threshold": RELATION_THRESHOLD,
        "reference_warning": "Full validation relation labels are distant supervision, not manual gold",
        "inference_wall_s_by_model": wall_s,
        "distant_label_scores": {name: {system: score(bags, ds_gold, pred)
                                       for system, pred in systems.items()}
                                 for name, bags in cohorts.items()},
    }

    # Read current confirmed annotations; preserve the experiment's original snapshot.
    reviewed_dir = ROOT / "data/pilot_runs/reviewed_validation_30_encoder_v1_20261001"
    reviewed_manifest = json.loads((reviewed_dir / "manifest.json").read_text(encoding="utf-8"))
    reviewed_labels, reference_metadata = load_reviewed_reference(ROOT, reviewed_manifest)
    summary["report_metadata"]["reviewed_reference"] = reference_metadata
    full_by_pair = {tuple(b["pair_ids"]): b for b in manifest}
    reviewed_bags, human_gold = [], {}
    for item in reviewed_manifest:
        pair = tuple(item["pair_ids"])
        full_bag = full_by_pair[pair]
        if full_bag["sentences"] != item["sentences"]:
            raise RuntimeError(f"Reviewed and full-run selected texts differ for {item['sample_id']}")
        reviewed_bags.append(full_bag)
        human_gold[pair] = reviewed_labels[item["sample_id"]]
    summary["reviewed_30_text_scores"] = {
        system: score(reviewed_bags, human_gold, pred) for system, pred in systems.items()
    }

    # These fixed diagnostic cutoffs reuse all saved relation scores. No model
    # inference or automatic threshold selection occurs during reporting.
    summary["threshold_diagnostics"] = {}
    for cutoff in (0.5, 0.7, 0.9):
        at_cutoff = {slug: {pair: labels_at_threshold(values, cutoff)
                            for pair, values in by_pair.items()}
                     for slug, by_pair in score_cache.items()}
        b, l = at_cutoff["gliner_relex_base"], at_cutoff["gliner_relex_large"]
        combined = {"relex_base": b, "relex_large": l,
                    "intersection": {pair: sorted(set(b[pair]) & set(l[pair])) for pair in pairs},
                    "union": {pair: sorted(set(b[pair]) | set(l[pair])) for pair in pairs}}
        summary["threshold_diagnostics"][str(cutoff)] = {
            "all_validation_distant_labels": {system: score(manifest, ds_gold, pred)
                                               for system, pred in combined.items()},
            "reviewed_30_text": {system: score(reviewed_bags, human_gold, pred)
                                 for system, pred in combined.items()},
        }

    # Retain the prior completed experiment as a distinct protocol comparison.
    prior_summary = PREVIOUS_RUN / "summary.json"
    if prior_summary.is_file():
        prior_settings = json.loads((PREVIOUS_RUN / "settings.json").read_text())
        prior = json.loads(prior_summary.read_text())
        previous = prior_settings["settings"]
        current = settings["settings"]
        for key in ("validation_sha256", "train_sha256", "rel2id_sha256", "bag_cap", "selection", "label_name", "entity_labels", "models", "device", "versions"):
            if previous[key] != current[key]:
                raise RuntimeError(f"Prior joint experiment is not comparable: {key}")
        if (previous["relation_threshold"] != RELATION_THRESHOLD or
                prior["run_fingerprint"] != prior_settings["fingerprint"]):
            raise RuntimeError("Prior report settings do not match the primary comparison")
        summary["previous_joint_run"] = {
            "run_directory": str(PREVIOUS_RUN.relative_to(ROOT)),
            "run_fingerprint": prior["run_fingerprint"], "entity_threshold": previous["entity_threshold"],
            "relation_threshold": previous["relation_threshold"],
            "all_validation_distant_labels": prior["distant_label_scores"]["all_validation_distant_labels"],
            "reviewed_30_text": prior["reviewed_30_text_scores"],
            "reviewed_reference_scope": "Historical original annotations; not rescored with the 6 October inference revision.",
            "inference_wall_s_by_model": prior["inference_wall_s_by_model"],
        }

    disagreements = []
    for bag in manifest:
        pair = tuple(bag["pair_ids"])
        if set(base[pair]) != set(large[pair]):
            disagreements.append({
                "pair_ids": bag["pair_ids"], "head": bag["head"], "tail": bag["tail"],
                "full_bag_size": bag["full_bag_size"],
                "ds_positive_labels": bag["ds_positive_labels"],
                "base_relations": base[pair], "large_relations": large[pair],
                "sentences": bag["sentences"],
            })
    summary["disagreeing_bags"] = len(disagreements)
    archive_existing_report(reference_metadata)
    save_json(RUN / "summary.json", summary)
    save_json(RUN / "disagreements.json", disagreements)

    rows = []
    for system, pred in systems.items():
        for relation in sorted(set(json.loads(REL2ID.read_text(encoding="utf-8"))) - {"NA"}):
            tp = fp = fn = 0
            for pair in pairs:
                g, p = relation in ds_gold[pair], relation in pred[pair]
                tp += g and p
                fp += not g and p
                fn += g and not p
            rows.append({"system": system, "relation": relation, "TP": tp, "FP": fp,
                         "FN": fn, "micro_f1_for_relation":
                         round(2 * tp / (2 * tp + fp + fn), 4) if 2 * tp + fp + fn else 0.0})
    csv_path = RUN / "per_relation_distant_scores.csv"
    temporary_csv = csv_path.with_suffix(".csv.tmp")
    with temporary_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary_csv.replace(csv_path)

    print("Full-validation scores below use noisy distant labels; reviewed-30 scores use text labels.")
    print(json.dumps({"all_validation": summary["distant_label_scores"]["all_validation_distant_labels"],
                      "reviewed_30": summary["reviewed_30_text_scores"],
                      "disagreeing_bags": len(disagreements),
                      "inference_wall_s_by_model": wall_s}, indent=2), flush=True)
    print(f"Saved report and disagreements in {RUN}", flush=True)


if __name__ == "__main__":
    report()
